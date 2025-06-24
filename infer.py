import torchaudio
import torch
import os
import json5
import dac
from anyenhance import AnyEnhance_v1, MaskGitTransformer, AudioEncoder

os.environ["PYTHONIOENCODING"] = "utf-8"

def pad_or_truncate(x, length=512*256):
    if x.size(-1) < length:
        # x = torch.nn.functional.pad(x, (0, length - x.size(-1)))
        repeat_times = length // x.size(-1) + 1
        x = x.repeat(1, repeat_times)
        x = x[..., :length]
    elif x.size(-1) > length:
        x = x[..., :length]
    return x

def get_model(config, device):
    # Load DAC models
    dac_model = dac.DAC.load(config['dac_path']).to(device)
    dac_model.to(device)
    dac_model.eval()
    dac_model.requires_grad_(False)

    # Initialize transformer and audio encoder
    transformer_config = config['MaskGitTransformer']
    audio_encoder_config = config['AudioEncoder']
    transformer = MaskGitTransformer(**transformer_config)
    audio_encoder = AudioEncoder(**audio_encoder_config)

    # Initialize AnyEnhance_v1 model
    maskgit_config = config['AnyEnhance_v1']
    model_class = AnyEnhance_v1
    model = model_class(
        vq_model=dac_model,
        transformer=transformer,
        audio_encoder=audio_encoder,
        **maskgit_config
    ).to(device)
    
    print(f"model Params: {round(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6, 2)}M")

    return model


def load_model(model_path, config, device):
    model_state_dict = torch.load(model_path)
    if "module" in list(model_state_dict.keys())[0]:
        print("New model detected. Loading new model.")
        model = get_model(
            config["model"], device
        )  # get_model needs to be defined or imported appropriately
        model = torch.nn.DataParallel(model)
        model.load_state_dict(model_state_dict, strict=False)
        model = model.module
    else:
        print("No MODULE.")
        model = get_model(config["model"], device)
        model.load_state_dict(model_state_dict, strict=False)
    model.eval()
    return model


def smooth_audio_transition(audio_chunks, overlap=1024):
    """
    Smoothly transition between audio chunks by hann windowing the overlap region.
    """
    if len(audio_chunks) == 0:
        return torch.Tensor([])

    window = torch.hann_window(overlap * 2, periodic=True).to(audio_chunks[0].device)

    result = audio_chunks[0]

    for i in range(1, len(audio_chunks)):
        prev_chunk_length = result.shape[-1]
        prev_overlap_len = min(overlap, prev_chunk_length)

        previous_end = result[:, -prev_overlap_len:].clone()

        curr_chunk_length = audio_chunks[i].shape[-1]
        curr_overlap_len = min(overlap, curr_chunk_length)

        previous_end *= (
            window[overlap : overlap + prev_overlap_len]
            if prev_overlap_len == overlap
            else window[-prev_overlap_len:]
        )
        current_start = audio_chunks[i][:, :curr_overlap_len].clone()
        current_start *= window[:curr_overlap_len]

        min_len = min(prev_overlap_len, curr_overlap_len)
        transition = previous_end[:, -min_len:] + current_start[:, :min_len]

        result[:, -min_len:] = transition

        result = torch.cat((result, audio_chunks[i][:, curr_overlap_len:]), dim=1)

    return result


def process_single_audio(
    model,
    signal,
    device,
    window_size,
    overlap=1024,
    timesteps=20,
    cond_scale=1,
):
    """
    Enhance a single audio signal based on the task type and optional prompt.
    """
    
    original_length = signal.shape[-1]
    enhanced_audio = []
    start = 0
    hop_size = window_size - overlap

    while start < signal.shape[-1]:
        if start + window_size >= signal.shape[-1]:
            segment = signal[:, -window_size:]
            is_last_segment = True
        else:
            end = start + window_size
            segment = signal[:, start:end]
            is_last_segment = False

        if segment.shape[-1] < window_size:
            is_padding = True
            valid_length = segment.shape[-1]
            segment = pad_or_truncate(segment, length=window_size)
        else:
            is_padding = False

        with torch.no_grad():
            ids, output_segment = model.generate(
                segment.unsqueeze(0),
                timesteps=timesteps,
                cond_scale=cond_scale,
            )
            output_segment = output_segment.squeeze(0)

            if is_last_segment:
                if is_padding:
                    output_segment = output_segment[:, :valid_length]
                last_valid_length = signal.shape[-1] - start
                if last_valid_length == 0:
                    last_valid_length = window_size
                output_segment = output_segment[:, -last_valid_length:]

            enhanced_audio.append(output_segment)

        start += hop_size

    enhanced_signal = smooth_audio_transition(enhanced_audio, overlap=overlap)
    enhanced_signal = enhanced_signal[:, :original_length]

    return enhanced_signal


def infer_single_audio(
    model,
    audio_path,
    output_path,
    device,
    timesteps=20,
    cond_scale=1,
):
    os.makedirs(output_path, exist_ok=True)

    signal, sr = torchaudio.load(audio_path)
    signal = torch.mean(signal, dim=0, keepdim=True)
    signal = signal.to(device)
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=44100).to(device)
    signal = resampler(signal)

    window_size = (
        512 * model.seq_len
    )
    overlap = 1024

    enhanced_signal = process_single_audio(
        model,
        signal,
        device,
        window_size=window_size,
        overlap=overlap,
        timesteps=timesteps,
        cond_scale=cond_scale,
    )

    output_file_path = os.path.join(output_path, os.path.basename(audio_path))
    torchaudio.save(output_file_path, enhanced_signal.detach().cpu(), 44100)


if __name__ == "__main__":
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser(
        description="Generate enhanced audio outputs from given audio files."
    )
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to the model config file."
    )
    parser.add_argument(
        "--ckpt_path", type=str, required=True, help="Path to the model weights file."
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input_file", type=str, help="Path to a single audio file to process."
    )
    input_group.add_argument(
        "--input_folder", type=str, help="Path to a folder containing audio files to process."
    )
    
    parser.add_argument(
        "--output_folder", type=str, default="./output/", help="Path to the output folder."
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run inference on."
    )
    parser.add_argument(
        "--timesteps", type=int, default=20, help="Number of timesteps to generate"
    )
    parser.add_argument("--cond_scale", type=float, default=1, help="CFG factor.")

    args = parser.parse_args()

    print(f"Loading model from checkpoint: {args.ckpt_path}")
    model = load_model(args.ckpt_path, json5.load(open(args.config_path)), args.device)
    
    files_to_process = []
    if args.input_file:
        if not os.path.exists(args.input_file):
            print(f"Error: Input file not found at {args.input_file}")
            exit(1)
        files_to_process.append(args.input_file)
    elif args.input_folder:
        if not os.path.isdir(args.input_folder):
            print(f"Error: Input folder not found at {args.input_folder}")
            exit(1)
            
        print(f"Scanning for audio files in: {args.input_folder}")
        supported_extensions = ('.wav', '.mp3', '.flac')
        files_to_process = [os.path.join(args.input_folder, f) for f in os.listdir(args.input_folder) if 
                            os.path.isfile(os.path.join(args.input_folder, f)) and f.lower().endswith(supported_extensions)]
    
    if not files_to_process:
        print("No audio files found to process.")
        exit(0)

    print(f"Found {len(files_to_process)} audio file(s) to process.")

    # 使用 tqdm 循环处理文件列表
    for audio_path in tqdm(files_to_process, desc="Processing audio files"):
        try:
            infer_single_audio(
                model,
                audio_path,
                args.output_folder,
                args.device,
                args.timesteps,
                args.cond_scale,
            )
        except Exception as e:
            print(f"Failed to process {os.path.basename(audio_path)}: {e}")
            continue
            
    print("\nInference complete.")