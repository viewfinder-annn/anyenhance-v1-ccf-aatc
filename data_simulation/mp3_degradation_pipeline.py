import os
import torch
import torchaudio
import subprocess
import tempfile
import random
import numpy as np
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def match2(x, d):
    """
    Calculates the delay between two waveforms and returns it.
    This is crucial for aligning the original audio with the processed audio.
    """
    assert x.dim()==2, x.shape
    assert d.dim()==2, d.shape
    minlen = min(x.shape[-1], d.shape[-1])
    x, d = x[:,0:minlen], d[:,0:minlen]
    Fx = torch.fft.rfft(x, dim=-1)
    Fd = torch.fft.rfft(d, dim=-1)
    Phi = Fd*Fx.conj()
    Phi = Phi / (Phi.abs() + 1e-3)
    Phi[:,0] = 0
    tmp = torch.fft.irfft(Phi, dim=-1)
    tau = torch.argmax(tmp.abs(),dim=-1).tolist()
    return tau

def apply_mp3_codec_ffmpeg(wav: torch.Tensor, sr: int, bitrate: int) -> torch.Tensor:
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_in_file, \
         tempfile.NamedTemporaryFile(suffix=".mp3") as tmp_mp3_file, \
         tempfile.NamedTemporaryFile(suffix=".wav") as tmp_out_file:

        torchaudio.save(tmp_in_file.name, wav, sr)

        bitrate_k = bitrate // 1000

        subprocess.run(
            ['ffmpeg', '-y', '-i', tmp_in_file.name, '-b:a', f'{bitrate_k}k', tmp_mp3_file.name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        subprocess.run(
            ['ffmpeg', '-y', '-i', tmp_mp3_file.name, tmp_out_file.name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        wav_processed, _ = torchaudio.load(tmp_out_file.name)
        return wav_processed

CODEC_CONFIG = [
    {
        'name': 'MP3',
        'backend': 'ffmpeg',
        'params': {
            'bitrates': [24000, 32000, 48000, 64000, 96000, 128000],
        }
    },
]

def simulate_and_align_random_codec(wav: torch.Tensor, sr: int) -> torch.Tensor:
    """
    Selects a random codec, applies it, and aligns the result with the original.
    """
    codec_choice = random.choice(CODEC_CONFIG)
    wav_processed = None

    if codec_choice['backend'] == 'ffmpeg':
        bitrate = random.choice(codec_choice['params']['bitrates'])
        wav_processed = apply_mp3_codec_ffmpeg(wav, sr, bitrate)

    if wav_processed is None:
        raise RuntimeError("Codec processing failed to produce an output.")

    # Pad or truncate to match original length
    if wav_processed.shape[-1] >= wav.shape[-1]:
        wav_processed = wav_processed[..., :wav.shape[-1]]
    else:
        wav_processed = torch.nn.functional.pad(wav_processed, (0, wav.shape[-1] - wav_processed.shape[-1]))
    
    tau = match2(wav, wav_processed)
    wav_aligned = torch.roll(wav_processed, -tau[0], -1)

    return wav_aligned

def process_single_item(speech_path: str, dst: str, sr: int):
    try:
        wav, original_sr = torchaudio.load(speech_path)
        
        if original_sr != sr:
            resampler = torchaudio.transforms.Resample(original_sr, sr)
            wav = resampler(wav)
            
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
            
        # Get the aligned, degraded waveform
        encoded_and_aligned_wav = simulate_and_align_random_codec(wav, sr=sr)
        
        # Save the result as a WAV file
        output_filename = os.path.basename(speech_path)
        output_path = os.path.join(dst, 'encoded', output_filename)
        
        torchaudio.save(output_path, encoded_and_aligned_wav, sr)
        
        return {"status": "Success", "file": speech_path}
    except Exception as e:
        return {"status": "Error", "file": speech_path, "message": str(e)}

def read_scp(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scp file not found: {path}")
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Process audio files based on a scp file.")
    parser.add_argument("--speech_scp", type=str, required=True, help="Path to the scp file for speech audio.")
    parser.add_argument("--dst_dir", type=str, required=True, help="Destination directory.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel worker threads.")
    parser.add_argument("--sr", type=int, default=44100, help="Target sample rate.")
    
    args = parser.parse_args()

    os.makedirs(args.dst_dir, exist_ok=True)
    os.makedirs(os.path.join(args.dst_dir, 'encoded'), exist_ok=True)

    speech_paths = read_scp(args.speech_scp)

    items_to_process = speech_paths

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(
                process_single_item, 
                speech_path, 
                args.dst_dir, 
                args.sr
            ) 
            for speech_path in items_to_process
        ]
        
        for future in tqdm(as_completed(futures), total=len(items_to_process), desc="Encoding and Aligning Audio"):
            result = future.result()
            if result["status"] == "Error":
                tqdm.write(f"Failed to process {result['file']}: {result['message']}")

    print("All tasks completed.")

if __name__ == "__main__":
    main()