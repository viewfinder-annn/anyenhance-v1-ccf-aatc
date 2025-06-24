# modified from urgent challenge https://github.com/urgent-challenge/urgent2024_challenge/blob/main/simulation/simulate_data_from_param.py
import sys

import librosa
import numpy as np
import scipy
import soundfile as sf
import random
from tqdm.contrib.concurrent import process_map
import torchaudio
import pedalboard as pd
from pedalboard import Pedalboard, HighShelfFilter
import math
import os

import io

def framing(
    x,
    frame_length: int = 512,
    frame_shift: int = 256,
    centered: bool = True,
    padded: bool = True,
):
    if x.size == 0:
        raise ValueError("Input array size is zero")
    if frame_length < 1:
        raise ValueError("frame_length must be a positive integer")
    if frame_length > x.shape[-1]:
        raise ValueError("frame_length is greater than input length")
    if 0 >= frame_shift:
        raise ValueError("frame_shift must be greater than 0")

    if centered:
        pad_shape = [(0, 0) for _ in range(x.ndim - 1)] + [
            (frame_length // 2, frame_length // 2)
        ]
        x = np.pad(x, pad_shape, mode="constant", constant_values=0)

    if padded:
        # Pad to integer number of windowed segments
        # I.e make x.shape[-1] = frame_length + (nseg-1)*nstep,
        #  with integer nseg
        nadd = (-(x.shape[-1] - frame_length) % frame_shift) % frame_length
        pad_shape = [(0, 0) for _ in range(x.ndim - 1)] + [(0, nadd)]
        x = np.pad(x, pad_shape, mode="constant", constant_values=0)

    # Created strided array of data segments
    if frame_length == 1 and frame_length == frame_shift:
        result = x[..., None]
    else:
        shape = x.shape[:-1] + (
            (x.shape[-1] - frame_length) // frame_shift + 1,
            frame_length,
        )
        strides = x.strides[:-1] + (frame_shift * x.strides[-1], x.strides[-1])
        result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return result

def detect_non_silence(
    x: np.ndarray,
    threshold: float = 0.01,
    frame_length: int = 1024,
    frame_shift: int = 512,
    window: str = "boxcar",
) -> np.ndarray:
    """Power based voice activity detection.

    Args:
        x: (Channel, Time)
    >>> x = np.random.randn(1000)
    >>> detect = detect_non_silence(x)
    >>> assert x.shape == detect.shape
    >>> assert detect.dtype == np.bool
    """
    if x.shape[-1] < frame_length:
        return np.full(x.shape, fill_value=True, dtype=np.bool)

    if x.dtype.kind == "i":
        x = x.astype(np.float64)
    # framed_w: (C, T, F)
    framed_w = framing(
        x,
        frame_length=frame_length,
        frame_shift=frame_shift,
        centered=False,
        padded=True,
    )
    framed_w *= scipy.signal.get_window(window, frame_length).astype(framed_w.dtype)
    # power: (C, T)
    power = (framed_w**2).mean(axis=-1)
    # mean_power: (C, 1)
    mean_power = np.mean(power, axis=-1, keepdims=True)
    if np.all(mean_power == 0):
        return np.full(x.shape, fill_value=True, dtype=bool)
    # detect_frames: (C, T)
    detect_frames = power / mean_power > threshold
    # detects: (C, T, F)
    detects = np.broadcast_to(
        detect_frames[..., None], detect_frames.shape + (frame_shift,)
    )
    # detects: (C, TF)
    detects = detects.reshape(*detect_frames.shape[:-1], -1)
    # detects: (C, TF)
    return np.pad(
        detects,
        [(0, 0)] * (x.ndim - 1) + [(0, x.shape[-1] - detects.shape[-1])],
        mode="edge",
    )

#############################
# Augmentations per sample
#############################
def add_noise(speech_sample, noise_sample, snr=5.0, rng=None):
    """Mix the speech sample with an additive noise sample at a given SNR.

    Args:
        speech_sample (np.ndarray): a single speech sample (Channel, Time)
        noise_sample (np.ndarray): a single noise sample (Channel, Time)
        snr (float): signal-to-nosie ratio (SNR) in dB
        rng (np.random.Generator): random number generator
    Returns:
        noisy_sample (np.ndarray): output noisy sample (Channel, Time)
        noise (np.ndarray): scaled noise sample (Channel, Time)
    """
    len_speech = speech_sample.shape[-1]
    len_noise = noise_sample.shape[-1]
    if len_noise < len_speech:
        offset = rng.integers(0, len_speech - len_noise)
        # Repeat noise
        noise_sample = np.pad(
            noise_sample,
            [(0, 0), (offset, len_speech - len_noise - offset)],
            mode="wrap",
        )
    elif len_noise > len_speech:
        offset = rng.integers(0, len_noise - len_speech)
        noise_sample = noise_sample[:, offset : offset + len_speech]

    non_silence_indices_speech = detect_non_silence(speech_sample)
    non_silence_indices_noise = detect_non_silence(noise_sample)

    if len(non_silence_indices_noise) == 0:
        # 用speech_sample替换noise_sample   
        noise = speech_sample
        return speech_sample, noise
    
    power_speech = (speech_sample[non_silence_indices_speech] ** 2).mean()
    power_noise = (noise_sample[non_silence_indices_noise] ** 2).mean()
    
    scale = 10 ** (-snr / 20) * np.sqrt(power_speech) / np.sqrt(max(power_noise, 1e-10))
    noise = scale * noise_sample
    noisy_speech = speech_sample + noise
    
    return noisy_speech, noise


def add_reverberation(speech_sample, rir_sample):
    """Mix the speech sample with an additive noise sample at a given SNR.

    Args:
        speech_sample (np.ndarray): a single speech sample (1, Time)
        rir_sample (np.ndarray): a single room impulse response (RIR) (Channel, Time)
    Returns:
        reverberant_sample (np.ndarray): output noisy sample (Channel, Time)
    """
    # print(f"speech_sample.shape: {speech_sample.shape}, rir_sample.shape: {rir_sample.shape}")
    reverberant_sample = scipy.signal.convolve(speech_sample, rir_sample, mode="full")
    return reverberant_sample[:, : speech_sample.shape[1]]

def add_reverberation_v2(speech_sample, noisy_speech, rir_sample, fs):
    # print(f"speech_sample.shape: {speech_sample.shape}, rir_sample.shape: {rir_sample.shape}")
    rir_wav = rir_sample
    wav_len = speech_sample.shape[1]
    delay_idx = np.argmax(np.abs(rir_wav[0]))  # get the delay index
    delay_before_num = int(0.001 * fs)
    delay_after_num = int(0.005 * fs)
    idx_start = delay_idx - delay_before_num
    idx_end = delay_idx + delay_after_num
    if idx_start < 0:
        idx_start = 0
    early_rir = rir_wav[:, idx_start:idx_end]
    
    reverbant_speech_early = scipy.signal.fftconvolve(speech_sample, early_rir, mode="full")
    reverbant_speech = scipy.signal.fftconvolve(noisy_speech, rir_wav, mode="full")
    
    reverbant_speech = reverbant_speech[:, idx_start:idx_start + wav_len]
    reverbant_speech_early = reverbant_speech_early[:, :wav_len]
    scale = max(abs(reverbant_speech[0]))
    if scale == 0:
        scale = 1
    else:
        scale = 0.5 / scale
    reverbant_speech_early = reverbant_speech_early * scale
    reverbant_speech = reverbant_speech * scale
    return reverbant_speech, reverbant_speech_early

def bandwidth_limitation(speech_sample, fs: int, fs_new: int, res_type="kaiser_best"):
    """Apply the bandwidth limitation distortion to the input signal.

    Args:
        speech_sample (np.ndarray): a single speech sample (1, Time)
        fs (int): sampling rate in Hz
        fs_new (int): effective sampling rate in Hz
        res_type (str): resampling method

    Returns:
        ret (np.ndarray): bandwidth-limited speech sample (1, Time)
    """
    opts = {"res_type": res_type}
    if fs == fs_new:
        return speech_sample
    # assert fs > fs_new, (fs, fs_new)
    ret = librosa.resample(speech_sample, orig_sr=fs, target_sr=fs_new, **opts)
    # resample back to the original sampling rate
    ret = librosa.resample(ret, orig_sr=fs_new, target_sr=fs, **opts)
    return ret[:, : speech_sample.shape[1]]


def clipping(speech_sample, min_quantile: float = 0.06, max_quantile: float = 0.9):
    """Apply the clipping distortion to the input signal.
    speech_sample: np.ndarray, a single speech sample (1, Time)
    threshold: float, the threshold for clipping
    """
    
    threshold = random.uniform(min_quantile, max_quantile)
    ret = np.clip(speech_sample, -threshold, threshold)
    
    return ret

#############################
# Audio utilities
#############################
def read_audio(filename, force_1ch=False, fs=None):
    if os.path.exists(filename):
        audio_bytes = open(filename, "rb").read()
    else:
        raise FileNotFoundError(f"Audio file {filename} does not exist.")
    audio_bytes = io.BytesIO(audio_bytes)

    audio, fs_ = sf.read(audio_bytes, always_2d=True) # (Time, Channel)
    if force_1ch:
        audio = audio.mean(axis=-1, keepdims=True)
    audio = audio.T # (Channel, Time)
    # audio = audio[:, :1].T if force_1ch else audio.T # (Channel, Time)
    if fs is not None and fs != fs_:
        audio = librosa.resample(audio, orig_sr=fs_, target_sr=fs, res_type="soxr_hq")
        return audio, fs
    return audio, fs_

def save_audio(audio, filename, fs):
    if audio.ndim != 1:
        audio = audio[0] if audio.shape[0] == 1 else audio.T
    sf.write(filename, audio, samplerate=fs)

default_degradation_config = {
    # add noise
    "p_noise": 0.9,
    "snr_min": -5,
    "snr_max": 20,
    # add voice snr
    "voice_snr_min": 0,
    "voice_snr_max": 10,
    # add reverb
    "p_reverb": 0.5,
    "reverb_time": 1.5,
    "reverb_fadeout": 0.5,
    "p_post_reverb": 0.25,
    # add clipping
    "p_clipping": 0.25,
    # "clipping_min_db": -20,
    # "clipping_max_db": 0,
    # apply bandwidth limitation
    "p_bandwidth_limitation": 0.5,
    "bandwidth_limitation_rates": [
        2000,
        4000,
        8000,
        16000,
        22050,
        24000,
        32000
    ],
    "bandwidth_limitation_methods": [
        "kaiser_best",
        "kaiser_fast",
        "scipy",
        "polyphase",
    ],
}

def process_from_audio_path(
    noise_path,
    vocal_path=None,
    rir_path=None,
    to_seperate_vocal_paths=None,
    fs=None,
    force_1ch=True,
    degradation_config=default_degradation_config,
    length=None,
    clean_audio=None,
    reverb_v3=False,
):
    reverb_func = add_reverberation_v2
    
    if fs is None and clean_audio is not None:
        fs = sf.info(vocal_path).samplerate

    if clean_audio is None:
        vocal, _ = read_audio(vocal_path, force_1ch=force_1ch, fs=fs)
    else:
        vocal = clean_audio
    noise, _ = read_audio(noise_path, force_1ch=force_1ch, fs=fs)
    # if length is not None:
    #     vocal = pad_or_truncate(vocal, length)
    noisy_vocal = vocal.copy()

    # mix different source vocals
    if to_seperate_vocal_paths is not None:
        # like add noise, using "voice_snr_min" and "voice_snr_max" to control the SNR
        for to_seperate_vocal_path in to_seperate_vocal_paths:
            to_seperate_vocal, _ = read_audio(to_seperate_vocal_path, force_1ch=force_1ch, fs=fs)
            snr = random.uniform(degradation_config["voice_snr_min"], degradation_config["voice_snr_max"])
            noisy_vocal, _ = add_noise(noisy_vocal, to_seperate_vocal, snr=snr, rng=np.random.default_rng())

    # add reverb
    if rir_path is not None and random.random() < degradation_config["p_reverb"]:
        # print('add reverb')
        rir_sample = read_audio(rir_path, force_1ch=force_1ch, fs=fs)[0]
        noisy_vocal, vocal = reverb_func(vocal, noisy_vocal, rir_sample, fs)

    if random.random() < degradation_config["p_clipping"]:
        # 0.06 - 0.9
        noisy_vocal = clipping(noisy_vocal)

    if random.random() < degradation_config["p_bandwidth_limitation"]:
        # print('add bandwidth limitation')
        fs_new = random.choice(degradation_config["bandwidth_limitation_rates"])
        res_type = random.choice(degradation_config["bandwidth_limitation_methods"])
        noisy_vocal = bandwidth_limitation(noisy_vocal, fs=fs, fs_new=fs_new, res_type=res_type)

    # add noise
    if random.random() < degradation_config["p_noise"]:
        # print('add noise')
        snr = random.uniform(degradation_config["snr_min"], degradation_config["snr_max"])
        noisy_vocal, noise_sample = add_noise(noisy_vocal, noise, snr=snr, rng=np.random.default_rng())

    # normalization
    scale = 1 / max(
        np.max(np.abs(noisy_vocal)),
        np.max(np.abs(vocal)),
        np.max(np.abs(noise)),
    )

    vocal *= scale
    noise *= scale
    noisy_vocal *= scale

    return vocal, noise, noisy_vocal, fs
    
import os
import json
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def read_scp(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scp file not found: {path}")
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def process_single_item(speech_path, noise_list, rir_list, degradation_config, dst_dir, sr):
    try:
        noise_path = random.choice(noise_list)
        rir_path = random.choice(rir_list)

        audio, fs = read_audio(speech_path, force_1ch=True, fs=sr)

        clean_sample, _, noisy_speech, _ = process_from_audio_path(
            noise_path=noise_path,
            rir_path=rir_path,
            fs=fs,
            force_1ch=True,
            degradation_config=degradation_config,
            clean_audio=audio,
        )

        filename = os.path.basename(speech_path)
        save_audio(clean_sample, os.path.join(dst_dir, "clean", filename), fs)
        save_audio(noisy_speech, os.path.join(dst_dir, "noisy", filename), fs)
        
        return f"Successfully processed {filename}"
    except Exception as e:
        return f"Error processing {speech_path}: {e}"

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate clean/noisy speech pairs for training.")
    parser.add_argument("--speech_scp", type=str, required=True, help="Path to the scp file for clean speeches.")
    parser.add_argument("--noise_scp", type=str, required=True, help="Path to the scp file for noises.")
    parser.add_argument("--rir_scp", type=str, required=True, help="Path to the scp file for room impulse responses (RIRs).")
    parser.add_argument("--dst_dir", type=str, required=True, help="Destination directory to save 'clean' and 'noisy' subfolders.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel worker threads.")
    parser.add_argument("--sr", type=int, default=44100, help="Target sample rate for all audio.")
    
    args = parser.parse_args()

    degradation_config = {
        "p_noise": 0.9, "snr_min": -5, "snr_max": 20,
        "voice_snr_min": 0, "voice_snr_max": 10,
        "p_reverb": 0.5, "reverb_time": 1.5, "reverb_fadeout": 0.5, "p_post_reverb": 0.25,
        "p_clipping": 0.25,
        "p_bandwidth_limitation": 0.5,
        "bandwidth_limitation_rates": [4000, 8000, 16000, 24000, 32000],
        "bandwidth_limitation_methods": ["kaiser_best", "kaiser_fast", "scipy", "polyphase"],
    }
    
    os.makedirs(args.dst_dir, exist_ok=True)
    os.makedirs(os.path.join(args.dst_dir, 'clean'), exist_ok=True)
    os.makedirs(os.path.join(args.dst_dir, 'noisy'), exist_ok=True)

    speech_list = read_scp(args.speech_scp)
    noise_list = read_scp(args.noise_scp)
    rir_list = read_scp(args.rir_scp)
    
    print(f'Found {len(speech_list)} speech files.')
    print(f'Found {len(noise_list)} noise files.')
    print(f'Found {len(rir_list)} RIR files.')
    
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(
                process_single_item, 
                speech_path, 
                noise_list, 
                rir_list, 
                degradation_config, 
                args.dst_dir, 
                args.sr
            ) 
            for speech_path in speech_list
        ]
        
        for future in tqdm(as_completed(futures), total=len(speech_list), desc="Generating degraded audio"):
            result = future.result()
            if "Error" in result:
                tqdm.write(result)
    
    print("\nAll tasks completed.")

if __name__ == "__main__":
    main()