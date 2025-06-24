import torch
from torch.utils.data import Dataset
import torchaudio
import os
import json
import random
import numpy as np

class JsonlAudioDataset(Dataset):
    def __init__(self, jsonl_file_path, seq_len=512*260, sr=44100):
        self.data_list = self._load_jsonl(jsonl_file_path)
        self.seq_len = seq_len
        self.sr = sr

    def _load_jsonl(self, jsonl_file_path):
        """
        从JSONL文件加载数据。
        """
        records = []
        if not os.path.exists(jsonl_file_path):
            raise FileNotFoundError(f"JSONL file not found: {jsonl_file_path}")
        
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                
                    if not os.path.exists(record["clean"]):
                        print(f"Warning: Clean file not found, skipping record: {record['clean']}")
                        continue
                    if not os.path.exists(record["noisy"]):
                        print(f"Warning: Noisy file not found, skipping record: {record['noisy']}")
                        continue
                    records.append(record)
                except json.JSONDecodeError as e:
                    print(f"Skipping malformed JSONL line: {line.strip()} - Error: {e}")
        print(f"Loaded {len(records)} valid records from {jsonl_file_path}")
        return records

    def __len__(self):
        return len(self.data_list)

    def _load_audio(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)
        
        if sample_rate != self.sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sr)
            waveform = resampler(waveform)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        return waveform

    def pad_or_truncate(self, clean_audio, degraded_audio):
        len_clean = clean_audio.size(-1)
        len_degraded = degraded_audio.size(-1)

        max_common_len = max(len_clean, len_degraded, self.seq_len)
        if len_clean < max_common_len:
            clean_audio = torch.nn.functional.pad(clean_audio, (0, max_common_len - len_clean))
        if len_degraded < max_common_len:
            degraded_audio = torch.nn.functional.pad(degraded_audio, (0, max_common_len - len_degraded))
        
        current_len = clean_audio.size(-1)

        if current_len > self.seq_len:
            offset = np.random.randint(0, current_len - self.seq_len + 1)
            clean_audio = clean_audio[..., offset:offset+self.seq_len]
            degraded_audio = degraded_audio[..., offset:offset+self.seq_len]

        return clean_audio, degraded_audio

    def __getitem__(self, idx):
        while True:
            try:
                record = self.data_list[idx]
                clean_path = record["clean"]
                noisy_path = record["noisy"]
                other_distortion_paths = record["other_distortion"]

                available_degraded_sources = [noisy_path] + other_distortion_paths
                
                chosen_degraded_path = random.choice(available_degraded_sources)

                clean_audio = self._load_audio(clean_path)
                degraded_audio = self._load_audio(chosen_degraded_path)

                clean_audio, degraded_audio = self.pad_or_truncate(clean_audio, degraded_audio)

                return clean_audio, degraded_audio
            except Exception as e:
                print(f"Error loading or processing index {idx}: '{record.get('clean', 'N/A')}' or '{chosen_degraded_path if 'chosen_degraded_path' in locals() else 'N/A'}'. Error: {e}")
                idx = random.randint(0, len(self.data_list) - 1)

if __name__ == '__main__':
    jsonl_file = "/mnt/data4/zhangjunan/ccf-aatc/data/train_v1/train_v1.jsonl" 
    
    target_sample_rate = 44100
    sequence_length_samples = 260*512

    try:
        dataset = JsonlAudioDataset(
            jsonl_file_path=jsonl_file,
            seq_len=sequence_length_samples,
            sr=target_sample_rate
        )

        print(f"Dataset initialized with {len(dataset)} samples.")

        if len(dataset) > 0:
            clean_sample, degraded_sample = dataset[0]
            print(f"First sample shapes: Clean {clean_sample.shape}, Degraded {degraded_sample.shape}")
            assert clean_sample.shape == (1, sequence_length_samples)
            assert degraded_sample.shape == (1, sequence_length_samples)

            torchaudio.save("test_clean_sample_0.wav", clean_sample, sample_rate=target_sample_rate)
            torchaudio.save("test_degraded_sample_0.wav", degraded_sample, sample_rate=target_sample_rate)

            for i in range(min(3, len(dataset))):
                clean_s, deg_s = dataset[i]
                print(f"Sample {i}: Clean shape {clean_s.shape}, Degraded shape {deg_s.shape}")

        from torch.utils.data import DataLoader
        batch_size = 2
        num_workers = 4

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        print(f"\nDataLoader initialized with batch_size={batch_size}, num_workers={num_workers}.")
        print("Iterating through one batch from DataLoader:")
        for batch_idx, (clean_batch, degraded_batch) in enumerate(dataloader):
            print(f"Batch {batch_idx}: Clean batch shape {clean_batch.shape}, Degraded batch shape {degraded_batch.shape}")
            assert clean_batch.shape == (batch_size, 1, sequence_length_samples)
            assert degraded_batch.shape == (batch_size, 1, sequence_length_samples)
            if batch_idx == 0:
                break
        print("DataLoader test complete.")

    except Exception as e:
        print(f"An error occurred during dataset initialization or testing: {e}")