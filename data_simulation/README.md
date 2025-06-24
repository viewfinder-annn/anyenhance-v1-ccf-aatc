## Simulate Noisy-Clean Audio

The `audio_degradation_pipeline.py` script simulates noisy-clean audio pairs, which can be used for training and evaluation of audio enhancement models. The script generates noisy audio by adding noise to clean audio files, and it can also simulate other distortions like MP3 encoding.

You need to prepare .scp files for speech, noise, and RIR (Room Impulse Response) data. The .scp files should contain paths to the audio files, one per line. The script will read these files and generate noisy audio by mixing clean speech with noise and applying RIR. Example Usage:

```bash
cd data_simulation
python audio_degradation_pipeline.py \
    --speech_scp ./example/speech.scp \
    --noise_scp ./example/noise.scp \
    --rir_scp ./example/rir.scp \
    --dst_dir ./example/simulated/ \
    --num_workers 2 \
    --sr 44100
```

## Simulate Mp3 Encoded Audio

The `mp3_degradation_pipeline.py` script simulates MP3 encoded audio by converting clean audio files to MP3 format. This can be useful for training models that need to handle compressed audio formats.

You need to prepare .scp files for speech data. The .scp files should contain paths to the audio files, one per line. The script will read these files and convert them to MP3 format. Example Usage:

```bash
cd data_simulation
python mp3_degradation_pipeline.py \
    --speech_scp ./example/speech.scp \
    --dst_dir ./example/simulated/ \
    --num_workers 2 \
    --sr 44100
```