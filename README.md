## Installation

```bash
conda create -n anyenhance python=3.9
conda activate anyenhance
pip install -r requirements.txt
conda install ffmpeg
```

## Prepare pretrained models

1. prepare dac codec
```bash
mkdir -p pretrained/dac
cd pretrained/dac
wget https://huggingface.co/descript/descript-audio-codec/resolve/main/weights.pth?download=true -O weights.pth
```

You should have the following structure:

```
pretrained/
├── dac 
    └── weights.pth
```

2. prepare w2v-bert2 for semantic enhancement stage
```bash
huggingface-cli download facebook/w2v-bert-2.0
```

If you have trouble with network, you can try changing the mirror to `https://hf-mirror.com/` like this: `export HF_ENDPOINT=https://hf-mirror.com`

## Data Preparation

### Baseline Training Data

We have provided training data for the baseline model in `TODO`, you can download it from [TODO](https://TODO). The data needs approximately **160GB** of disk space. The data includes approximately **200 hours** of paired audio, which consists of clean audio, paired noisy audio, MP3 encoded audio, and a enhanced model's output corresponds to the noisy audio, **all under the same file name**. The file structure is as follows:

```
train_v1/
├── train_v1.jsonl
├── clean
│   ├── 0001.wav
│   ├── 0002.wav
│   └── ...
├── noisy
│   ├── 0001.wav
│   ├── 0002.wav
│   └── ...
├── encoded # MP3 encoded audio content, in the wav format
│   ├── 0001.wav
│   ├── 0002.wav
│   └── ...
├── generated # Enhanced audio content, in the wav format
│   ├── anyenhance
│   |   ├── 0001.wav
│   |   ├── 0002.wav
│   |   └── ...
│   ├── demucs
│   |   ├── 0001.wav
│   |   ├── 0002.wav
│   |   └── ...
│   └── ...
```

**After downloading the data, you should run this script to prepare the data paths:**

```bash
cd dataset
python generate_jsonl.py --train_v1_src ["your path ending with xxx/train_v1/"]
```

The metadata is arranged by a jsonl file `train_v1.jsonl`,  which contains the paths to the clean audio, noisy audio, and other distortion types (e.g., MP3 encoded audio, enhanced audio). The jsonl file is structured as follows:
```json
{"clean": "/path/to/clean/0001.wav", "noisy": "/path/to/noisy/0001.wav", "other_distortion": ["/path/to/generated/storm/0001.wav"]}
{"clean": "/path/to/clean/0002.wav", "noisy": "/path/to/noisy/0002.wav", "other_distortion": ["/path/to/encoded/0002.wav", "/path/to/generated/anyenhance/0002.wav"]}
```

### Custom Data
If you want to train the model with your own data, please create the same structure as above. The metadata should be arranged in a jsonl file, in which each line is a json object containing `clean`, `noisy`, and `other_distortion` keys. The `clean` key should point to the clean audio file, the `noisy` key should point to the noisy audio file, and the `other_distortion` key should be a list of paths to other distortion types (e.g., MP3 encoded audio, enhanced audio).

## Training

### Baseline Training

The config file for the baseline model is `config/anyenhance_v1.json`. You need to change the `["dataset"]["jsonl_file_path"]` to the path of your prepared jsonl file. The training command is as follows:

```bash
accelerate launch --mixed_precision=fp16 --main_process_port=20096 trainer.py \
    --config "./config/anyenhance_v1.json" \
    --exp_path "./exp/"
```

### Resume Training

You can use `--resume_path` to resume training with your path containing:

```bash
accelerate launch --mixed_precision=fp16 --main_process_port=20096 trainer.py \
    --config [your_config] \
    --exp_path "./exp/" \
    --resume_path [./exp/your_exp/model/checkpoint]
```

## Inference

To run inference, you need to provide the path to the pretrained model checkpoint and the input audio file/folder. The output will be saved in the specified output folder. The example command is as follows:

```python
# infer single audio file
python infer.py \
    --config_path "./config/anyenhance_v1.json" \
    --ckpt_path "/mnt/data4/zhangjunan/ccf-aatc/masksr-ccf-aatc/exp/20250620-20:27-anyenhance/model/epoch-52-step-129000-loss-4.3193/model.pt" \
    --input_file "./dataset/debug_audio/noisy/2.wav" \
    --output_folder "./output/"
# infer audio folder
python infer.py \
    --config_path "./config/anyenhance_v1.json" \
    --ckpt_path "/mnt/data4/zhangjunan/ccf-aatc/masksr-ccf-aatc/exp/20250620-20:27-anyenhance/model/epoch-52-step-129000-loss-4.3193/model.pt" \
    --input_folder "./dataset/debug_audio/noisy/" \
    --output_folder "./output/debug_audio_enhanced/"
```

## Evaluation

To evaluate the model, you can use the `evaluate.py` script. The evaluation can compute the DNSMOS, intrusive metrics (PESQ, ESTOI) ans WER metrics. The example command is as follows:

```python
python evaluate.py \
    --enhanced_folder "./output/debug_audio_enhanced/" \
    --gt_folder "./dataset/debug_audio/clean/" \
    --dnsmos \
    --intrusive \
    --wer
```

## Other Resources

### Data Simulation

We provide two scripts to simulate data. The first one simulates noisy-clean audio from (speech, noise, rir) pairs, and the second one simulates MP3 encoded audio from clean audio. The scripts are under `data_simulation` folder. You can refer to [data_simulation/README.md](data_simulation/README.md) for more details.