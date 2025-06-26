# Baseline for the CCF-AATC 2025 Challenge [Track 1](https://ccf-aatc.org.cn/)

This repository provides a baseline for the CCF-AATC 2025 Challenge Track 1, which aims to do speech restoration under multiple distortions:

- Acoustic Degradation: (Noise & Reverberation)
- Signal Chain Artifacts: (Clipping, Bandwidth Limitation, Codec Distortions)
- Processing Artifacts: (Residual & Algorithm-induced Distortions)

Our baseline model is a lightweight version of the [AnyEnhance](https://arxiv.org/abs/2501.15417) framework. We provide a [pre-trained checkpoint](#prepare-baseline-model-weights) that has been trained on the challenge's training dataset.

> **!!Important: The baseline model is not the same as the one in the original [anyenhance paper](https://arxiv.org/abs/2501.15417), it is only used for this challenge.**

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

## Prepare baseline model weights

The baseline model is a simplified version of the AnyEnhance model with semantic alignment loss, which:

1. has a smaller model size, and different training data.
2. does not explicitly activate the prompt-guidance mechanism.
3. does not use the self-critic mechanism.

You can find the baseline weights [here](https://aishell-jiaofu.oss-cn-hangzhou.aliyuncs.com/ccf2025/aatc_track1/baseline/epoch-83-step-200000-loss-4.4187.tar). The baseline was trained on 2×A800 GPUs using our provided training set for 200k steps. For more configuration, please refer to `config/anyenhance_v1.json`.

After downloading the weights, you can extract them to the `pretrained/` folder, the structure should look like this:

```bash
epoch-83-step-200000-loss-4.4187
├── checkpoint.pth
├── model.pt
├── optimizer.pt
└── scheduler.pt
```

### Infer with pretrained model

You can run inference with the pretrained model by specifying `--ckpt_path` to the path of the baseline `model.pt` like this:

```bash
python infer.py \
    --config_path "./config/anyenhance_v1.json" \
    --ckpt_path "./pretrained/epoch-83-step-200000-loss-4.4187/model.pt" \
    --input_file "./dataset/debug_audio/noisy/2.wav" \
    --output_folder "./output/"
```

### Finetune with pretrained model

If you want to finetune the pretrained model, you can use the `--resume_path` argument to specify the path to the baseline folder **(make sure you have prepared the training data as described below)**. The command is as follows:

```bash
accelerate launch --mixed_precision=fp16 --main_process_port=20086 trainer.py \
    --config "./config/anyenhance_v1.json" \
    --exp_path "./exp/" \
    --resume_path "./pretrained/epoch-83-step-200000-loss-4.4187/"
```

## Data Preparation

### Baseline Training Set & Development Set

The training data and the development set are available for download to all registered participants. To gain access, please register on the official competition website:

**[https://ccf-aatc.org.cn/](https://ccf-aatc.org.cn/)**

Upon registration, you will find the download links.

#### Training Dataset

The training dataset requires approximately **320GB** of disk space and contains around **200 hours** of paired audio. For each entry, the data includes clean audio, a corresponding noisy version, an MP3-encoded version, and the output from a baseline enhancement model. Crucially, **all these associated files share the same base filename** for easy pairing.

The file structure is organized as follows:

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

#### Development Dataset

The development dataset contains 500 paired audio files, which are used for evaluation. It only has the clean and noisy audio folders, but covers all the distortion types described above.

### Data Simulation & Custom Data

#### Data Simulation

We provide two scripts to simulate data. The first one simulates noisy-clean audio from (speech, noise, rir) pairs, and the second one simulates MP3 encoded audio from clean audio. The scripts are under `data_simulation` folder. You can refer to [data_simulation/README.md](data_simulation/README.md) for more details.

#### Custom Data

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

## Citations

You can cite the original paper as follows:

```bibtex
@article{zhang2025anyenhance,
  title={AnyEnhance: A Unified Generative Model with Prompt-Guidance and Self-Critic for Voice Enhancement},
  author={Zhang, Junan and Yang, Jing and Fang, Zihao and Wang, Yuancheng and Zhang, Zehua and Wang, Zhuo and Fan, Fan and Wu, Zhizheng},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  year={2025}
}
```