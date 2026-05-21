# NeuMark: Robust Audio Watermarking Resistant to Neural Resynthesis

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.12+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://goannan.github.io/NeuMark/"><img alt="Demo Page" src="https://img.shields.io/badge/Demo%20Page-NeuMark-purple?style=for-the-badge&logo=google-chrome&logoColor=white"></a>

NeuMark is an open-source robust audio watermarking project utilizing SpeechTokenizer. This repository contains complete pipelines for training, validation, multi-dimensional robustness attacks, loss functions, a ready-to-test model checkpoint, and an interactive [audio demonstration page](https://goannan.github.io/NeuMark/).

## 🚀 Checkpoint
We provide a pre-trained NeuMark checkpoint ready for validation and testing:
```bash
neumark_150000.pt
```

## 1. Environment

Create and activate a Python environment. Python 3.12 is recommended.

```bash
git clone https://github.com/goannan/NeuMark.git
cd NeuMark

pyenv virtualenv 3.12.11 neumark
pyenv local neumark

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Dataset

All dataset preparation lives in `dataset/`. Do not read from an external `dataset` directory.

Run:

```bash
bash dataset/prepare_data.sh
```

This will download LibriTTS and LibriSpeech, and prepare file lists.

To use different subsets:

```bash
LIBRITTS_TRAIN_SUBSETS=train-clean-100,train-clean-360,train-other-500 \
LIBRISPEECH_TEST_SUBSETS=test-clean \
TRAIN_VALID_SIZE=100 \
bash dataset/prepare_data.sh
```

## 3. WavTokenizer

Validation includes a WavTokenizer attack. Put WavTokenizer inside this NeuMark project, not in a separate external project directory.

Clone the code:

```bash
cd NeuMark
git clone https://github.com/jishengpeng/WavTokenizer.git
```

Download the WavTokenizer config and checkpoint into `NeuMark/WavTokenizer/`:

```bash
cd WavTokenizer

curl -L -o wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml \
  https://huggingface.co/novateur/WavTokenizer/resolve/main/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml

curl -L -o WavTokenizer_small_600_24k_4096.ckpt \
  https://huggingface.co/novateur/WavTokenizer/resolve/main/WavTokenizer_small_600_24k_4096.ckpt
```

If WavTokenizer is missing, validation can still run, but the WavTokenizer attack will be skipped.

## 4. Validation

A pretrained NeuMark checkpoint is already provided:

```bash
neumark_150000.pt
```

After preparing the dataset, run validation directly:

```bash
bash train/valid.sh
```

Validation will print detection accuracy, bit accuracy, watermark probability, and transparency metrics (PESQ, STOI, SI-SNR).

## 5. Training

Prepare the dataset first:

```bash
bash dataset/prepare_data.sh
```

Then start training:

```bash
bash train/train_example.sh
```

Training outputs are saved under the log directory specified in the config.

Useful runtime overrides:

```bash
NUM_PROCESSES=1 bash train/train_example.sh
NUM_PROCESSES=4 MIXED_PRECISION=bf16 bash train/train_example.sh
CONFIG_PATH=/path/to/config.json bash train/train_example.sh
```

The main training settings are in `config/default.json`, including batch size, learning rate, epochs, loss weights, data file lists, and output folder.

## 6. Project Layout

```text
NeuMark/
├── attacks/              # audio attacks and augmentation
├── config/default.json   # default training and validation config
├── dataset/              # data download and preparation scripts
├── infer/                # inference and checkpoint loading
├── losses/loss.py        # training losses
├── STmodels/             # SpeechTokenizer code and pretrained checkpoint
├── train/                # training and validation entrypoints
├── models.py             # watermark embedder and detector
├── quality.py            # audio quality metrics (PESQ, STOI, SI-SNR)
└── neumark_150000.pt     # ready-to-test checkpoint
```

## 7. Quick Start

For a fresh user, the shortest path is:

```bash
git clone https://github.com/goannan/NeuMark.git
cd NeuMark
python -m pip install -r requirements.txt
bash dataset/prepare_data.sh
# follow the WavTokenizer clone/download commands above
bash train/valid.sh
```
