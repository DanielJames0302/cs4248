# Chinese-English Neural Machine Translation

A full MLOps pipeline for Chinese-to-English neural machine translation, fine-tuning Google's mT5 encoder-decoder models on parallel corpora with support for full and LoRA-based parameter-efficient training across single and multi-GPU setups.

## Features

- **mT5 fine-tuning** across three model sizes (small, base, large) with configurable YAML-driven experiments
- **LoRA** via HuggingFace PEFT for parameter-efficient fine-tuning, targeting all attention projection layers (rank 32)
- **Multi-GPU training** using PyTorch DDP (`torchrun`) with automatic GPU detection
- **Streaming tokenization** for large-scale datasets with chunked processing and garbage collection
- **Beam search inference** with length penalty and early stopping, supporting both interactive and batch modes
- **Dual evaluation** with SacreBLEU (n-gram matching) and COMET (neural semantic metric) on Tatoeba and WMT 2022 test sets
- **Dataset subsetting and verification** tools for reproducible experimentation
- **SLURM-compatible** job scripts for HPC cluster execution

## Model Architecture

- **Base:** Google mT5 (Multilingual T5) — Transformer encoder-decoder pretrained on 101 languages
- **Sizes:** mT5-small (~300M params), mT5-base (~580M), mT5-large (~1.2B)
- **Tokenizer:** mT5 SentencePiece with max sequence length of 128 tokens
- **LoRA config:** rank 32, alpha 32, targeting `q`, `k`, `v`, `o` attention projections

## Project Structure

```
├── train_mt.py              # Training script (single/multi-GPU)
├── inference.py             # Inference (interactive + batch)
├── configs/                 # YAML training configs (8 variants)
├── tokenizer/
│   ├── tokenizer.py         # Standard tokenization
│   ├── tokenizer_streaming.py  # Memory-efficient streaming tokenization
│   └── chunk_merging.py     # Merge tokenized chunks
├── create_dataset_subset.py # Create reproducible dataset subsets
├── verify_subset.py         # Verify subset integrity
├── dataset/                 # Raw parallel corpora
├── tokenized_dataset/       # Pre-tokenized datasets (Arrow format)
├── requirements.txt         # Python dependencies
└── *.sh                     # SLURM job scripts
```

## Dataset

- **Training:** ALMA Human Parallel Chinese-English corpus
- **Evaluation:** Tatoeba parallel corpus + WMT 2022 Chinese-English test set

## Getting Started

### Prerequisites

- Python 3.10+
- NVIDIA GPU (recommended: H100 for large model training)
- CUDA and PyTorch installed

### 1. Environment Setup

```bash
conda create -n mt_env python=3.10
conda activate mt_env
pip install -r requirements.txt
```

### 2. Tokenize Dataset

```bash
# Standard (small datasets)
python tokenizer/tokenizer.py

# Streaming (large datasets — processes in chunks with GC)
python tokenizer/tokenizer_streaming.py
```

### 3. Create Dataset Subset (optional)

```bash
python create_dataset_subset.py \
  --input ./tokenized_dataset/WMT22_Train_Merged \
  --output ./tokenized_dataset/WMT22_Train_Merged_5pct \
  --percentage 0.05 --seed 42
```

### 4. Train

```bash
# Single GPU
python train_mt.py --config ./configs/mT5-large-training-single-gpu.yaml

# Multi-GPU (2x H100)
torchrun --nproc_per_node=2 --master_port=29500 \
  train_mt.py --config ./configs/mT5-large-training-multi-gpu.yaml --multi-gpu

# LoRA variant
torchrun --nproc_per_node=2 --master_port=29500 \
  train_mt.py --config ./configs/mT5-large-training-lora-multi-gpu.yaml --multi-gpu
```

### 5. Inference

```bash
# Single text
python inference.py \
  --model-path ./models/mt5-large-finetuned/checkpoint-XXXX \
  --input-text "28岁厨师被发现死于旧金山一家商场"

# Batch file
python inference.py \
  --model-path ./models/mt5-large-finetuned/checkpoint-XXXX \
  --input-file ./dataset/tatoeba.zh \
  --output-file ./outputs/tatoeba_mt5_large.en \
  --num-beams 4 --max-length 512 --batch-size 32
```

### 6. Evaluate

```bash
# SacreBLEU
sacrebleu -tok 13a -w 2 ./dataset/tatoeba.en < ./outputs/tatoeba_mt5_large.en

# COMET
comet-score -s ./dataset/tatoeba.zh \
  -t ./outputs/tatoeba_mt5_large.en \
  -r ./dataset/tatoeba.en --batch_size 256 --gpus 1 --only_system
```

## Training Configs

| Config | Model | LoRA | GPUs | Eval |
|--------|-------|------|------|------|
| `mT5-small-training-multi-gpu.yaml` | small | No | Multi | — |
| `mT5-small-training-lora-multi-gpu.yaml` | small | Yes | Multi | — |
| `mT5-base-training-multi-gpu.yaml` | base | No | Multi | Tatoeba |
| `mT5-base-training-lora-multi-gpu.yaml` | base | Yes | Multi | Tatoeba |
| `mT5-large-training-single-gpu.yaml` | large | No | Single | — |
| `mT5-large-training-multi-gpu.yaml` | large | No | Multi | Tatoeba |
| `mT5-large-training-lora-multi-gpu.yaml` | large | Yes | Multi | Tatoeba |

## Tech Stack

- **Framework:** HuggingFace Transformers, PyTorch, Accelerate
- **PEFT:** HuggingFace PEFT (LoRA)
- **Evaluation:** SacreBLEU, COMET (Unbabel)
- **Tokenization:** SentencePiece via mT5 tokenizer
- **Infrastructure:** SLURM, Miniconda, HPC cluster (NVIDIA H100 GPUs)

## License

See [LICENSE](LICENSE) for details.
