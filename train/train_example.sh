#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -f "$PROJECT_ROOT/dataset/filelists/train.txt" ] || [ ! -f "$PROJECT_ROOT/dataset/filelists/valid.txt" ]; then
    echo "Missing dataset file lists. Run this first:"
    echo "  bash dataset/prepare_data.sh"
    exit 1
fi

cd "$PROJECT_ROOT"

accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    train/train_example.py \
    --config "$PROJECT_ROOT/config/default.json" \
    --audio_dir "$PROJECT_ROOT/dataset/LibriTTS" \
    --exts wav \
    --valid_set_size 0
