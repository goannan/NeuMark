#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -d "$PROJECT_ROOT/dataset/LibriSpeech/test-clean" ]; then
    echo "Validation dataset not found: $PROJECT_ROOT/dataset/LibriSpeech/test-clean" >&2
    echo "Run this first:"
    echo "  bash dataset/prepare_data.sh"
    exit 1
fi

cd "$PROJECT_ROOT"

python train/valid.py \
    --checkpoint "$PROJECT_ROOT/neumark_150000.pt" \
    --dataset_root "$PROJECT_ROOT/dataset/LibriSpeech/test-clean" \
    --config "$PROJECT_ROOT/config/default.json" \
    --output_dir "$PROJECT_ROOT/valid_samples"

echo "[valid] finished."

