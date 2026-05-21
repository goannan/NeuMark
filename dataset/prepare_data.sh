#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/dataset}"

LIBRITTS_TRAIN_SUBSETS="${LIBRITTS_TRAIN_SUBSETS:-train-clean-100,train-clean-360,train-other-500}"
LIBRISPEECH_TEST_SUBSETS="${LIBRISPEECH_TEST_SUBSETS:-test-clean}"
TRAIN_VALID_SIZE="${TRAIN_VALID_SIZE:-100}"

echo "[dataset] project root: $PROJECT_ROOT"
echo "[dataset] data root:    $DATA_ROOT"

python "$PROJECT_ROOT/dataset/prepare_data.py" \
    --root "$DATA_ROOT" \
    --libritts-train-subsets "$LIBRITTS_TRAIN_SUBSETS" \
    --librispeech-test-subsets "$LIBRISPEECH_TEST_SUBSETS" \
    --train-valid-size "$TRAIN_VALID_SIZE" \
    --filelist-dir "$PROJECT_ROOT/dataset/filelists"

echo "[dataset] all data assets are ready."
