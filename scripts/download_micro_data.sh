#!/usr/bin/env bash
# Download official FineWeb shard prefixes + the official SP1024 tokenizer
# via hf-mirror.com, then build the micro dataset.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DL="$REPO/downloads"
mkdir -p "$DL"
BASE="https://hf-mirror.com/datasets/willdepueoai/parameter-golf/resolve/main"

curl -L -o "$DL/manifest.json" "$BASE/datasets/manifest.json"
curl -L -o "$DL/fineweb_1024_bpe.model" "$BASE/datasets/tokenizers/fineweb_1024_bpe.model"
curl -L -r 0-16001023 -o "$DL/train_head.bin" "$BASE/datasets/datasets/fineweb10B_sp1024/train_000000.bin"
curl -L -r 0-4001023  -o "$DL/val_head.bin"   "$BASE/datasets/datasets/fineweb10B_sp1024/val_000000.bin"

python "$REPO/prepare_local_data.py" --prefix-dir "$DL" --tokenizer "$DL/fineweb_1024_bpe.model"
echo "micro data ready under $REPO/data"
