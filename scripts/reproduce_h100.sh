#!/usr/bin/env bash
# One-command reproduction of the full submission on 8xH100 SXM.
# Prerequisites: official FineWeb SP1024 shards available at $DATA_PATH
# (prepare them with the official challenge preprocessing:
#  https://github.com/openai/parameter-golf -> data/prepare.py).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

pip install -r requirements.txt

export DATA_PATH="${DATA_PATH:-./data/full}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"

# Training shape (modded-nanoGPT-family defaults for 8xH100)
export ITERATIONS=10000
export WARMDOWN_ITERS=800
export BATCH_TOKENS=524288          # 8 ranks x 64 seqs x 1024
export SEQ_LEN=1024
export MODEL_DIM=512
export NUM_LAYERS=9
export NUM_HEADS=8
export NUM_KV_HEADS=4
export VAL_LIMIT_TOKENS=0
export MAX_WALLCLOCK_SECONDS=575

# Full stack: parallel residuals + depth recurrence + A-SDClip + legal TTT
export PARALLEL_RESIDUAL=1
export PR_FROM_LAYER=2
export DEPTH_RECURRENCE=1
export RECUR_LAYERS="4,5,6"
export QUANT_SCHEME=int6_asdclip
export TTT_ENABLED=1
export TTT_CHUNK_TOKENS=32768
export TTT_EPOCHS=3
export TTT_LR=0.005

torchrun --standalone --nproc_per_node=8 train_gpt.py
python pack_submission.py --check artifacts/*_submission.tar.gz
