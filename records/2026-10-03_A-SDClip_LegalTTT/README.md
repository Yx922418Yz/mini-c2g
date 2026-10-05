# Record: Adaptive SDClip + Legal Score-First TTT

- **Author**: Li Yaxuan (github_id: liyaxuan), Zhengzhou Sias University, AI+X Elite 20
- **Date**: 2026-10-03
- **Track**: 10min / 16MB (micro-scale CPU development snapshot)

## Technique summary

1. **Adaptive SDClip (A-SDClip)** — original. The SDClip clip multiple `k` is
   selected per weight row from an empirical-Bayes-shrunk excess-kurtosis estimate:
   bias-corrected kurtosis, shrinkage `λ = m/(m+24)`, then
   `k_i = clip(k_base · exp(α κ_s), k_min, k_max)` with final `α=0.10`,
   `k_base=12.85` (int6), bounds `[8,20]`. Degenerates exactly to SDClip when the
   kurtosis estimate is uninformative; adds no bytes.
2. **GPTQ-lite** — float64 Hessians, 1% damp with adaptive retry, dead-feature
   handling, 128-column blocks with residual error propagation; int8 embeddings
   with SDClip k=20.
3. **Parallel residual lanes** (layers ≥ 4) and **depth recurrence** (layers 3,4
   repeated) carried from the leaderboard lineage; measured neutral/slightly
   negative at micro scale — reported honestly, to be re-verified on H100.
4. **Legal score-first TTT** — every chunk is fully scored under no_grad before the
   SGD update; 3 epochs, momentum 0.9, lr 0.005, cosine decay across chunks,
   grad clip 1.0; no token is rescored.

## Micro-scale results (real FineWeb prefix, CPU)

- Final artifact (this record): pre-quant BPB 2.15862; post-quant 2.21748;
  TTT **2.16518**; artifact 606,623 bytes.
- A-SDClip vs SDClip: mean Δ **+0.34782 BPB**, n=8 seeds, paired permutation
  **p=0.0078**.
- TTT vs post-quant: mean Δ **+0.05154 BPB**, n=8, **p=0.0078**.
- Architecture: PR ≈ baseline (neutral); DR −0.0046 at micro (negative result).

## Boundary

These are micro-scale CPU numbers on an 8M/2M-token FineWeb prefix with a d=128
model; they are NOT leaderboard BPB. The 8×H100 run is reproduced by
`scripts/reproduce_h100.sh` in the repository root. No H100 number is fabricated.

## Contents

- `train_gpt.py`, `pgolf/` — exact code snapshot producing this record;
- `submission.json` — metrics and metadata;
- `final_a010.jsonl` — structured event log of the run.
