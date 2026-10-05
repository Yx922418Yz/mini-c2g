# Parameter Golf — Adaptive SDClip (A-SDClip) submission

**Author:** Li Yaxuan (李亚轩), Zhengzhou Sias University — AI+X Elite 20 Program
**Challenge:** C2G *Parameter Golf: Train the Best nanoGPT in 10 Minutes, 16 MB*
**Target metric:** Bits-Per-Byte (BPB) on FineWeb validation — lower is better.

> **Hardware boundary (read first).** The official leaderboard runs on
> 8×H100 SXM with a 10-minute training budget. This repository was developed
> on a CPU-only machine; therefore every reported number in `results/` comes
> from **real runs on real FineWeb tokens at micro scale**, while the exact
> 8×H100 submission is reproduced with one command (`scripts/reproduce_h100.sh`).
> The boundary is stated explicitly everywhere numbers appear; no H100 BPB is
> fabricated.

## What is new here

**Adaptive SDClip (A-SDClip).** SDClip clips each weight row at
`k · std(row)` with one global `k` (12.85 for int6 matrices). A-SDClip chooses
`k` **per row** from the row's empirical-Bayes-shrunk excess kurtosis:

```
k_i = clip( k_base · exp(α · κ_i), k_min, k_max )
```

Heavy-tailed rows (κ>0) receive a wider clip so outlier weights are not
collapsed; near-Gaussian rows keep the base value. Motivation, derivation and
the honest (including null) results are in `docs/LiYaxuan_C2G_方案设计.md`.

The full stack combines four components, each ablated separately:
parallel residual lanes (GPT-J), depth recurrence (Universal Transformer),
legal score-first test-time training, and A-SDClip int6 quantization with
GPTQ-lite calibration.

## Headline results (micro-scale, real FineWeb prefix)

Paired deltas from real local runs; **not** leaderboard BPB and not extrapolated:

| Comparison | CPU (fp32), n=8 | GPU RTX 5060 (bf16), n=8 |
|---|---|---|
| A-SDClip vs SDClip | **+0.3478**, p=0.0078 | **0.3419**, p=0.0078 |
| Legal TTT improvement (fp32 TTT) | +0.0515, p=0.0078 | +0.0494, p=0.0078 |
| Parallel residual vs baseline | ~0, neutral (n=3) | −0.0004, p=0.36 |
| Depth recurrence vs baseline | −0.0046, harmful (n=3) | −0.0042, p=0.0078 |

The full 48-cell matrix was independently replicated across hardware and
numerical precision; every headline conclusion reproduced in direction and
significance. A paired precision test on the same artifacts shows fp32 TTT beats
bf16 TTT by ~0.012 BPB (3/3 seeds), so fp32 TTT is the default
(`TTT_FORCE_FP32=True`); bf16 is reproducible via `--ttt-bf16`. Final micro
artifact: 606,940 bytes (16 MB gate passed).

## Quick start

### CPU smoke / reproduction (Windows)
```powershell
D:\HJY\python.exe -m pip install -r requirements-cpu.txt
powershell -ExecutionPolicy Bypass -File scripts/smoke_cpu.ps1
```

### CPU smoke / reproduction (Linux, macOS)
```bash
pip install -r requirements-cpu.txt
bash scripts/download_micro_data.sh
python scripts/run_experiments.py --seeds 1337 2024 4242
python scripts/analyze.py
```

### Full 8×H100 leaderboard run
```bash
pip install -r requirements.txt
DATA_PATH=/path/to/full/shards bash scripts/reproduce_h100.sh
```

### CI
GitHub Actions runs the full pipeline on hermetic synthetic shards (no
network data dependency); see `.github/workflows/ci.yml`.

## Repository layout

```
train_gpt.py            # entry: train -> eval -> quantize -> pack
eval_bpb.py             # standalone evaluator for packed artifacts
prepare_local_data.py   # rewrite shard prefixes into micro shards
pack_submission.py      # size-gate checker (16,000,000 bytes)
pgolf/                  # config, model, muon, data, bpb, quant, ttt, artifacts
scripts/                # experiment matrix, analysis, data download, H100 repro
data/                   # populated by scripts (git-ignored; see data/README.md)
logs/                   # structured JSONL logs of every run
results/                # aggregated tables + paired permutation tests
artifacts/              # packed *.tar.gz submissions + sidecar *.json
records/LiYaxuan_C2G/   # official PR-style record folder
docs/                   # proposal, design, ablations, leaderboard, notices
```

## Challenge documents (required deliverables)

All hard-named deliverables live in `docs/` (and the record folder):
方案草案, 方案设计, ablation, leaderboard, AI 日志, 拿来说明, and the English
technical report. See `docs/README.md` for the index.

## License

MIT. This repository incorporates code from the MIT-licensed
[openai/parameter-golf](https://github.com/openai/openai/parameter-golf)
baseline; see `LICENSE` and `THIRD_PARTY_NOTICES.md`.

## Citation

```bibtex
@misc{li2026asdclip,
  author = {Li, Yaxuan},
  title  = {Adaptive SDClip: Per-Row Kurtosis-Guided Clipping for Parameter Golf},
  year   = {2026},
  howpublished = {Challenge submission, C2G Parameter Golf}
}
```
