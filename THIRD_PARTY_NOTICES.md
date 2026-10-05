# Third-party notices

This repository is an independent submission to the Parameter Golf challenge
and builds on the following prior work. All borrowed code is MIT/APACHE licensed
and is attributed here and in `docs/LiYaxuan_C2G_拿来说明.md`.

## openai/parameter-golf (MIT, Copyright (c) 2026 OpenAI)
- https://github.com/openai/parameter-golf
- Borrowed: the overall train/eval/pack flow, shard format, Muon wiring,
  U-Net skip structure, per-lane learned residual scales, QK RMSNorm,
  sliding-window BPB evaluation, and the int8 RTN quantization structure.

## Keller Jordan — Muon / modded-nanoGPT
- Muon: https://github.com/KellerJordan/Muon
- modded-nanoGPT: https://github.com/KellerJordan/modded-nanogpt
- Background: https://kellerjordan.github.io/posts/muon/
- Borrowed: Newton-Schulz orthogonalized optimizer.

## Andrej Karpathy — nanoGPT
- https://github.com/karpathy/nanoGPT
- Borrowed: general code style and model layout.

## GPTQ — Frantar, Ashkboos, Hoefler, Alistarh (2023)
- arXiv:2210.17323 ; code: https://github.com/IST-DASLab/gptq
- Borrowed: Hessian-guided sequential quantization (GPTQ-lite here).

## SDClip — Kevin Clark, parameter-golf record (2026-04-05)
- https://github.com/openai/parameter-golf/tree/main/records/
- Borrowed: clip = k * std(row) with k=12.85 (int6) / k=20 (int8 embeddings).

## FineWeb — Penedo et al. (2024)
- arXiv:2406.17557
- https://huggingface.co/datasets/HuggingFaceFW/fineweb
- Data source for all experiments (via the challenge's hosted shards).

## SentencePiece — Kudo & Richardson (2018)
- arXiv:1808.06226 ; https://github.com/google/sentencepiece
- Tokenizer used by the challenge.

## Architecture references
- Parallel residual lanes: Wang & Komatsuzaki, GPT-J-6B (2021).
- Depth recurrence / Universal Transformer: Dehghani et al., arXiv:1807.03819.
- RoPE: Su et al., arXiv:2104.09864.
- GQA: Ainslie et al., arXiv:2305.13245.
- RMSNorm: Zhang & Sennrich, arXiv:1910.07467.
- FlashAttention: Dao et al., arXiv:2205.14135 (and follow-ups).
