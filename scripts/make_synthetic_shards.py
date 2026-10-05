"""Create hermetic synthetic shards + tokenizer for CI smoke tests.

No network access required. The tokens are NOT language data; the purpose is
only to exercise every code path (training, GPTQ-lite calibration, A-SDClip,
TTT, packing, size gate). Real-data experiments use download_micro_data.* .
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sentencepiece as spm

from pgolf.data import write_shard

SAMPLE = (
    "the model trains on tokens and learns to predict the next token. "
    "attention mixes every position with its causal prefix. "
    "quantization maps weights into small integer codes. "
    "bits per byte measures compression of the original text. "
    "recurrence repeats layers so that shared weights run at more depths. "
    "parallel residual lanes let attention and mlp read the same input. "
) * 40


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./data/micro")
    ap.add_argument("--tokenizer-out", default="./data/tokenizers")
    ap.add_argument("--vocab-size", type=int, default=256)
    ap.add_argument("--train-tokens", type=int, default=20_000)
    ap.add_argument("--val-tokens", type=int, default=8_000)
    args = ap.parse_args()

    out = Path(args.out)
    tokdir = Path(args.tokenizer_out)
    out.mkdir(parents=True, exist_ok=True)
    tokdir.mkdir(parents=True, exist_ok=True)
    text_file = out / "sample_text.txt"
    text_file.write_text(SAMPLE, encoding="utf-8")

    prefix = tokdir / "fineweb_1024_bpe"
    spm.SentencePieceTrainer.train(
        input=str(text_file),
        model_prefix=str(prefix),
        vocab_size=args.vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=0,
        unk_id=1,
        bos_id=-1,
        eos_id=2,
    )

    sp = spm.SentencePieceProcessor(model_file=str(prefix) + ".model")
    import torch

    base = torch.tensor(sp.encode(SAMPLE), dtype=torch.int64)
    for name, n in [
        ("fineweb_train_000000.bin", args.train_tokens),
        ("fineweb_val_000000.bin", args.val_tokens),
    ]:
        reps = n // base.numel() + 2
        tokens = base.repeat(reps)[:n].to(torch.int32)
        write_shard(out / name, tokens, extra_header={"synthetic": 1})
        print(f"{out / name}: {n:,} synthetic tokens")
    print(f"tokenizer: {prefix}.model")


if __name__ == "__main__":
    main()
