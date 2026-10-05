"""Build the local micro dataset from official shard prefixes.

The prefix files are byte-range downloads of the official FineWeb shards
(see data/README.md). They contain real FineWeb tokens; this script rewrites
them into correctly-headered micro shards and copies the official tokenizer.

Usage:
    python prepare_local_data.py --prefix-dir <dir-with-train_head.bin> \\
        --tokenizer <fineweb_1024_bpe.model>
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from pgolf.data import load_data_shard, write_shard


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix-dir", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", default="./data")
    args = ap.parse_args()

    prefix = Path(args.prefix_dir)
    out = Path(args.out)
    micro = out / "micro"
    tokdir = out / "tokenizers"
    micro.mkdir(parents=True, exist_ok=True)
    tokdir.mkdir(parents=True, exist_ok=True)

    for src_name, dst_name in [
        ("train_head.bin", "fineweb_train_000000.bin"),
        ("val_head.bin", "fineweb_val_000000.bin"),
    ]:
        tokens = load_data_shard(prefix / src_name)
        dst = micro / dst_name
        write_shard(dst, tokens, extra_header={"micro": 1})
        print(f"{dst}: {tokens.numel():,} tokens")

    tok_dst = tokdir / Path(args.tokenizer).name
    shutil.copy2(args.tokenizer, tok_dst)
    print(f"{tok_dst}: tokenizer copied")


if __name__ == "__main__":
    main()
