"""Inspect and gate-check a packed submission artifact.

Usage:
    python pack_submission.py --check artifacts/<run>_submission.tar.gz
"""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

CAP = 16_000_000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", required=True)
    ap.add_argument("--cap", type=int, default=CAP)
    args = ap.parse_args()

    p = Path(args.check)
    size = p.stat().st_size
    print(f"artifact: {p}")
    print(f"size: {size:,} bytes (cap {args.cap:,})")
    with tarfile.open(p, "r:gz") as tar:
        for m in sorted(tar.getmembers(), key=lambda m: m.name):
            print(f"  {m.size:>10,}  {m.name}")
        manifest_member = next((m for m in tar.getmembers() if m.name == "MANIFEST.json"), None)
        if manifest_member is not None:
            manifest = json.load(tar.extractfile(manifest_member))
            print("manifest:")
            print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if size > args.cap:
        raise SystemExit(f"FAIL: exceeds cap by {size - args.cap:,} bytes")
    print("PASS: within size cap")


if __name__ == "__main__":
    main()
