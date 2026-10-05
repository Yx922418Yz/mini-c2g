"""Artifact packing: code + compressed weights + tokenizer in one tar.gz.

The challenge counts every code byte and caps the artifact at decimal
16,000,000 bytes. The tar is written deterministically (fixed mtime, sorted
members) so two runs with identical inputs produce identical bytes.
"""

from __future__ import annotations

import io
import json
import tarfile
import time
from pathlib import Path

CODE_FILES = [
    "train_gpt.py",
    "eval_bpb.py",
    "prepare_local_data.py",
    "pack_submission.py",
    "pgolf/__init__.py",
    "pgolf/config.py",
    "pgolf/muon.py",
    "pgolf/data.py",
    "pgolf/bpb.py",
    "pgolf/model.py",
    "pgolf/quant.py",
    "pgolf/ttt.py",
    "pgolf/artifacts.py",
]


def collect_code(repo_root: str | Path) -> dict[str, bytes]:
    root = Path(repo_root)
    out: dict[str, bytes] = {}
    for rel in CODE_FILES:
        p = root / rel
        if p.exists():
            out[f"code/{rel}"] = p.read_bytes()
    return out


def build_tar_gz(
    repo_root: str | Path,
    weight_blob: bytes,
    tokenizer_path: str | Path,
    metadata: dict,
) -> bytes:
    members: dict[str, bytes] = collect_code(repo_root)
    members["weights.ptz"] = weight_blob
    members["tokenizer/" + Path(tokenizer_path).name] = Path(tokenizer_path).read_bytes()
    members["MANIFEST.json"] = json.dumps(metadata, indent=2, sort_keys=True).encode("utf-8")

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=9) as tar:
        for name in sorted(members):
            data = members[name]
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mtime = 1_700_000_000  # fixed for byte-reproducible archives
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def write_submission(
    repo_root: str | Path,
    out_path: str | Path,
    weight_blob: bytes,
    tokenizer_path: str | Path,
    metadata: dict,
    cap_bytes: int = 16_000_000,
) -> int:
    blob = build_tar_gz(repo_root, weight_blob, tokenizer_path, metadata)
    if len(blob) > cap_bytes:
        raise SizeGateError(
            f"artifact {len(blob):,} bytes exceeds cap {cap_bytes:,}"
        )
    Path(out_path).write_bytes(blob)
    return len(blob)


class SizeGateError(RuntimeError):
    pass


def write_submission_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
