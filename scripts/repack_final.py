"""Repack final_a010 tar with architecture metadata added to MANIFEST.json."""

from __future__ import annotations

import json
import os
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); os.chdir(REPO)

from pgolf.artifacts import build_tar_gz

art = REPO / "artifacts/final_a010_submission.tar.gz"
with tarfile.open(art, "r:gz") as tar:
    members = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
wn = next(n for n in members if n.endswith("weights.ptz"))
old_meta = json.loads(members["MANIFEST.json"])

old_meta.update({
    "parallel_residual": True,
    "pr_from_layer": 4,
    "depth_recurrence": True,
    "recur_layers": "3,4",
    "model_dim": 128,
    "num_layers": 6,
    "num_heads": 4,
    "num_kv_heads": 2,
    "seq_len": 256,
})

data = build_tar_gz(
    REPO, members[wn], "data/tokenizers/fineweb_1024_bpe.model", old_meta,
)
art.write_bytes(data)
print("repacked", art.stat().st_size)
print(json.dumps(json.loads(tarfile.open(art).extractfile("MANIFEST.json").read()), indent=1))
