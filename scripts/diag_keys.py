import io, json, os, sys, tarfile
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); os.chdir(REPO)
import torch
from pgolf.config import Config
from pgolf.model import GPT
from pgolf.quant import decompress_state, dequantize_state_dict

cfg = Config()
art = REPO / "artifacts/final_a010_submission.tar.gz"
with tarfile.open(art, "r:gz") as tar:
    members = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
wn = next(n for n in members if n.endswith("weights.ptz"))
obj = decompress_state(members[wn])
state = dequantize_state_dict(obj)
fresh = GPT(cfg).state_dict()
missing = [k for k in fresh if k not in state]
extra = [k for k in state if k not in fresh]
print("missing keys:", missing)
print("extra keys:", extra)
for k in missing:
    print(k, tuple(fresh[k].shape), fresh[k].dtype)
