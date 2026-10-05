"""Diagnose GPTQ calibration: finiteness of activations and Hessian eigenvalues."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

import torch

from pgolf.config import Config
from pgolf.data import LocalTokenLoader
from pgolf.model import GPT
from pgolf.bpb import build_sentencepiece_luts
from train_gpt import collect_calibration, build_optimizers, warmdown_scale

os.environ["SEED"] = "1337"
os.environ["ITERATIONS"] = "400"
os.environ["PARALLEL_RESIDUAL"] = "1"
os.environ["DEPTH_RECURRENCE"] = "1"
os.environ["QUANT_SCHEME"] = "int6_asdclip"
os.environ["TRAIN_LOG_EVERY"] = "100"

cfg = Config()
device = torch.device("cpu")
torch.manual_seed(cfg.seed)

import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file=cfg.tokenizer_path)
luts = build_sentencepiece_luts(sp, cfg.vocab_size, device)
model = GPT(cfg).to(device)
print("params", sum(p.numel() for p in model.parameters()))

opts, _muon, base_lrs = build_optimizers(cfg, model, device)
loader = LocalTokenLoader(os.path.join(cfg.data_path, cfg.train_glob), device)
model.train()
for step in range(1, cfg.iterations + 2):
    x, y = loader.next_batch(cfg.batch_tokens, cfg.seq_len)
    scale = warmdown_scale(step - 1, cfg.iterations, cfg.warmdown_iters)
    li = iter(base_lrs)
    for o in opts:
        for g in o.param_groups:
            g["lr"] = next(li) * scale
    for o in opts:
        o.zero_grad(set_to_none=True)
    loss = model(x, y)
    loss.backward()
    for o in opts:
        o.step()
    if step % 100 == 0:
        print("step", step, "loss", float(loss))

model.eval()
calib = collect_calibration(cfg, model, device, batches=3)
print("calib layers:", len(calib))
bad = 0
for name, X in calib.items():
    fin = torch.isfinite(X).all().item()
    xmin, xmax = float(X.min()), float(X.max())
    xd = X.double().reshape(-1, X.size(-1))
    H = xd.T @ xd
    md = torch.diag(H).mean()
    H += torch.diag(md * 0.01 * torch.eye(H.size(0), dtype=torch.float64))
    eig = torch.linalg.eigvalsh(H)
    flag = "" if (fin and float(eig.min()) > 0) else "  <<< PROBLEM"
    if flag:
        bad += 1
    print(f"{name:22s} finite={fin} min={xmin:.3g} max={xmax:.3g} "
          f"min_eig={float(eig.min()):.3g}{flag}")
print("problem layers:", bad)
