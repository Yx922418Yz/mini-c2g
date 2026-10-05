"""Parameter Golf — train, evaluate, quantize, pack.

One script, two paths:
- H100 leaderboard:  torchrun --standalone --nproc_per_node=8 train_gpt.py
  (override the micro defaults via env vars; see scripts/reproduce_h100.sh)
- Local CPU smoke:   python train_gpt.py
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import sentencepiece as spm
import torch

from pgolf.artifacts import write_submission, write_submission_json
from pgolf.bpb import build_sentencepiece_luts, eval_bpb
from pgolf.config import Config
from pgolf.data import (
    DistributedTokenLoader,
    LocalTokenLoader,
    load_split_tokens,
)
from pgolf.model import CONTROL_PATTERNS, CastedLinear, GPT
from pgolf.muon import Muon
from pgolf.quant import (
    dequantize_state_dict,
    quantize_state_dict,
    serialize_compressed,
)
from pgolf.ttt import eval_ttt


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        import numpy as np

        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)


def warmdown_scale(step: int, iterations: int, warmdown: int) -> float:
    if warmdown <= 0 or step < iterations - warmdown:
        return 1.0
    return max((iterations - step) / warmdown, 0.0)


def build_optimizers(cfg: Config, model: GPT, device: torch.device):
    block_params = list(model.blocks.named_parameters())
    matrix = [p for n, p in block_params if p.ndim == 2 and not any(c in n for c in CONTROL_PATTERNS)]
    scalar = [p for n, p in block_params if p.ndim < 2 or any(c in n for c in CONTROL_PATTERNS)]
    scalar.append(model.skip_weights)
    fused = device.type == "cuda"
    opts = []
    tok_lr = cfg.tied_embed_lr if cfg.tie_embeddings else cfg.embed_lr
    opts.append(
        torch.optim.Adam([model.tok_emb.weight], lr=tok_lr, betas=(cfg.beta1, cfg.beta2),
                         eps=cfg.adam_eps, fused=fused, weight_decay=0.0)
    )
    if model.lm_head is not None:
        opts.append(
            torch.optim.Adam(model.lm_head.parameters(), lr=cfg.head_lr, betas=(cfg.beta1, cfg.beta2),
                             eps=cfg.adam_eps, fused=fused)
        )
    muon = None
    if cfg.use_muon and matrix:
        muon = Muon(matrix, lr=cfg.matrix_lr, momentum=cfg.muon_momentum,
                    backend_steps=cfg.muon_backend_steps)
        opts.append(muon)
    opts.append(
        torch.optim.Adam(scalar, lr=cfg.scalar_lr, betas=(cfg.beta1, cfg.beta2),
                         eps=cfg.adam_eps, fused=fused, weight_decay=cfg.weight_decay)
    )
    base_lrs = [g["lr"] for o in opts for g in o.param_groups]
    return opts, muon, base_lrs


@torch.no_grad()
def update_ema(ema_state: dict, model: torch.nn.Module, decay: float) -> None:
    for k, v in model.state_dict().items():
        if v.is_floating_point():
            ema_state[k].mul_(decay).add_(v.detach(), alpha=1 - decay)
        else:
            ema_state[k].copy_(v)


def collect_calibration(cfg: Config, model: GPT, device, batches: int = 4) -> dict:
    """Capture Linear input activations on training data for GPTQ-lite."""
    captured: dict[str, torch.Tensor] = {}
    handles = []
    for mod_name, mod in model.named_modules():
        if isinstance(mod, CastedLinear):
            key = f"{mod_name}.weight"

            def hook(_m, inp, _out, key=key):
                x = inp[0].detach().cpu().reshape(-1, inp[0].size(-1))
                if key in captured:
                    x = torch.cat([captured[key], x])[-4096:]
                captured[key] = x[-4096:]

            handles.append(mod.register_forward_hook(hook))
    loader = LocalTokenLoader(os.path.join(cfg.data_path, cfg.train_glob), device)
    model.eval()
    for _ in range(batches):
        x, _y = loader.next_batch(cfg.batch_tokens, cfg.seq_len)
        with torch.no_grad():
            model(x, _y)
    for h in handles:
        h.remove()
    model.train()
    return captured


def main() -> None:
    cfg = Config()
    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)
    Path("logs").mkdir(exist_ok=True)
    log_path = Path(f"logs/{cfg.run_id}.jsonl")

    def log(event: str, **kw) -> None:
        rec = {"event": event, "time": round(time.time(), 3), **kw}
        line = json.dumps(rec, ensure_ascii=False)
        print(line)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    set_seed(cfg.seed)
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ and torch.cuda.is_available()
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    grad_accum = max(8 // world, 1) if distributed else 1
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if distributed:
            import torch.distributed as dist

            dist.init_process_group(backend="nccl", device_id=device)
            dist.barrier()
    log("start", run_id=cfg.run_id, seed=cfg.seed, device=str(device),
        world_size=world, python=sys.version.split()[0], torch=torch.__version__)

    # Tokenizer + byte LUTs
    sp = spm.SentencePieceProcessor(model_file=cfg.tokenizer_path)
    luts = build_sentencepiece_luts(sp, cfg.vocab_size, device)

    # Model
    model = GPT(cfg)
    if device.type == "cuda":
        model = model.to(device).bfloat16()
        for m in model.modules():
            if isinstance(m, CastedLinear):
                m.float()
        model = model.to(device)
    else:
        model = model.to(device)
    train_model = model
    if device.type == "cuda":
        compiled = torch.compile(model, dynamic=False, fullgraph=True)
        if distributed:
            from torch.nn.parallel import DistributedDataParallel as DDP

            train_model = DDP(compiled, device_ids=[local_rank], broadcast_buffers=False)
        else:
            train_model = compiled

    opts, muon, base_lrs = build_optimizers(cfg, model, device)
    n_params = sum(p.numel() for p in model.parameters())
    log("model", params=n_params, virtual_layers=model.virtual_layers(),
        parallel_residual=cfg.parallel_residual, depth_recurrence=cfg.depth_recurrence)

    loader = (
        DistributedTokenLoader(os.path.join(cfg.data_path, cfg.train_glob), rank, world, device)
        if distributed
        else LocalTokenLoader(os.path.join(cfg.data_path, cfg.train_glob), device)
    )

    ema_state = None
    if cfg.ema_decay > 0:
        ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # Training loop
    t0 = time.perf_counter()
    train_model.train()
    step = 0
    stop = False
    while not stop:
        stop = step >= cfg.iterations
        scale = warmdown_scale(step, cfg.iterations, cfg.warmdown_iters)
        li = iter(base_lrs)
        for o in opts:
            for g in o.param_groups:
                g["lr"] = next(li) * scale
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for micro in range(grad_accum):
            if distributed:
                train_model.require_backward_grad_sync = micro == grad_accum - 1
            x, y = loader.next_batch(cfg.batch_tokens, cfg.seq_len, grad_accum) if distributed else loader.next_batch(
                cfg.batch_tokens, cfg.seq_len
            )
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = train_model(x, y)
                (loss / grad_accum).backward()
            else:
                loss = train_model(x, y)
                loss.backward()
            loss_acc += float(loss.detach()) / grad_accum
        if cfg.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        for o in opts:
            o.step()
        if ema_state is not None:
            update_ema(ema_state, model, cfg.ema_decay)
        step += 1
        if step <= 5 or step % cfg.train_log_every == 0 or stop:
            elapsed = time.perf_counter() - t0
            log("train", step=step, loss=round(loss_acc, 5), lr_scale=round(scale, 4),
                elapsed_s=round(elapsed, 2))
        if cfg.max_wallclock_seconds > 0 and time.perf_counter() - t0 > cfg.max_wallclock_seconds:
            log("wallclock_stop", step=step)
            stop = True
    train_time = time.perf_counter() - t0

    eval_model = model
    if ema_state is not None:
        model.load_state_dict(ema_state)
        eval_model = model

    val_tokens = load_split_tokens(
        os.path.join(cfg.data_path, cfg.val_glob), cfg.seq_len, cfg.val_limit_tokens
    )
    log("eval_start", val_tokens=val_tokens.numel() - 1)
    pre_loss, pre_bpb = eval_bpb(eval_model, val_tokens, cfg.seq_len, luts, device,
                                 cfg.val_batch_tokens, rank, world)
    log("eval_prequant", val_loss=round(pre_loss, 5), val_bpb=round(pre_bpb, 5))

    # GPTQ-lite calibration + quantization
    calib = None
    if cfg.quant_scheme != "int8":
        calib = collect_calibration(cfg, eval_model, device, batches=3)
    quant_obj, qstats = quantize_state_dict(cfg, eval_model.state_dict(), calib)
    weight_blob, raw_len = serialize_compressed(quant_obj, cfg.compression_level)
    log("quant", scheme=cfg.quant_scheme, weight_blob_bytes=len(weight_blob),
        raw_bytes=raw_len, payload_bytes=qstats["payload_bytes"],
        quant_tensors=qstats["num_quant_tensors"])

    deq = dequantize_state_dict(quant_obj)
    model.load_state_dict(deq, strict=False)
    post_loss, post_bpb = eval_bpb(model, val_tokens, cfg.seq_len, luts, device,
                                   cfg.val_batch_tokens, rank, world)
    log("eval_postquant", val_loss=round(post_loss, 5), val_bpb=round(post_bpb, 5))

    ttt_bpb = None
    if cfg.ttt_enabled:
        ttt_loss, ttt_bpb = eval_ttt(
            model, val_tokens, cfg.seq_len, luts, device,
            cfg.ttt_chunk_tokens, cfg.ttt_epochs, cfg.ttt_lr,
            cfg.ttt_momentum, cfg.ttt_grad_clip,
            log=lambda m: log("ttt", msg=m),
        )
        log("eval_ttt", val_loss=round(ttt_loss, 5), val_bpb=round(ttt_bpb, 5))

    # Pack artifact (size gate enforced)
    metadata = {
        "author": "Li Yaxuan",
        "github_id": "liyaxuan",
        "run_id": cfg.run_id,
        "seed": cfg.seed,
        "quant_scheme": cfg.quant_scheme,
        "parallel_residual": cfg.parallel_residual,
        "pr_from_layer": cfg.pr_from_layer,
        "depth_recurrence": cfg.depth_recurrence,
        "recur_layers": cfg.recur_layers,
        "model_dim": cfg.model_dim,
        "num_layers": cfg.num_layers,
        "num_heads": cfg.num_heads,
        "num_kv_heads": cfg.num_kv_heads,
        "seq_len": cfg.seq_len,
        "bpb_pre_quant": round(pre_bpb, 6),
        "bpb_post_quant": round(post_bpb, 6),
        "bpb_ttt": round(ttt_bpb, 6) if ttt_bpb is not None else None,
        "train_time_s": round(train_time, 2),
        "num_params": n_params,
        "hardware": "CPU micro run" if device.type == "cpu" else "8xH100 SXM",
    }
    art_path = Path(f"artifacts/{cfg.run_id}_submission.tar.gz")
    art_path.parent.mkdir(exist_ok=True)
    size = write_submission(repo_root, art_path, weight_blob, cfg.tokenizer_path,
                           metadata, cfg.artifact_bytes_cap)
    log("artifact", path=str(art_path), bytes=size)
    write_submission_json(art_path.with_suffix(".json"), metadata)

    if distributed:
        import torch.distributed as dist

        dist.destroy_process_group()
    log("done", train_time_s=round(train_time, 2))


if __name__ == "__main__":
    main()
