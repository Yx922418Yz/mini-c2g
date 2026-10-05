"""Standalone BPB evaluator for a packed artifact.

Usage:
    python eval_bpb.py --artifact artifacts/<run>_submission.tar.gz
    python eval_bpb.py --artifact <file> --ttt --val-limit-tokens 200000
"""

from __future__ import annotations

import argparse
import io
import json
import os
import tarfile

import sentencepiece as spm
import torch

from pgolf.bpb import build_sentencepiece_luts, eval_bpb
from pgolf.config import Config
from pgolf.data import load_split_tokens
from pgolf.model import CastedLinear, GPT
from pgolf.quant import decompress_state, dequantize_state_dict
from pgolf.ttt import eval_ttt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--ttt", action="store_true")
    ap.add_argument("--ttt-bf16", action="store_true",
                    help="run TTT in bf16 (default follows manifest / fp32)")
    ap.add_argument("--val-limit-tokens", type=int, default=0)
    ap.add_argument("--seq-len", type=int, default=None)
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo_root)
    artifact_path = os.path.abspath(args.artifact)

    cfg = Config()
    if args.seq_len is not None:
        cfg.seq_len = args.seq_len
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    with tarfile.open(artifact_path, "r:gz") as tar:
        members = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
    weight_name = next(n for n in members if n.endswith("weights.ptz"))
    tok_name = next(n for n in members if n.startswith("tokenizer/"))
    manifest = json.loads(members["MANIFEST.json"])

    # The artifact is self-describing: restore the architecture it was trained with
    for k in ("parallel_residual", "depth_recurrence"):
        setattr(cfg, k, bool(manifest.get(k, getattr(cfg, k))))
    for k in ("pr_from_layer", "model_dim", "num_layers", "num_heads",
              "num_kv_heads", "seq_len"):
        if k in manifest:
            setattr(cfg, k, manifest[k])
    if "recur_layers" in manifest:
        cfg.recur_layers = manifest["recur_layers"]

    os.makedirs(os.path.dirname(cfg.tokenizer_path), exist_ok=True)
    if not os.path.exists(cfg.tokenizer_path):  # extract bundled tokenizer once
        with open(cfg.tokenizer_path, "wb") as f:
            f.write(members[tok_name])
    sp = spm.SentencePieceProcessor(model_file=cfg.tokenizer_path)
    luts = build_sentencepiece_luts(sp, cfg.vocab_size, device)

    quant_obj = decompress_state(members[weight_name])
    state = dequantize_state_dict(quant_obj)

    model = GPT(cfg).to(device)
    model.load_state_dict(state, strict=False)

    val_tokens = load_split_tokens(
        os.path.join(cfg.data_path, cfg.val_glob), cfg.seq_len,
        args.val_limit_tokens or cfg.val_limit_tokens,
    )
    val_loss, bpb = eval_bpb(model, val_tokens, cfg.seq_len, luts, device,
                             cfg.val_batch_tokens)
    result = {"artifact": args.artifact, "val_loss": val_loss, "val_bpb": bpb,
              "manifest_bpb_post_quant": manifest.get("bpb_post_quant")}
    if args.ttt or cfg.ttt_enabled:
        use_fp32 = not args.ttt_bf16 and manifest.get("ttt_force_fp32", True)
        if device.type == "cuda":
            if use_fp32:
                model.float()
            else:
                # replicate the training pipeline: bf16 model, fp32 linears
                model.bfloat16()
                for m in model.modules():
                    if isinstance(m, CastedLinear):
                        m.float()
        ttt_loss, ttt_bpb = eval_ttt(
            model, val_tokens, cfg.seq_len, luts, device,
            cfg.ttt_chunk_tokens, cfg.ttt_epochs, cfg.ttt_lr,
            cfg.ttt_momentum, cfg.ttt_grad_clip,
        )
        result["ttt_loss"] = ttt_loss
        result["ttt_bpb"] = ttt_bpb
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
