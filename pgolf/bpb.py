"""Tokenizer-agnostic Bits-Per-Byte evaluation.

BPB converts token-level cross entropy into a byte-level compression rate so
that submissions with different tokenizers are comparable:

    bpb = (mean_token_nll / ln 2) * (num_tokens / num_original_bytes)

The number of original UTF-8 bytes represented by each target token is rebuilt
from the SentencePiece piece table, including the leading-space convention.
"""

from __future__ import annotations

import math

import numpy as np
import sentencepiece as spm
import torch
from torch import nn

try:
    import torch.distributed as dist
except Exception:
    dist = None  # type: ignore[assignment]


def build_sentencepiece_luts(sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device):
    sp_vocab = int(sp.vocab_size())
    size = max(sp_vocab, vocab_size)
    base_bytes = np.zeros(size, dtype=np.int16)
    leading_space = np.zeros(size, dtype=np.bool_)
    is_boundary = np.ones(size, dtype=np.bool_)
    for tok in range(sp_vocab):
        if sp.is_control(tok) or sp.is_unknown(tok) or sp.is_unused(tok):
            continue
        is_boundary[tok] = False
        if sp.is_byte(tok):
            base_bytes[tok] = 1
            continue
        piece = sp.id_to_piece(tok)
        if piece.startswith("▁"):
            leading_space[tok] = True
            piece = piece[1:]
        base_bytes[tok] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes, dtype=torch.int16, device=device),
        torch.tensor(leading_space, dtype=torch.bool, device=device),
        torch.tensor(is_boundary, dtype=torch.bool, device=device),
    )


@torch.no_grad()
def eval_bpb(
    model: nn.Module,
    val_tokens: torch.Tensor,
    seq_len: int,
    luts,
    device: torch.device,
    batch_tokens: int = 65_536,
    rank: int = 0,
    world_size: int = 1,
    autocast_cuda: bool = True,
):
    base_bytes_lut, leading_space_lut, boundary_lut = luts
    total_seqs = (val_tokens.numel() - 1) // seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    batch_seqs = max(batch_tokens // seq_len, 1)

    loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    token_count = torch.zeros((), dtype=torch.float64, device=device)
    byte_count = torch.zeros((), dtype=torch.float64, device=device)

    model.eval()
    for bstart in range(seq_start, seq_end, batch_seqs):
        bend = min(bstart + batch_seqs, seq_end)
        raw_start = bstart * seq_len
        raw_end = bend * seq_len + 1
        local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_cuda):
                batch_loss = model(x, y).detach()
        else:
            batch_loss = model(x, y).detach()
        n = float(y.numel())
        loss_sum += batch_loss.to(torch.float64) * n
        token_count += n
        prev = x.reshape(-1)
        tgt = y.reshape(-1)
        nbytes = base_bytes_lut[tgt].to(torch.int16)
        nbytes += (leading_space_lut[tgt] & ~boundary_lut[prev]).to(torch.int16)
        byte_count += nbytes.to(torch.float64).sum()

    if dist is not None and dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    val_loss = loss_sum / token_count
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = token_count / byte_count
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)
