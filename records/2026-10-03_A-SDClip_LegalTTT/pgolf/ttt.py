"""Legal score-first test-time training (TTT).

Compliance conditions (matching the challenge's Track-B interpretation, see the
2026-04-09 top record and its references):
1. Causality: every position is scored from prefix tokens only, one pass.
2. Normalized distribution: standard softmax, no logit biasing or n-gram cache.
3. Score before update: each chunk is fully scored under no_grad BEFORE any SGD.
4. Single pass: each token is scored exactly once; no rescoring.

Adaptation on chunk c therefore only affects tokens of later chunks, which is
legal: those tokens have not been scored yet.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .bpb import build_sentencepiece_luts  # noqa: F401  (re-export convenience)


@torch.no_grad()
def _score_chunk(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return model(x, y).detach()


def _chunk_xy(tokens: torch.Tensor, seq_len: int, start_seq: int, n_seq: int, device):
    a = start_seq * seq_len
    b = (start_seq + n_seq) * seq_len + 1
    local = tokens[a:b].to(device=device, dtype=torch.int64)
    return local[:-1].reshape(-1, seq_len), local[1:].reshape(-1, seq_len)


def eval_ttt(
    model: nn.Module,
    val_tokens: torch.Tensor,
    seq_len: int,
    luts,
    device: torch.device,
    chunk_tokens: int = 8_192,
    epochs: int = 3,
    lr: float = 0.005,
    momentum: float = 0.9,
    grad_clip: float = 1.0,
    log=lambda *_: None,
):
    base_bytes_lut, leading_space_lut, boundary_lut = luts
    total_seqs = (val_tokens.numel() - 1) // seq_len
    chunk_seqs = max(chunk_tokens // seq_len, 1)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    n_chunks = (total_seqs + chunk_seqs - 1) // chunk_seqs
    # cosine decay of the adaptation lr across chunks
    base_lr = lr

    loss_sum = torch.zeros((), dtype=torch.float64)
    token_count = torch.zeros((), dtype=torch.float64)
    byte_count = torch.zeros((), dtype=torch.float64)

    model.eval()
    for c in range(n_chunks):
        s0 = c * chunk_seqs
        n = min(chunk_seqs, total_seqs - s0)
        x, y = _chunk_xy(val_tokens, seq_len, s0, n, device)

        # (1) score the whole chunk before any update
        with torch.no_grad():
            chunk_loss = model(x, y).detach()
        ntok = float(y.numel())
        loss_sum += chunk_loss.to(torch.float64) * ntok
        token_count += ntok
        prev = x.reshape(-1)
        tgt = y.reshape(-1)
        nbytes = base_bytes_lut[tgt].to(torch.int16)
        nbytes += (leading_space_lut[tgt] & ~boundary_lut[prev]).to(torch.int16)
        byte_count += nbytes.to(torch.float64).sum()

        # (2) adapt on the already-scored chunk
        model.train()
        for ep in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = model(x, y)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        model.eval()
        for g in optimizer.param_groups:
            g["lr"] = base_lr * 0.5 * (1.0 + math.cos(math.pi * (c + 1) / n_chunks))
        log(f"ttt_chunk:{c + 1}/{n_chunks} scored_loss:{float(chunk_loss):.4f}")

    val_loss = loss_sum / token_count
    bpb = float((val_loss / math.log(2.0)) * (token_count / byte_count))
    model.train()
    return float(val_loss.item()), bpb
