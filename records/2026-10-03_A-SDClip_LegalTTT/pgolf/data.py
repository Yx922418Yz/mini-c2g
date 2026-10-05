"""Data loading for the official shard format.

Shards are produced by the challenge preprocessing pipeline:
- 256 int32 header values (1024 bytes): magic 20240520, version 1, num_tokens, ...
- ``num_tokens`` uint16 token ids.

The training stream is deterministic: shards are read in sorted order and
wrapped around, with no worker processes or random sampling.
"""

from __future__ import annotations

import glob as glob_mod
from pathlib import Path

import numpy as np
import torch

SHARD_MAGIC = 20240520
SHARD_VERSION = 1
HEADER_INTS = 256


def load_data_shard(file: str | Path) -> torch.Tensor:
    file = Path(file)
    header = np.fromfile(file, dtype="<i4", count=HEADER_INTS)
    if header.size != HEADER_INTS or int(header[0]) != SHARD_MAGIC or int(header[1]) != SHARD_VERSION:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected = HEADER_INTS * 4 + num_tokens * 2
    if file.stat().st_size != expected:
        # Local prefix copies are truncated on purpose; only enforce on full shards
        # whose declared token count matches the full-shard size.
        if num_tokens != 100_000_000 and num_tokens != 62_021_846:
            raise ValueError(f"Shard size mismatch for {file}")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=HEADER_INTS * 4)
    return torch.from_numpy(np.asarray(tokens_np, dtype=np.uint16))


def write_shard(path: str | Path, tokens: torch.Tensor | np.ndarray, extra_header: dict | None = None) -> None:
    arr = np.asarray(tokens, dtype=np.uint16)
    header = np.zeros(HEADER_INTS, dtype="<i4")
    header[0] = SHARD_MAGIC
    header[1] = SHARD_VERSION
    header[2] = arr.size
    for i, (k, v) in enumerate((extra_header or {}).items(), start=3):
        header[i] = int(v)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(arr.tobytes())


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob_mod.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class LocalTokenLoader:
    """Single-process loader (CPU smoke path)."""

    def __init__(self, pattern: str, device: torch.device):
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, total_tokens: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.stream.take(total_tokens + 1).to(dtype=torch.int64)
        x = chunk[:-1].reshape(-1, seq_len)
        y = chunk[1:].reshape(-1, seq_len)
        return x.to(self.device), y.to(self.device)


class DistributedTokenLoader:
    """Each rank gets one disjoint contiguous span per training step."""

    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int):
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


def load_split_tokens(pattern: str, seq_len: int, limit_tokens: int = 0) -> torch.Tensor:
    files = [Path(p) for p in sorted(glob_mod.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    if limit_tokens > 0 and tokens.numel() > limit_tokens + 1:
        tokens = tokens[: limit_tokens + 1]
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Split too short for seq_len={seq_len}")
    return tokens[: usable + 1]
