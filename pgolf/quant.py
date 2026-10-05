"""Post-training quantization and compression.

Lineage (see docs/LiYaxuan_C2G_拿来说明.md):
- GPTQ, Frantar et al. (2023), arXiv:2210.17323: Hessian-guided column order.
- SDClip, Kevin Clark's 2026-04-05 parameter-golf record: clip = k * std(row),
  with int6 / k=12.85 for matrices and int8 / k=20 for embeddings.

Original contribution of this repository:
- A-SDClip: the clip multiplier k is chosen per row from the row's (shrinkage-
  estimated) excess kurtosis instead of being a single global value. Heavy-tailed
  rows get a wider clip to avoid outlier collapse; near-Gaussian rows keep the
  base k. See ``adaptive_k_table`` and docs/LiYaxuan_C2G_方案设计.md.
"""

from __future__ import annotations

import io
import zlib

import torch
from torch import Tensor

KEEP_FLOAT_MAX_NUMEL = 65_536
SCALE_DTYPE = torch.float16


# ---------------------------------------------------------------------------
# Per-row clipping
# ---------------------------------------------------------------------------

def row_std(t32: Tensor) -> Tensor:
    return t32.std(dim=1, unbiased=False).clamp_min(1e-8)


def excess_kurtosis(t32: Tensor) -> Tensor:
    """Sample excess kurtosis per row (NaN-safe)."""
    x = t32.float()
    m = x.size(1)
    mu = x.mean(dim=1, keepdim=True)
    d = x - mu
    s = d.square().mean(dim=1).clamp_min(1e-12)
    m4 = d.pow(4).mean(dim=1)
    k_biased = m4 / s.square() - 3.0
    # Bias correction for the normal-theory kurtosis estimator, then empirical
    # Bayes shrinkage toward 0 (variance of the estimator ~ 24/m for Gaussians).
    if m > 3:
        k_biased = ((m - 1) / ((m - 2) * (m - 3))) * ((m + 1) * k_biased + 6.0)
    lam = m / (m + 24.0)
    return k_biased * lam


def adaptive_k_table(
    t32: Tensor, k_base: float, alpha: float, k_min: float, k_max: float
) -> Tensor:
    kappa = excess_kurtosis(t32)
    k = k_base * torch.exp(alpha * kappa)
    return k.clamp(k_min, k_max)


def _q_from_clip(t32: Tensor, clip: Tensor, bits: int) -> tuple[Tensor, Tensor]:
    qmax = 2 ** (bits - 1) - 1
    scale = (clip / qmax).clamp_min(1.0 / qmax)
    q = torch.clamp(torch.round(t32 / scale[:, None]), -qmax, qmax)
    return q.to(torch.int8), scale


def quantize_rows(
    t: Tensor,
    bits: int,
    scheme: str,
    k_base: float,
    alpha: float = 0.15,
    k_min: float = 8.0,
    k_max: float = 20.0,
):
    t32 = t.float()
    std = row_std(t32)
    if scheme == "rtn_quantile":
        # Baseline style: per-row high quantile clip.
        clip = torch.quantile(t32.abs(), 0.9999984, dim=1)
    elif scheme == "sdclip":
        clip = k_base * std
    elif scheme == "asdclip":
        k = adaptive_k_table(t32, k_base, alpha, k_min, k_max)
        clip = k * std
    else:
        raise ValueError(f"unknown clip scheme {scheme}")
    q, scale = _q_from_clip(t32, clip, bits)
    return q.contiguous(), scale.to(SCALE_DTYPE).contiguous()


# ---------------------------------------------------------------------------
# GPTQ-lite: Hessian-guided quantization for one 2D weight matrix
# ---------------------------------------------------------------------------

@torch.no_grad()
def gptq_quantize_matrix(
    w: Tensor,
    calib_x: Tensor,
    bits: int,
    scheme: str,
    k_base: float,
    alpha: float,
    k_min: float,
    k_max: float,
    blocksize: int = 128,
    damp: float = 0.01,
):
    t32 = w.float()
    qmax = 2 ** (bits - 1) - 1
    std = row_std(t32)
    if scheme == "asdclip":
        k = adaptive_k_table(t32, k_base, alpha, k_min, k_max)
        clip = k * std
    else:
        clip = k_base * std
    scale = (clip / qmax).clamp_min(1.0 / qmax)

    x = calib_x.detach().cpu().float().reshape(-1, calib_x.size(-1)).double()
    H0 = x.T @ x
    # Standard GPTQ dead-feature handling: columns with zero Hessian diagonal carry
    # no calibration signal; pin their pivot to 1 and zero the matching weights.
    dead = torch.diag(H0) <= 1e-12
    H0[dead, dead] = 1.0
    t32 = t32.clone()
    t32[:, dead] = 0.0

    # Adaptive dampening: retry with progressively larger diagonal damp if the
    # factorization is still numerically non-positive-definite.
    damp_amt = float(torch.diag(H0).mean()) * damp
    Hinv = None
    for _ in range(8):
        H = H0 + damp_amt * torch.eye(H0.size(0), dtype=torch.float64)
        try:
            L = torch.linalg.cholesky(H)
            Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True).float()
            break
        except torch.linalg.LinAlgError:
            damp_amt *= 4.0
    if Hinv is None:
        raise RuntimeError("GPTQ Hessian could not be made positive-definite")
    Hd = torch.diag(Hinv).clamp_min(1e-8)

    Wq = torch.zeros_like(t32)
    for i in range(0, t32.size(1), blocksize):
        j = min(i + blocksize, t32.size(1))
        qcnt = torch.round(t32[:, i:j] / scale[:, None]).clamp(-qmax, qmax)
        qvals = qcnt * scale[:, None]
        Wq[:, i:j] = qvals
        err = (t32[:, i:j] - qvals) / Hd[i:j][None, :]
        if j < t32.size(1):
            t32[:, j:] -= err @ Hinv[i:j, j:]
    return Wq.contiguous(), scale.to(SCALE_DTYPE).contiguous()


# ---------------------------------------------------------------------------
# State-dict export / import
# ---------------------------------------------------------------------------

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def quantize_state_dict(cfg, state_dict: dict[str, Tensor], calib: dict[str, Tensor] | None = None):
    scheme_map = {
        "int8": (8, "rtn_quantile", cfg.sdclip_k_int8),
        "int6_sdclip": (6, "sdclip", cfg.sdclip_k_int6),
        "int6_asdclip": (6, "asdclip", cfg.sdclip_k_int6),
    }
    bits, scheme, k_base = scheme_map[cfg.quant_scheme]
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    stats = {"param_count": 0, "payload_bytes": 0, "num_quant_tensors": 0}

    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        stats["param_count"] += int(t.numel())
        if not t.is_floating_point():
            passthrough[name] = t
            stats["payload_bytes"] += tensor_nbytes(t)
            continue
        # All real 2D weight matrices are quantized, including the small matrices
        # of the micro model. Only non-2D parameters and trivially small 2D
        # buffers stay at higher precision.
        if t.ndim != 2 or t.numel() <= 64:
            kept = t.to(torch.float16 if t.numel() > 256 else t.dtype).contiguous()
            passthrough[name] = kept
            stats["payload_bytes"] += tensor_nbytes(kept)
            continue
        is_embedding = "tok_emb" in name
        b = 8 if is_embedding else bits
        if is_embedding:
            q, s = quantize_rows(t, 8, "sdclip", cfg.sdclip_k_int8)
        elif calib is not None and name in calib:
            wq, s = gptq_quantize_matrix(
                t, calib[name], b, scheme, k_base, cfg.asdclip_alpha,
                cfg.asdclip_k_min, cfg.asdclip_k_max,
            )
            q = torch.clamp(torch.round(wq / s[:, None]), -2 ** (b - 1) + 1,
                            2 ** (b - 1) - 1).to(torch.int8)
        else:
            q, s = quantize_rows(t, b, scheme, k_base, cfg.asdclip_alpha,
                                 cfg.asdclip_k_min, cfg.asdclip_k_max)
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)
        stats["num_quant_tensors"] += 1

    obj = {
        "__quant_format__": f"pgolf_{cfg.quant_scheme}_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    return obj, stats


def dequantize_state_dict(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    for name, q in obj["quantized"].items():
        s = obj["scales"][name].float()
        dtype = getattr(torch, obj["dtypes"][name])
        out[name] = (q.float() * s[:, None]).to(dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out[name] = t
    return out


def serialize_compressed(obj, level: int = 9) -> tuple[bytes, int]:
    buf = io.BytesIO()
    torch.save(obj, buf)
    raw = buf.getvalue()
    return zlib.compress(raw, level), len(raw)


def decompress_state(blob: bytes):
    return torch.load(io.BytesIO(zlib.decompress(blob)), map_location="cpu")
