"""Transformer model.

The baseline is the modded-nanoGPT descendant shipped by openai/parameter-golf:
tied embeddings, RMSNorm, RoPE, QK RMSNorm with a learnable per-head gain, GQA,
relu^2 MLP, learned per-lane residual scales, and a U-Net skip pattern between
the first and second half of the stack.

Two ablation switches are implemented here:
- parallel residual lanes (GPT-J style; Wang & Komatsuzaki, 2021)
- depth recurrence via adjacent block repeats (Universal Transformer style;
  Dehghani et al., 2019)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

CONTROL_PATTERNS = (
    "attn_scale",
    "mlp_scale",
    "resid_mix",
    "q_gain",
    "skip_weights",
)


class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    """fp32 master weights; cast to the activation dtype at matmul time."""

    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10_000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq = 0
        self._cos: Tensor | None = None
        self._sin: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if self._cos is None or self._sin is None or self._seq != seq_len or self._cos.device != device:
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos = freqs.cos()[None, None, :, :]
            self._sin = freqs.sin()[None, None, :, :]
            self._seq = seq_len
        return self._cos.to(dtype), self._sin.to(dtype)


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        if dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, rope_base)

    def forward(self, x: Tensor) -> Tensor:
        b, s, _ = x.shape
        q = self.c_q(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(s, x.device, q.dtype)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        q = q * self.q_gain.to(q.dtype)[None, :, None, None]
        if self.num_kv_heads != self.num_heads:
            if x.device.type == "cuda":
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            else:
                rep = self.num_heads // self.num_kv_heads
                k = k.repeat_interleave(rep, dim=1)
                v = v.repeat_interleave(rep, dim=1)
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, s, -1)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
                 rope_base: float, qk_gain_init: float, parallel: bool = False):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.parallel = parallel

    def _mix(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(x.dtype)
        return mix[0] * x + mix[1] * x0

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        x_in = self._mix(x, x0)
        if self.parallel:
            attn_out = self.attn(self.attn_norm(x_in))
            mlp_out = self.mlp(self.mlp_norm(x_in))
            return x_in + self.attn_scale.to(x_in.dtype) * attn_out + self.mlp_scale.to(x_in.dtype) * mlp_out
        x = x_in + self.attn_scale.to(x_in.dtype) * self.attn(self.attn_norm(x_in))
        x = x + self.mlp_scale.to(x.dtype) * self.mlp(self.mlp_norm(x))
        return x


def build_schedule(num_layers: int, recur_idxs: list[int]) -> tuple[list[int], list[int]]:
    n_enc = num_layers // 2
    enc = list(range(n_enc))
    dec: list[int] = []
    for i in range(n_enc, num_layers):
        dec.append(i)
        if i in recur_idxs:
            dec.append(i)  # adjacent repeat = one depth-recurrence step
    return enc, dec


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.model_dim)
        n_enc = cfg.num_layers // 2
        n_dec = cfg.num_layers - n_enc
        self.n_enc = n_enc
        self.num_skip = min(n_enc, n_dec)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip, cfg.model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList(
            [
                Block(
                    cfg.model_dim,
                    cfg.num_heads,
                    cfg.num_kv_heads,
                    cfg.mlp_mult,
                    cfg.rope_base,
                    cfg.qk_gain_init,
                    parallel=(cfg.parallel_residual and i >= cfg.pr_from_layer),
                )
                for i in range(cfg.num_layers)
            ]
        )
        self.final_norm = RMSNorm()
        self.lm_head = None if cfg.tie_embeddings else CastedLinear(cfg.model_dim, cfg.vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self.enc_schedule, self.dec_schedule = build_schedule(
            cfg.num_layers, cfg.recur_layer_idxs if cfg.depth_recurrence else []
        )
        self._init_weights()

    def _init_weights(self) -> None:
        if self.cfg.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, std=self.cfg.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def virtual_layers(self) -> int:
        return len(self.enc_schedule) + len(self.dec_schedule)

    def forward(self, input_ids: Tensor, targets: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []
        for i in self.enc_schedule:
            x = self.blocks[i](x, x0)
            skips.append(x)
        seen_dec: set[int] = set()
        for slot, i in enumerate(self.dec_schedule):
            if i not in seen_dec and skips:
                x = x + self.skip_weights[len(seen_dec)].to(x.dtype) * skips.pop()
                seen_dec.add(i)
            x = self.blocks[i](x, x0)
        x = self.final_norm(x).reshape(-1, x.size(-1))
        if self.cfg.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        logits = self.cfg.logit_softcap * torch.tanh(logits_proj / self.cfg.logit_softcap)
        return F.cross_entropy(logits.float(), targets.reshape(-1))
