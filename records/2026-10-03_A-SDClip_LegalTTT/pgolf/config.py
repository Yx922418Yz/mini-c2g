"""Configuration for Parameter Golf runs.

All hyperparameters are overridable through environment variables so that the
H100 leaderboard path (``torchrun train_gpt.py``) and the local CPU smoke path
(``python train_gpt.py``) share one code base.
"""

from __future__ import annotations

import os
import uuid


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    return bool(_env_int(name, int(default)))


class Config:
    # Run identity
    run_id = os.environ.get("RUN_ID", f"smoke-{uuid.uuid4().hex[:8]}")
    seed = _env_int("SEED", 1337)

    # Data layout (official shard format: 1024-byte int32 header + uint16 tokens)
    data_path = os.environ.get("DATA_PATH", "./data/micro")
    train_glob = os.environ.get("TRAIN_GLOB", "fineweb_train_*.bin")
    val_glob = os.environ.get("VAL_GLOB", "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")

    # Validation / logging cadence
    val_batch_tokens = _env_int("VAL_BATCH_TOKENS", 65_536)
    val_limit_tokens = _env_int("VAL_LIMIT_TOKENS", 0)  # 0 = full split
    val_loss_every = _env_int("VAL_LOSS_EVERY", 0)       # 0 = final only
    train_log_every = _env_int("TRAIN_LOG_EVERY", 50)

    # Training length
    iterations = _env_int("ITERATIONS", 400)
    warmdown_iters = _env_int("WARMDOWN_ITERS", 120)
    warmup_steps = _env_int("WARMUP_STEPS", 0)
    batch_tokens = _env_int("BATCH_TOKENS", 16_384)
    seq_len = _env_int("SEQ_LEN", 256)
    max_wallclock_seconds = _env_float("MAX_WALLCLOCK_SECONDS", 0.0)  # 0 = no cap

    # Model shape (micro defaults; H100 overrides via env)
    vocab_size = _env_int("VOCAB_SIZE", 1024)
    num_layers = _env_int("NUM_LAYERS", 6)
    model_dim = _env_int("MODEL_DIM", 128)
    num_heads = _env_int("NUM_HEADS", 4)
    num_kv_heads = _env_int("NUM_KV_HEADS", 2)
    mlp_mult = _env_int("MLP_MULT", 2)
    tie_embeddings = _env_bool("TIE_EMBEDDINGS", True)
    rope_base = _env_float("ROPE_BASE", 10_000.0)
    logit_softcap = _env_float("LOGIT_SOFTCAP", 30.0)
    tied_embed_init_std = _env_float("TIED_EMBED_INIT_STD", 0.02)
    qk_gain_init = _env_float("QK_GAIN_INIT", 1.5)

    # Architecture add-ons (the ablation switches)
    parallel_residual = _env_bool("PARALLEL_RESIDUAL", False)
    pr_from_layer = _env_int("PR_FROM_LAYER", 4)       # layers >= this use parallel lanes
    depth_recurrence = _env_bool("DEPTH_RECURRENCE", False)
    recur_layers = os.environ.get("RECUR_LAYERS", "3,4")  # physical layer indices looped once
    recur_activate_frac = _env_float("RECUR_ACTIVATE_FRAC", 0.0)  # 0 = from the start (micro)

    # Optimizer
    matrix_lr = _env_float("MATRIX_LR", 0.03)
    scalar_lr = _env_float("SCALAR_LR", 0.03)
    tied_embed_lr = _env_float("TIED_EMBED_LR", 0.05)
    embed_lr = _env_float("EMBED_LR", 0.6)
    head_lr = _env_float("HEAD_LR", 0.008)
    use_muon = _env_bool("USE_MUON", True)
    muon_momentum = _env_float("MUON_MOMENTUM", 0.95)
    muon_backend_steps = _env_int("MUON_BACKEND_STEPS", 5)
    beta1 = _env_float("BETA1", 0.9)
    beta2 = _env_float("BETA2", 0.95)
    adam_eps = _env_float("ADAM_EPS", 1e-8)
    weight_decay = _env_float("WEIGHT_DECAY", 0.0)
    grad_clip_norm = _env_float("GRAD_CLIP_NORM", 0.0)

    # EMA (mean-teacher) weights
    ema_decay = _env_float("EMA_DECAY", 0.0)  # 0 = disabled

    # Quantization / packing: "int8" | "int6_sdclip" | "int6_asdclip"
    quant_scheme = os.environ.get("QUANT_SCHEME", "int8")
    sdclip_k_int6 = _env_float("SDCLIP_K_INT6", 12.85)
    sdclip_k_int8 = _env_float("SDCLIP_K_INT8", 20.0)
    asdclip_alpha = _env_float("ASDCLIP_ALPHA", 0.10)
    asdclip_k_min = _env_float("ASDCLIP_K_MIN", 8.0)
    asdclip_k_max = _env_float("ASDCLIP_K_MAX", 20.0)
    compression_level = _env_int("COMPRESSION_LEVEL", 9)

    # Legal score-first TTT at evaluation time
    ttt_enabled = _env_bool("TTT_ENABLED", False)
    ttt_chunk_tokens = _env_int("TTT_CHUNK_TOKENS", 8_192)
    ttt_epochs = _env_int("TTT_EPOCHS", 3)
    ttt_lr = _env_float("TTT_LR", 0.005)
    ttt_momentum = _env_float("TTT_MOMENTUM", 0.9)
    ttt_grad_clip = _env_float("TTT_GRAD_CLIP", 1.0)

    # Artifact
    artifact_bytes_cap = _env_int("ARTIFACT_BYTES_CAP", 16_000_000)
    out_dir = os.environ.get("OUT_DIR", ".")

    @property
    def recur_layer_idxs(self) -> list[int]:
        return [int(x) for x in self.recur_layers.split(",") if x.strip()]
