"""Run the ablation matrix locally (CPU).

Each cell is a subprocess invocation of train_gpt.py with a fixed env block.
Training is deterministic for a given (config, seed), so quantization-scheme
variants of the same cell share an identical checkpoint and give paired
post-quant comparisons.

Usage:
    python scripts/run_experiments.py --seeds 1337 2024 4242
    python scripts/run_experiments.py --quick          # 1 seed, tiny matrix
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Common micro knobs (override defaults here; H100 path uses scripts/reproduce_h100.sh)
COMMON = {
    "ITERATIONS": "400",
    "WARMDOWN_ITERS": "120",
    "BATCH_TOKENS": "16384",
    "SEQ_LEN": "256",
    "MODEL_DIM": "128",
    "NUM_LAYERS": "6",
    "NUM_HEADS": "4",
    "NUM_KV_HEADS": "2",
    "VAL_LIMIT_TOKENS": "0",        # full 2M-token validation split
    "TTT_CHUNK_TOKENS": "8192",
    "TRAIN_LOG_EVERY": "100",
}

# (config name, arch env, list of quant schemes to pack)
CELLS = [
    ("B", {}, ["int8"]),
    ("PR", {"PARALLEL_RESIDUAL": "1"}, ["int8"]),
    ("DR", {"DEPTH_RECURRENCE": "1"}, ["int8"]),
    ("Full", {"PARALLEL_RESIDUAL": "1", "DEPTH_RECURRENCE": "1"},
     ["int8", "int6_sdclip", "int6_asdclip"]),
]


def run_cell(config: str, arch: dict, scheme: str, seed: int, ttt: bool,
             prefix: str = "") -> int:
    run_id = f"{prefix}{config}_{scheme}_s{seed}"
    env = os.environ.copy()
    env.update(COMMON)
    env.update(arch)
    env["QUANT_SCHEME"] = scheme
    env["SEED"] = str(seed)
    env["RUN_ID"] = run_id
    env["TTT_ENABLED"] = "1" if ttt else "0"
    log_file = REPO / "logs" / f"{run_id}.stdout.log"
    print(f"=== {run_id} ===", flush=True)
    t0 = time.perf_counter()
    with open(log_file, "w", encoding="utf-8") as f:
        proc = subprocess.run(
            [sys.executable, "train_gpt.py"], cwd=REPO, env=env,
            stdout=f, stderr=subprocess.STDOUT,
        )
    print(f"=== {run_id} exit={proc.returncode} {time.perf_counter()-t0:.0f}s ===",
          flush=True)
    return proc.returncode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[1337, 2024, 4242])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--configs", nargs="+", default=None)
    ap.add_argument("--prefix", default="", help="run-id prefix (e.g. gpu_)")
    args = ap.parse_args()

    if args.quick:
        COMMON.update({"ITERATIONS": "60", "WARMDOWN_ITERS": "20",
                       "VAL_LIMIT_TOKENS": "60000"})
        args.seeds = args.seeds[:1]

    cells = CELLS
    if args.configs:
        wanted = set(args.configs)
        cells = []
        for cfg_name, arch, schemes in CELLS:
            keep = [s for s in schemes
                    if cfg_name in wanted or f"{cfg_name}/{s}" in wanted]
            if keep:
                cells.append((cfg_name, arch, keep))

    failures = []
    for seed in args.seeds:
        for config, arch, schemes in cells:
            for scheme in schemes:
                ttt = config == "Full" and scheme == "int6_asdclip"
                rc = run_cell(config, arch, scheme, seed, ttt, args.prefix)
                if rc != 0:
                    failures.append(f"{config}/{scheme}/s{seed}")
    if failures:
        print("FAILED CELLS:", failures)
        sys.exit(1)
    print("matrix complete")


if __name__ == "__main__":
    main()
