"""Aggregate run logs into tables and paired statistical tests.

Reads artifacts/<run>_submission.json, groups by config/scheme/seed, and runs
two-sided paired permutation tests (exact when 2^n <= 2^18, otherwise Monte
Carlo with 200,000 reshuffles).

Outputs: results/results.json, results/tables.md
"""

from __future__ import annotations

import json
import random
from itertools import product
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "artifacts"
RES = REPO / "results"


def load_runs(prefix: str = "") -> dict:
    runs = {}
    for p in ART.glob("*.json"):
        meta = json.loads(p.read_text(encoding="utf-8"))
        rid = meta["run_id"]
        if not rid.startswith(prefix):
            continue
        if prefix:
            meta = dict(meta)
            meta["run_id"] = rid[len(prefix):]
        runs[meta["run_id"]] = meta
    return runs


def paired_perm_p(diffs: list[float], n_mc: int = 200_000, seed: int = 7) -> float:
    diffs = [d for d in diffs if d != 0]
    n = len(diffs)
    if n == 0:
        return 1.0
    observed = abs(sum(diffs))
    if 2 ** n <= 2 ** 18:
        ge = 0
        for signs in product((-1, 1), repeat=n):
            s = abs(sum(sg * d for sg, d in zip(signs, diffs)))
            if s >= observed - 1e-12:
                ge += 1
        return ge / (2 ** n)
    rng = random.Random(seed)
    ge = 0
    for _ in range(n_mc):
        s = abs(sum(d * rng.choice((-1.0, 1.0)) for d in diffs))
        if s >= observed - 1e-12:
            ge += 1
    return (ge + 1) / (n_mc + 1)


def mean_std(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / max(n - 1, 1)
    return m, var ** 0.5


def summarize(runs: dict) -> dict:
    # index: (config, scheme, seed)
    idx = {}
    for rid, m in runs.items():
        # names: B_int8_s1337 | PR_int8_s1337 | Full_int6_asdclip_s1337
        # special runs (final_a010, sweep_*) are excluded from the paired matrix
        scheme = m["quant_scheme"]
        marker = "_" + scheme + "_"
        if marker not in rid:
            continue
        seed = int(m["seed"])
        config = rid.split(marker)[0].rstrip("_")
        idx[(config, scheme, seed)] = m

    seeds = sorted({k[2] for k in idx})
    out = {"seeds": seeds, "arch": {}, "quant": {}, "ttt": {}}

    # Architecture: pre-quant BPB (int8 cells)
    arch_configs = ["B", "PR", "DR", "Full"]
    arch_table = {}
    for c in arch_configs:
        vals = [idx[(c, "int8", s)]["bpb_pre_quant"] for s in seeds
                if (c, "int8", s) in idx]
        if vals:
            arch_table[c] = vals
    out["arch"]["table"] = arch_table
    base = arch_table.get("B", [])
    arch_tests = {}
    for c in arch_configs[1:]:
        if c in arch_table and len(arch_table[c]) == len(base):
            d = [base[i] - arch_table[c][i] for i in range(len(base))]
            arch_tests[c] = {
                "delta_bpb": mean_std(d)[0],
                "p_perm": paired_perm_p(d),
            }
    out["arch"]["tests_vs_B"] = arch_tests

    # Quantization: post-quant BPB on Full checkpoints, paired per seed
    q_schemes = ["int8", "int6_sdclip", "int6_asdclip"]
    q_by_seed: dict[str, dict[int, float]] = {q: {} for q in q_schemes}
    for (config, q, s), m in idx.items():
        if config == "Full" and q in q_by_seed:
            q_by_seed[q][s] = m["bpb_post_quant"]
    q_table = {q: [q_by_seed[q][s] for s in sorted(q_by_seed[q])]
               for q in q_schemes if q_by_seed[q]}
    out["quant"]["table"] = q_table

    def paired_test(a: str, b: str) -> dict | None:
        common = sorted(set(q_by_seed[a]) & set(q_by_seed[b]))
        if not common:
            return None
        d = [q_by_seed[a][s] - q_by_seed[b][s] for s in common]
        return {"delta_bpb": mean_std(d)[0], "p_perm": paired_perm_p(d),
                "n": len(common), "seeds": common}

    q_tests = {}
    for a, b, name in [
        ("int6_asdclip", "int6_sdclip", "asdclip_vs_sdclip"),
        ("int6_asdclip", "int8", "asdclip_vs_int8"),
        ("int6_sdclip", "int8", "sdclip_vs_int8"),
    ]:
        t = paired_test(a, b)
        if t:
            q_tests[name] = t
    out["quant"]["tests"] = q_tests

    # TTT
    ttt = []
    for s in sorted({s for (c, q, s) in idx if c == "Full" and q == "int6_asdclip"}):
        key = ("Full", "int6_asdclip", s)
        if key in idx and idx[key].get("bpb_ttt") is not None:
            ttt.append({
                "seed": s,
                "post_quant": idx[key]["bpb_post_quant"],
                "ttt": idx[key]["bpb_ttt"],
                "delta": idx[key]["bpb_post_quant"] - idx[key]["bpb_ttt"],
            })
    if ttt:
        d = [t["delta"] for t in ttt]
        out["ttt"] = {"per_seed": ttt, "delta_bpb": mean_std(d)[0],
                      "p_perm": paired_perm_p(d)}
    return out


def render_md(summary: dict) -> str:
    lines = ["# Experiment tables (micro FineWeb, CPU)", ""]
    lines.append(f"Seeds: {summary['seeds']}")
    lines.append("")
    lines.append("## Architecture (pre-quant BPB; lower is better)")
    lines.append("| config | per-seed BPB | mean | std |")
    lines.append("|---|---|---|---|")
    for c, vals in summary["arch"]["table"].items():
        m, sd = mean_std(vals)
        lines.append(f"| {c} | {', '.join(f'{v:.5f}' for v in vals)} | {m:.5f} | {sd:.5f} |")
    lines.append("")
    lines.append("| comparison | mean delta (BPB) | paired perm p |")
    lines.append("|---|---|---|")
    for k, t in summary["arch"]["tests_vs_B"].items():
        lines.append(f"| {k} - B | {t['delta_bpb']:+.5f} | {t['p_perm']:.4f} |")
    lines.append("")
    lines.append("## Quantization (post-quant BPB on Full checkpoints)")
    lines.append("| scheme | per-seed BPB | mean | std |")
    lines.append("|---|---|---|---|")
    for q, vals in summary["quant"]["table"].items():
        m, sd = mean_std(vals)
        lines.append(f"| {q} | {', '.join(f'{v:.5f}' for v in vals)} | {m:.5f} | {sd:.5f} |")
    lines.append("")
    lines.append("| comparison | mean delta (BPB) | n | paired perm p |")
    lines.append("|---|---|---|---|")
    for k, t in summary["quant"]["tests"].items():
        lines.append(f"| {k} | {t['delta_bpb']:+.5f} | n={t['n']} | {t['p_perm']:.4f} |")
    lines.append("")
    if summary["ttt"]:
        t = summary["ttt"]
        lines.append("## Legal score-first TTT")
        lines.append("| seed | post-quant | TTT | delta |")
        lines.append("|---|---|---|---|")
        for r in t["per_seed"]:
            lines.append(f"| {r['seed']} | {r['post_quant']:.5f} | {r['ttt']:.5f} | {r['delta']:+.5f} |")
        lines.append(f"\nMean delta {t['delta_bpb']:+.5f}, p={t['p_perm']:.4f}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="")
    args = ap.parse_args()
    RES.mkdir(exist_ok=True)
    runs = load_runs(args.prefix)
    summary = summarize(runs)
    tag = args.prefix.rstrip("_")
    stem = f"{tag}_" if tag else ""
    (RES / f"{stem}results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    md = render_md(summary).replace(
        "(micro FineWeb, CPU)",
        f"(micro FineWeb, {tag or 'CPU'})")
    (RES / f"{stem}tables.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
