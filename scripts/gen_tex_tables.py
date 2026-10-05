"""Generate LaTeX table fragments for the technical report from results.json."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"


def mean_std(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = (sum((x - m) ** 2 for x in xs) / max(n - 1, 1)) ** 0.5
    return m, sd


def arch_tex(s):
    seeds = s["seeds"]
    tab = s["arch"]["table"]
    lines = [r"\begin{tabular}{l c c c c}", r"\toprule",
             r"Config & Per-seed BPB & Mean & Std \\", r"\midrule"]
    for c, vals in tab.items():
        m, sd = mean_std(vals)
        per = ", ".join(f"{v:.4f}" for v in vals)
        lines.append(f"{c} & {per} & {m:.4f} & {sd:.4f} \\\\")
    lines += [r"\midrule"]
    for k, t in s["arch"]["tests_vs_B"].items():
        lines.append(f"\\multicolumn{{3}}{{l}}{{{k} vs B: $\\Delta={t['delta_bpb']:+.4f}$, "
                     f"$p={t['p_perm']:.4f}$}} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def quant_tex(s):
    tab = s["quant"]["table"]
    lines = [r"\begin{tabular}{l c c c c}", r"\toprule",
             r"Scheme & Per-seed BPB & Mean & Std \\", r"\midrule"]
    for q, vals in tab.items():
        m, sd = mean_std(vals)
        per = ", ".join(f"{v:.4f}" for v in vals)
        qlab = q.replace("_", r"\_")
        lines.append(f"{qlab} & {per} & {m:.4f} & {sd:.4f} \\\\")
    lines += [r"\midrule"]
    label_map = {
        "asdclip_vs_sdclip": ("A-SDClip vs SDClip", +1),
        "asdclip_vs_int8": ("A-SDClip vs int8", -1),
        "sdclip_vs_int8": ("SDClip vs int8", -1),
    }
    for k, t in s["quant"]["tests"].items():
        label, sgn = label_map[k]
        lines.append(f"\\multicolumn{{3}}{{l}}{{{label}: $\\Delta={sgn*t['delta_bpb']:+.4f}$, "
                     f"$n={t['n']}$, $p={t['p_perm']:.4f}$}} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def ttt_tex(s):
    t = s["ttt"]
    if not t:
        return ""
    lines = [r"\begin{tabular}{l c c c}", r"\toprule",
             r"Seed & Post-quant & TTT & $\Delta$ \\", r"\midrule"]
    for r in t["per_seed"]:
        lines.append(f"{r['seed']} & {r['post_quant']:.4f} & {r['ttt']:.4f} & "
                     f"{r['delta']:+.4f} \\\\")
    lines += [r"\midrule",
              f"\\multicolumn{{4}}{{l}}{{Mean $\\Delta={t['delta_bpb']:+.4f}$, "
              f"$p={t['p_perm']:.4f}$}} \\\\",
              r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main():
    s = json.loads((REPO / "results" / "results.json").read_text(encoding="utf-8"))
    (DOCS / "results_arch.tex").write_text(arch_tex(s), encoding="utf-8")
    (DOCS / "results_quant.tex").write_text(quant_tex(s), encoding="utf-8")
    (DOCS / "results_ttt.tex").write_text(ttt_tex(s) or "% no ttt results", encoding="utf-8")
    print("tex fragments written")


if __name__ == "__main__":
    main()
