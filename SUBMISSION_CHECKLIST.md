# Submission checklist (C2G Parameter Golf — Li Yaxuan)

## Hard constraints
- [ ] Training wallclock ≤ 600 s (8×H100 path; micro runs report their own time)
- [ ] Evaluation wallclock ≤ 600 s
- [ ] Artifact ≤ 16,000,000 decimal bytes (enforced by `pack_submission.py`)
- [ ] No network access / external calls during evaluation
- [ ] Metric: BPB on FineWeb validation, lower is better

## Required deliverables (hard naming)
- [ ] `LiYaxuan_C2G_方案草案.md` (≥500 chars, 4 gate questions) — docs/
- [ ] `LiYaxuan_C2G_方案设计.md` — docs/
- [ ] `LiYaxuan_C2G_train_gpt.py` — repo root (record snapshot in records/)
- [ ] `LiYaxuan_C2G_submission.tar.gz` — artifacts/
- [ ] `LiYaxuan_C2G_submission.json` — artifacts/
- [ ] `LiYaxuan_C2G_logs/` — logs/ (≥3 independent seeds, JSONL)
- [ ] `LiYaxuan_C2G_ablation.md` — docs/
- [ ] `LiYaxuan_C2G_leaderboard.md` — docs/
- [ ] `LiYaxuan_C2G_AI日志.md` — docs/
- [ ] `LiYaxuan_C2G_拿来说明.md` — docs/
- [ ] `LiYaxuan_C2G_tech_report.pdf` (≥6 pages English) — docs/

## GitHub-ready
- [ ] README.md, LICENSE, THIRD_PARTY_NOTICES.md, .gitignore
- [ ] requirements.txt / requirements-cpu.txt
- [ ] One-command reproduction: scripts/reproduce_h100.sh, scripts/smoke_cpu.ps1
- [ ] CI green (.github/workflows/ci.yml)
- [ ] No `__pycache__`, no stray files, deterministic archives
- [ ] Record folder records/LiYaxuan_C2G/ snapshot

## Upload to GitHub
1. Create an empty GitHub repository (no auto README).
2. From the repo folder:
   ```
   git init
   git add .
   git commit -m "C2G Parameter Golf submission: A-SDClip"
   git branch -M main
   git remote add origin https://github.com/<user>/<repo>.git
   git push -u origin main
   ```
3. Confirm CI passes on the push.
