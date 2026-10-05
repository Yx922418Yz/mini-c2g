# Docs index — C2G Parameter Golf (Li Yaxuan)

| 交付物 | 文件 | 状态 |
|---|---|---|
| L1 方案草案（≥500 字，4 个门槛问题） | [LiYaxuan_C2G_方案草案.md](LiYaxuan_C2G_方案草案.md) | ✅ |
| 方案设计（目标级别、选型、实验矩阵） | [LiYaxuan_C2G_方案设计.md](LiYaxuan_C2G_方案设计.md) | ✅ |
| 消融（组件单独 + 组合贡献） | LiYaxuan_C2G_ablation.md | 实验后完成 |
| 榜单（方案 vs baseline vs SOTA） | LiYaxuan_C2G_leaderboard.md | 实验后完成 |
| AI 协作日志（实时记录） | [LiYaxuan_C2G_AI日志.md](LiYaxuan_C2G_AI日志.md) | 持续追加 |
| 拿来说明（拿了什么、改了什么、为什么） | [LiYaxuan_C2G_拿来说明.md](LiYaxuan_C2G_拿来说明.md) | ✅ |
| 英文技术报告（≥6 页） | LiYaxuan_C2G_tech_report.pdf | 实验后完成 |

训练脚本与 artifact：
- `train_gpt.py`（仓库根；最终训练脚本）
- `artifacts/*_submission.tar.gz`（16 MB 内 artifact）
- `artifacts/*_submission.json`（BPB、训练时间、硬件、seed）
- `logs/`（每 run 的结构化 JSONL 日志）

官方 PR 风格记录快照：`records/LiYaxuan_C2G/`。
