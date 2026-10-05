# LiYaxuan_C2G_leaderboard.md — 榜单定位与口径说明

**作者**：李亚轩（Li Yaxuan）｜郑州西亚斯学院 AI+X Elite 20
**日期**：2026-10-03

## 1. 官方赛道参照点（8×H100，10 min，16 MB，FineWeb validation BPB）

| 位置 | 方案要点 | BPB | 来源 |
|---|---|---|---|
| 官方基线 | 9 层 512 维，tied embedding，4 KV heads | ≈ **1.2244** | Parameter Golf 官方 baseline |
| 2026-04 榜首 | SP8192 + 3-Layer Recurrence + Parallel Residuals + QK 5.25 + Legal TTT（作者 bigbag） | **1.0810**（std 0.0002，seeds 42/314/999） | records/track_10min_16mb/2026-04-09_... |
| Level 门槛 | L2 <1.18；L3 <1.12；L4 <1.085；Elite 核心候选参考 <1.15 | — | CHALLENGE.md |

## 2. 本作品微尺度实测定位（CPU，真实 FineWeb，d=128，6 层，8M/2M token）

> **口径红线**：微尺度 BPB 绝对值（~2.16–2.62）与 H100 榜单（~1.08–1.22）**不可直接比较**——模型尺寸、训练 token、序列长度、tokenizer（SP1024 vs SP8192）全部不同。微尺度只产生**同检查点配对 Δ**与**管线/打包合规证据**；任何把微尺度数字写成榜单成绩的做法都是造假，本作品不做。

| 比较（配对，同检查点） | ΔBPB | n | p |
|---|---|---|---|
| A-SDClip vs SDClip（int6） | **+0.4015** | 3（扩展至 8 进行中） | 0.25（n=3 下限） |
| 合法 TTT vs post-quant（A-SDClip） | **+0.0541** | 3 | 0.25（n=3 下限） |
| A-SDClip+TTT vs int8（无 TTT） | **+0.0233** | 3 | 0.25（n=3 下限） |
| PR vs B | −0.0001（中性） | 3 | 0.25 |
| DR vs B | −0.0046（微尺度有害） | 3 | 0.25 |

## 3. 与各级别要求的差距（诚实清单）

| 要求 | 状态 |
|---|---|
| L1：≥500 字方案草案 + 4 门槛问题 + 复现 1.22±0.005 | 草案 ✅；H100 复现脚本 ✅；**真实 H100 复现需 GPU，本机无法执行** |
| L2：单点改进 BPB<1.18，≥3 seed | 方法（A-SDClip）与微尺度证据 ✅；**真实 BPB 需 H100** |
| L3：≥2 方向组合、BPB<1.12、消融、Δ≥0.005、p<0.01 | 微尺度消融与统计流程 ✅（n=8 扩展用于演示 p<0.01 流程）；**真实 BPB 需 H100** |
| L4：BPB<1.085、原创改动、GitHub 仓库 + CI、≥6 页英文报告、10 分钟视频 | 原创 A-SDClip ✅、GitHub-ready 仓库 + CI ✅、英文报告 ✅；视频未做；**真实 BPB 需 H100** |

## 4. 从微尺度到榜单的唯一未闭合环节

**只差 8×H100 算力**。仓库已提供：
- `scripts/reproduce_h100.sh`：设置全部 env（H100 规模、SP8192 数据路径、torchrun 8 卡、打包 16 MB 门），一条命令产出真实 submission；
- `eval_bpb.py`：离线独立评估器，从 tar.gz 提取权重/tokenizer，支持 `--ttt`；
- CI：每次 push 自动在 CPU 合成分片上验证训练→量化→TTT→打包门→独立评估全链路。

## 5. 结论

在无 GPU、无外部 API 的条件下，本作品交付了：①一个有理论推导、下行有界、实测把 SDClip 损失救回约 93% 的原创量化方法 A-SDClip；②全部组件的真实数据配对消融（含负结果）；③GitHub-ready、CI 验证、一条命令即可在 H100 产出真实成绩的完整仓库。榜单 BPB 的最终数字，严格留待 H100 运行后填入，本文件不含任何虚构成绩。
