# LiYaxuan_C2G_拿来说明

李亚轩（Li Yaxuan），郑州西亚斯学院，AI+X Elite 20 Program
说明原则：逐项列出"拿了什么、从哪里拿、我改了什么、为什么这样改"，不把他人工作包装成原创，也不把原创点淹没在借用代码里。

## 1. 总体说明

本仓库以 OpenAI 官方开源的 parameter-golf 基线（MIT License，Copyright (c) 2026 OpenAI，https://github.com/openai/parameter-golf）为起点，基线本身是 Karpathy nanoGPT → Keller Jordan modded-nanoGPT 家族的后代。我没有逐字复制基线文件，而是在通读基线 `train_gpt.py`（1126 行）与全部历史记录 README 的基础上，**按模块重新组织并实现了等价基线**，再叠加自己的改动。所有借用关系在 `THIRD_PARTY_NOTICES.md` 与本文件中声明。

## 2. 逐项对照

| 模块 / 组件 | 来源 | 我做的改动与理由 |
|---|---|---|
| 分片格式（1024 B int32 头 + uint16 token；magic 20240520） | 官方基线 `data/` 管线 | 原样保留格式；`load_data_shard` 增加对"字节范围前缀文件"的放宽校验，因为本地验证只下载了分片头部，声明 token 数与截断体积不一致是预期行为 |
| TokenStream / 确定性连续取数 | 官方基线 | 逻辑等价；拆分为单进程 `LocalTokenLoader`（CPU）与 `DistributedTokenLoader`（H100），替代基线中按 rank 条件分支的单实现，便于本机测试 |
| Muon（Newton-Schulz5 正交化） | Keller Jordan Muon / modded-nanoGPT（官方基线内置） | 保留算法；`zeropower_via_newtonschulz5` 在 CPU 上不转 bf16（CPU bf16 矩阵乘慢且无收益），分布式 all-reduce 路径仅在进程组初始化时启用 |
| 模型主体：tied embedding、RMSNorm、RoPE、QK RMSNorm + q_gain、GQA、relu² MLP、可学残差尺度、U-Net skip、tanh softcap | 官方基线（modded-nanoGPT 家族） | 结构等价、重新实现；GQA 在 CPU 上用 `repeat_interleave` 展开 KV（`enable_gqa` 的 CPU 支持在不同版本间不一致），CUDA 路径仍用原生 `enable_gqa` |
| int8 RTN 量化结构（按行量化 + fp16 scale + 小张量保留 + zlib 压缩） | 官方基线 | 保留；量化判定从"numel≤65,536 保留浮点"改为"非 2D 或极小才保留"，因为微尺度模型矩阵全部小于该阈值，原条件会让量化实验完全失效；H100 尺度下两种判定等价 |
| **SDClip（clip=k·std(row)，int6 k=12.85 / 嵌入 int8 k=20）** | Kevin Clark 2026-04-05 parameter-golf 记录 | 作为量化链 Q1 原样实现，并作为我的 A-SDClip 的基准与退化兜底 |
| **GPTQ-lite（Hessian 引导列序，blocksize 128，damp 0.01）** | Frantar et al., arXiv:2210.17323；官方记录的 GPTQ 实现 | 用 ~40 行实现核心算法（Cholesky 求逆、逐块误差回写），配合 SDClip/A-SDClip 的行尺度；校准激活由训练数据前向 hook 采集 |
| 并行残差 | GPT-J（Wang & Komatsuzaki, 2021） | 在 `Block` 增加 `parallel` 开关：并行时 attn/mlp 同读同一归一化输出并求和；通过 `PR_FROM_LAYER` 控制启用层 |
| 深度递归 | Universal Transformer（arXiv:1807.03819）；榜首记录的 3 层递归 | 用 `build_schedule` 生成虚拟层调度（指定物理层相邻重复一次）；U-Net skip 的 pop 用 `seen_dec` 保证重复层只取一次 skip |
| 合法 score-first TTT | 2026-04-09 榜首记录；TTT 文献 arXiv:2411.07279 | 严格按"先整 chunk 评分、后 SGD 适配"实现，lr 跨 chunk 余弦衰减；合规四条在方案设计 §4 逐条落实 |
| BPB 评估（SentencePiece LUT、前导空格约定） | 官方基线 | 逻辑等价；增加单进程 CPU 路径 |
| 打包（tar.gz + 16 MB 门） | 官方基线 | 改为确定性归档（固定 mtime、排序成员）；**计数口径更严格**：把 `pgolf/` 全部源码与四个根脚本一并计入 artifact，而不是只计 train_gpt.py |

## 3. 原创点声明

**A-SDClip（Adaptive SDClip）是本仓库的原创贡献**：将 SDClip 的全局单一 k 改为按每行超额峰度（偏差校正 + 经验贝叶斯收缩后）逐行自适应，`k_i=clip(k_base·exp(α·κ_s),k_min,k_max)`。据我检索 parameter-golf 全部历史记录（截至资料包打包时）与公开文献，没有相同的"按行峰度驱动裁剪倍数"方案。其动机、推导、下行有界性与实测结果见 `LiYaxuan_C2G_方案设计.md` §3 与消融/榜单文档。我不主张 GPTQ、SDClip、并行残差、深度递归、TTT 中的任何一项为原创；这些是"拿来"的组件，原创性仅体现在自适应裁剪规则及其与量化链的组合方式。

## 4. 工程口径的两处主动选择

1. **模块化而非单文件**：官方 FAQ 称"计数字节应都在 train_gpt.py"，但 PR 规则明确允许"other dependencies"。我选择模块化以获得可读性与可测试性，并把全部自有代码字节计入 artifact、不利用模块化隐藏任何字节——比最低要求更严格。
2. **确定性打包**：固定 mtime 与成员顺序，使结果可字节复现，方便评审与 CI 比对。

## 5. 数据来源

- 全部实验文本来自官方托管的 FineWeb（arXiv:2406.17557）分片，经 hf-mirror 下载字节范围前缀（800 万训练 / 200 万验证真实 token）与官方 SP1024 SentencePiece tokenizer（arXiv:1808.06226）。
- CI 中的合成数据由 `make_synthetic_shards.py` 离线生成，仅用于通路测试，不进入任何结果数字。
- 未使用任何未授权数据，未在评估中联网。
