# LiYaxuan_C2G_AI 协作日志

李亚轩（Li Yaxuan），郑州西亚斯学院，AI+X Elite 20 Program
记录方式：按时间顺序记录我与 AI（Doubao）的协作过程，包括任务拆解、关键提问、AI 产出、我的判断与修正、遇到的 bug、实验结果与迭代。本日志持续追加，不做事后美化。

## 2026-10-02（承接 C2 / C2A）

- 完成 C2「AI for Math 论文」，AI 初评 89/100；完成 C2A「C9 认知能力」MetaCal-Shift（Track 2 Metacognition），AI 初评 94/100。
- 点评中暴露的共性问题：①外部证据停留在"可执行路径"、缺真实实测；②开发中出现多次低级 bug；③部分数字为文献预期而非实测。**C2G 的策略据此调整：必须在本机用真实数据跑通整条管线并产出真实数字，边界如实声明。**

## 2026-10-03

### 阶段 0：任务拆解与资料精读
- 我给 AI 的指令：完成 C2G，按 GitHub 仓库标准做，最终放桌面"提交挑战"文件夹，追求高分。
- AI 精读 `CHALLENGE.md`、`rubric.json`，提炼硬约束（≤600 s、≤16,000,000 字节、BPB、断网）与四级任务、11 项硬命名交付物；我确认了"两套评分口径都要照顾，BPB 头号"。
- AI 通读官方基线 `train_gpt.py`（1126 行）与榜首、SDClip 两份关键记录 README。

### 阶段 1：环境侦察
- 发现 PATH 上 python 为 3.14.7（无 torch），另有 D:\HJY Python 3.11.6（numpy、tqdm 齐全），选定 D:\HJY 为运行环境。
- 网络实测：pypi、download.pytorch.org、hf-mirror 可达；huggingface.co 直连超时。本机无 GPU、无 API key——**确认真实 8×H100 打榜无法在本机完成，确定"CPU + 真实 FineWeb 微尺度实测 + H100 一键复现 + 边界声明"的路径**，这是我与 AI 共同确认的关键决策。

### 阶段 2：真实数据获取
- AI 通过 HF API 列树，纠正路径怪癖（manifest/tokenizer 在 `datasets/`，分片在 `datasets/datasets/` 双层），经 hf-mirror + Range 请求下载：manifest、官方 SP1024 tokenizer（.model 254,483 B）、train_head（800 万真实 token）、val_head（200 万真实 token），header 核验通过（magic 20240520）。

### 阶段 3：方案与原创点
- AI 提出 SDClip 的自适应改造，我要求给出严格推导与有界下行：最终定为 **A-SDClip**——逐行超额峰度（偏差校正 + λ=m/(m+24) 经验贝叶斯收缩）驱动 k_i，最坏情况退化回 SDClip。
- 我要求实验必须配对：同一检查点三条量化链（int8 RTN / int6 SDClip / int6 A-SDClip），分离"int6 效应"与"自适应效应"；统计用配对置换检验，失败结果也要报告。

### 阶段 4：仓库开发
- 建立 GitHub-ready 骨架，逐模块写 config / muon / data / bpb / model / quant / ttt / artifacts 与四个根脚本、实验/分析脚本、CI、下载脚本、文档。
- **静态代码审查中我让 AI 自查并修复的问题（吸取前序挑战低级 bug 教训）：**
  1. train_gpt.py 残留一段无意义的 zip 循环 → 删除；
  2. analyze.py 解析 run_id 得到的 config 多了尾随下划线（"Full_"）→ rstrip 修复；
  3. **关键逻辑问题**：量化"小张量保留"阈值 65,536 会让微尺度全部矩阵保留 fp16、量化实验失效 → 改为按维度判定；
  4. gptq_quantize_matrix 残留 `to(int8 if False else float32)` 的垃圾代码 → 清理；
  5. tokenizer 在下载脚本中的子路径猜错（多了 fineweb1024/ 子目录），经 API 核实后修正为 `tokenizers/fineweb_1024_bpe.model`。

### 阶段 5：环境安装受阻与换源
- torch CPU 安装第一次因两个 pip 进程文件锁冲突失败（WinError 32）；第二次官方源下载极慢且反复超时。
- 尝试阿里云 pytorch-wheels 镜像（镜像内容陈旧、无对应 wheel）、上海交大镜像（SSL 错误）。
- 最终改用 curl 断点续传 + 30 次重试直接拉官方 CPU wheel（后台进行），避免 pip 超时即丢弃进度的问题。

### 阶段 6：单步冒烟与矩阵实验
- （待续：wheel 安装完成后，先合成数据单步验证，再真实数据小规模计时，然后跑 5 seeds 矩阵，所有结果与 bug 如实续记。）

## 协作模式小结（阶段性）

- 我的角色：定目标与边界、审查关键推导与实验设计、要求真实实测与诚实报告、逐一把关注项（GitHub-ready、配对检验、字节口径）变成硬约束。
- AI 的角色：资料精读、代码实现、文献核验、数据下载、静态自查、统计实现。
- 关键提问方式：不接受"都试试"式回答，要求单一方向 + 证据 + 预算 + 失败预案；不接受未验证数字，每个数必须可复现。

## 2026-10-03 18:10 ｜ GPTQ Hessian 非正定：死 ReLU 通道导致（真实矩阵暴露）

- **现象**：Full 架构 int6_sdclip / int6_asdclip cell 在 GPTQ 阶段报 `linalg.cholesky: not positive-definite (leading minor of order 48)`；20 步冒烟不出现（模型未训练）。
- **第一次误判与修复**：以为是 float32 累加精度问题，把 Hessian 改为 float64 计算——未解决，说明不是纯数值精度问题。
- **诊断方法**：写 `scripts/diag_calib.py`，训练完整 Full 模型后逐层检查校准激活的有限性与 H 特征值：36 层中仅 `blocks.5.mlp.proj.weight` 一个问题层——最后一层 MLP 投影输入（relu² 输出）大量通道为死 ReLU，H 高度稀疏奇异，最小特征值约 -4e-12，1% 平均对角阻尼不足以覆盖。
- **根因**：死通道（零方差列）使 H 的对应行列全零，Cholesky 在第 48 个主元处失败。
- **最终修复**（对齐官方 GPTQ 标准做法）：(1) 先做死通道判定 `diag(H)<=1e-12`，死通道主元钉为 1、对应权重列清零；(2) 阻尼改为自适应重试，Cholesky 失败时阻尼 ×4 最多 8 次。
- **教训**：①冒烟步数太少覆盖不了训练后分布，关键路径冒烟必须包含"训练完成态"；②同一个报错要区分"精度问题"与"结构奇异问题"，第一次修复属于没找到根因；③诊断脚本逐层打印特征值是最直接的定位手段。

## 2026-10-03 22:00 ｜ α 敏感性扫描与默认值迭代（0.15 → 0.10）

- **背景**：8 seed 主矩阵使用**预先注册**的 α=0.15，得到 A-SDClip vs SDClip +0.348 BPB、p=0.0078；TTT +0.052、p=0.0078。
- **扫描设计**（s1337，无 TTT，post-quant）：α ∈ {0.10, 0.15, 0.25, 0.40}，另测 k_max=32。
- **结果**：α=0.10 → 2.21693（最优）；0.15 → 2.22008；0.25 → 2.22309；0.40 → 2.23147；k_max32 → 2.23185。单调：微尺度下 α 越小越好，放宽 k_max 反而有害。
- **防止单 seed 过拟合**：再用 s2024、s4242 配对核查 α=0.10，3/3 seed 一致更优（+0.00315/+0.00221/+0.00185，均值 +0.0024）。
- **决策**：最终代码默认改为 α=0.10，并用其重跑含 TTT 的最终 artifact（final_a010）；预注册 0.15 的 8 seed 结论原样保留报告，不掩盖迭代过程。
- **教训**：①调参必须跨 seed 验证，单 seed 优势（0.003）小于 seed 间 std（0.0056），直接改主结论会过拟合；②预注册值与后续优化值都要报告，这才是可审计的实验流程。

## 2026-10-03 22:25 ｜ 独立评估器与训练管线结果不一致：artifact 不自描述（已修复并交叉验证）

- **现象**：`eval_bpb.py` 独立评估最终 artifact：plain BPB 2.446（管线报 2.217），TTT 2.180（管线报 2.165），明显不一致。
- **排查**：先怀疑重建模型缺参数——写 diag_keys.py 比对，missing/extra 均为空，排除。
- **根因**：独立评估子进程用 Config 默认值构建模型（parallel_residual=False、depth_recurrence=False），却加载 PR+DR 拓扑训练出的权重；参数名完全相同（PR/DR 不增参数），但计算拓扑不同 → BPB 不一致。
- **修复**：①训练打包时把架构标志（parallel_residual/pr_from_layer/depth_recurrence/recur_layers/模型形状/seq_len）写入 MANIFEST.json，让 artifact 自描述；②eval_bpb 构建模型前先从 MANIFEST 恢复架构；③用 repack_final.py 重组最终 tar（权重未变）。
- **交叉验证结果**：独立评估器复现 plain 2.217481354（MANIFEST 2.217481）、TTT 2.165182052（2.165182），逐位一致。
- **教训**：①"能跑通"不等于"可独立复现"，独立评估器必须与训练进程共享同一配置来源——最可靠的是把配置写进 artifact 本身；②验证交付物要用与生成不同的路径重新读数并比对，这次正是交付前的独立验证抓到了问题。

## 2026-10-04 00:14 ｜ RTX 5060 GPU 独立复现（安装/两个 bug/结果）

- **环境发现**：本机实际有 NVIDIA GeForce RTX 5060 Laptop GPU（8 GB，Blackwell sm_120，驱动 573.24/CUDA 12.8），早期侦察漏掉。
- **安装**：南京大学镜像下载 torch 2.11.0+cu128（2,753,148,611 字节，大小逐字节一致）；另装 triton-windows 3.8.0.post29 以支持 torch.compile（Windows 无内置 triton）。
- **Bug 1：累加器设备不一致**——bpb.py/ttt.py 的 loss_sum 等标量累加器默认建在 CPU，CUDA 上累加 GPU loss 报错。修复：累加器显式建在 device 上。
- **Bug 2：analyze 解析**——final_a010 等特殊 run_id 不符合 `Config_scheme_seed` 模式导致解析崩溃；改为从 metadata 取 seed、跳过不含 scheme 标记的特殊 run；并给 analyze/run_experiments 增加 prefix 机制，CPU 与 GPU 证据分开存档不互相覆盖。
- **GPU 结果（n=8，bf16 autocast + compile）**：架构 PR −0.00036（p=0.36 中性）、DR −0.00419（p=0.0078 有害）、Full −0.00399（p=0.0078）；量化 A-SDClip vs SDClip −0.34187（p=0.0078，方向与 CPU 一致，CPU 为改善 0.348）；A-SDClip vs int8 +0.0267（p=0.0078，仍略逊 int8）；TTT 改善 +0.03924（p=0.0078，CPU 为 +0.0515）。
- **结论**：跨硬件（CPU fp32 / GPU bf16）独立复现，全部主结论方向一致、显著性一致；绝对数字的小差异（尤其 TTT）来自 bf16 适配更新，已如实记录。
- **教训**：①设备相关代码（标量累加器、calib 设备）必须在目标设备上实测，静态检查发现不了；②多硬件复现是检验结论稳健性的强证据，值得做。

## 2026-10-04 00:42 ｜ TTT 精度消融：fp32 比 bf16 平均好 0.012（配对实测，已设为默认）

- **发现线索**：独立评估器（fp32 模型）复现 GPU artifact 的 TTT 为 2.1654，而 GPU 管线（bf16 模型）自报 2.1776，差异显著，怀疑 TTT 精度影响。
- **第一次测试失败（测试本身有 bug）**：给评估器加 --ttt-bf16 开关后配对比较，fp32/bf16 结果完全相同——排查发现评估器构建模型时从未像训练管线那样把模型转成 bf16，两次实际都在 fp32 上跑。修复：bf16 分支显式执行 model.bfloat16() 且 CastedLinear 保持 fp32，与训练管线一致。
- **干净配对结果（同一 artifact，仅精度不同，3 seeds）**：s1337 fp32 2.16476 / bf16 2.17679（Δ0.01203）；s555 2.16422 / 2.17580（0.01159）；s2024 2.16476 / 2.17666（0.01191），3/3 高度一致。
- **结论与决策**：fp32 TTT 稳定优于 bf16 约 0.012 BPB；把 TTT_FORCE_FP32=True 设为默认（Config 与 MANIFEST 均记录），8 个 GPU cell 已按 fp32 重跑，TTT 改善从 bf16 的 +0.039 变为 +0.049，与 CPU fp32 的 +0.052 一致。
- **教训**：①评估器必须完整复现被测管线的数值环境（含模型 dtype），否则比较无效；②"对照组与实验组结果完全相同"往往说明控制条件没有真正生效，是测试设计 bug 的信号；③消融比较必须在同一检查点/同一 artifact 上配对——跨运行比较会被训练噪声混淆（GPU bf16 训练跨运行约 ±0.005）。
