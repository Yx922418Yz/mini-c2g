# LiYaxuan_C2G_方案设计

李亚轩（Li Yaxuan），郑州西亚斯学院，AI+X Elite 20 Program
版本：v1.0，2026-10-03

## 1. 目标与定位

- 赛事：OpenAI Parameter Golf（2026-03-18 至 2026-04-30 窗口的原型赛事）。
- 硬约束：8×H100 SXM；训练 wallclock ≤ 600 s；评估 wallclock ≤ 600 s；artifact ≤ **16,000,000 十进制字节**（代码 + 压缩权重 + tokenizer）；评估期间断网、无外部调用。
- 指标：FineWeb validation 上的 Bits-Per-Byte（BPB，越低越好，tokenizer 无关）。
- 参照系：基线 ≈ 1.2244 BPB；2026-04 榜首（SP8192 + 3 层递归 + 并行残差 + 合法 TTT）= 1.0810 BPB。
- 目标级别：**按 L3 Gold（BPB<1.12，≥2 方向组合、消融、p<0.01）与 L4 Platinum（BPB<1.085、原创改动、GitHub 仓库 CI 一键复现、≥6 页英文技术报告、10 分钟讲解视频）的最高可行标准组织交付。**

## 2. 总体技术栈（四个组件 + 一条量化链）

| 组件 | 来源 | 作用 | 评估方式 |
|---|---|---|---|
| 并行残差（PR） | GPT-J（Wang & Komatsuzaki, 2021） | 指定层起 attn/mlp 同读归一化输入、并行求和，省一次串行变换 | 配对 seed，pre-quant BPB |
| 深度递归（DR） | Universal Transformer（arXiv:1807.03819） | 指定物理层相邻重复一次，共享权重、增加虚拟深度 | 配对 seed，pre-quant BPB |
| 合法 score-first TTT | 2026-04-09 榜首记录及 TTT 文献（arXiv:2411.07279） | 评估时逐 chunk：先完整评分、再 SGD 适配；后续 chunk 受益 | TTT BPB vs post-quant BPB |
| **A-SDClip（原创）** | SDClip 记录 + GPTQ（arXiv:2210.17323） | 逐行峰度自适应裁剪倍数 k | 与全局 k SDClip 配对 |
| GPTQ-lite | GPTQ | Hessian 引导列序，配合 A-SDClip 尺度 | 同检查点 RTN vs GPTQ 误差 |

基线 B 保留官方 modded-nanoGPT 家族的全部默认设计：tied embedding、RMSNorm、RoPE、QK RMSNorm + 可学 q_gain、GQA、relu² MLP、可学逐通道残差尺度、U-Net skip、Muon + Adam 分组优化、tanh logit softcap。

## 3. 原创点：A-SDClip 的推导

### 3.1 问题形式化

对一个 fp32 权重矩阵 W∈R^(r×m)，按行做均匀量化。第 i 行先确定裁剪阈值 c_i，再以 qmax=2^(b−1)−1 级量化：scale_i = c_i/qmax，q=round(W_i/scale_i)，反量化 Ŵ=q·scale。

两类误差：
1. **裁剪误差**：|W|>c 的权重坍缩到边界 ±c；
2. **粒度误差**：bin 宽 2·scale，舍入噪声方差 ≈ scale²/3。

SDClip 取 c_i = k·std(row_i)，k 全局固定（int6 矩阵 k=12.85；int8 嵌入 k=20）。其熵分析给出压缩后每权熵 H(q) ≈ b − log2 k + const：k 增大 → 体积减小，但当 k 超过该行分布的实际尾部支撑后，粒度误差线性增大。因此**最优 k 是行分布形状的函数**，全局 k 只对"平均形状"的行最优。

### 3.2 自适应规则

对每行估计超额峰度 κ_i = E[(W−μ)^4]/σ^4 − 3：

- 重尾行（κ>0）：超过 kσ 的概率质量大于高斯情形，裁剪误差主导 → 增大 k；
- 近高斯行（κ≈0）：保持基准 k；
- 轻尾行（κ<0）：尾部支撑小于高斯，可减小 k 换取更小粒度误差。

小样本（行宽 m 有限）下样本峰度噪声大，做两步处理：
1. 正态理论偏差校正：κ_corr = ((m−1)/((m−2)(m−3)))·((m+1)κ_raw + 6)；
2. 经验贝叶斯收缩（峰度估计在正态下方差≈24/m）：κ_s = λ·κ_corr，λ = m/(m+24)。

最终裁剪倍数：

```
k_i = clip( k_base · exp(α · κ_s), k_min, k_max )
```

默认 k_base=12.85（int6），α=0.15，[k_min,k_max]=[8,20]。指数形式保证 k_i 恒正且对 κ 局部线性：k_i/k_base ≈ 1 + α·κ_s。κ∈[−1,2] 时调整幅度约 0.86–1.35 倍，温和可控。

### 3.3 下行风险有界

- 当 m 很小、峰度无信息时 λ→0，k_i→k_base，**严格退化回 SDClip**，不会比基准差；
- k 的硬边界防止漂移；
- α 扫描 {0.05,0.10,0.15,0.25} 检验敏感性；
- 若实测无收益，如实报告零结果（见 §7）。

## 4. 合法 score-first TTT 的合规性

严格满足挑战 Track-B 的四条解释：① 因果性：每个位置只由前缀评分；② 标准 softmax，无 logit 偏置、无 n-gram 缓存；③ **先评分后更新**：每个 chunk 先在 no_grad 下完整评分记录 NLL，再做 SGD；④ 单次评分：每个 token 只评一次、不重评。chunk c 的适配只影响尚未评分的后续 chunk，合法。默认 chunk=8192（微尺度）/32768（H100），epochs=3，SGD momentum=0.9，lr=0.005 并跨 chunk 余弦衰减，grad clip=1.0。

## 5. 实验矩阵

### 5.1 微尺度实测（本机 CPU，真实 FineWeb 头部分片：800 万训练 token / 200 万验证 token，官方 SP1024 tokenizer）

模型默认 d=128、6 层、4 头/2 KV、seq=256、batch=16,384 token、400 iter（先计时再锁定规模），验证取前 500,000 token。

训练配置（5 seeds：1337、2024、4242、5050、9090，视计时调整）：

| 配置 | PR | DR | 打包量化链 | TTT |
|---|---|---|---|---|
| B | – | – | int8 RTN | – |
| PR | ✓ | – | int8 RTN | – |
| DR | – | ✓ | int8 RTN | – |
| Full-int8 | ✓ | ✓ | int8 RTN | – |
| Full-sdclip | ✓ | ✓ | int6 全局 k | – |
| Full-asdclip | ✓ | ✓ | int6 A-SDClip | ✓ |

关键配对：架构比较用 int8 链的 pre-quant BPB（B/PR/DR/Full-int8）；量化比较在 Full 同一检查点上比较三条链的 post-quant BPB（训练确定性保证检查点一致）；TTT 比较 Full-asdclip 的 TTT BPB 与 post-quant BPB。

### 5.2 统计分析

- 双侧配对置换检验：n≤18 用精确枚举，否则 200,000 次蒙特卡洛；
- 报告每 seed 值、均值、标准差、配对差值 ΔBPB 与 p 值；
- L3 口径：改进 ≥0.005 nats 且 p<0.01；
- 量化配对同时报告 pooled（全部检查点）与逐配置分析，并说明独立性口径；
- 失败/不显著结果同样入报告。

### 5.3 H100 打榜路径

`scripts/reproduce_h100.sh` 一条命令：d=512、9 层、8/4 头、seq=1024、全局 batch=524,288 token、wallclock 门 575 s、PR 从第 2 层起、DR 重复 4/5/6 层、int6 A-SDClip + GPTQ-lite、TTT chunk=32768；打包后自动过 16 MB 门。

## 6. 工程与仓库标准（GitHub-ready）

- 模块化 `pgolf/` 包 + 四个根脚本；artifact 打包**把全部代码字节计入**（比 FAQ 的最低口径更严格，不隐藏字节），选择理由写入拿来说明。
- 确定性 tar（固定 mtime、成员排序），两次构建字节一致。
- README、LICENSE（MIT，保留 OpenAI 声明）、THIRD_PARTY_NOTICES、.gitignore、requirements / requirements-cpu 齐全。
- CI（GitHub Actions）：编译检查 + 合成数据上跑通基线与 Full（含 TTT）+ 打包门 + 独立评估器，全程不依赖外网数据。
- 数据获取脚本（PowerShell + bash，hf-mirror）保证真实数据管线可复现。

## 7. 风险、边界与失败预案

1. **无 GPU/API key 的边界**：本机无法产生真实 8×H100 BPB。所有本机数字均来自真实 FineWeb token 的微尺度实测；H100 成绩留待算力到位后一条命令产生，任何文档不虚构 H100 数字。
2. **A-SDClip 无效**：按草案第四节四步降级（定位→α 减半→缩小作用域→回退 SDClip 并报告零结果）。
3. **TTT 评估超时**：微尺度下可限制 TTT 评分 token 数并标注口径。
4. **微尺度外推风险**：架构组件在 d=128 的收益方向可能与 d=512 不同，报告中明确区分"微尺度实测结论"与"H100 待验证假设"。
5. **低级 bug 风险**（前序挑战的教训）：所有脚本先小规模单步验证，再批量运行；交付前用独立路径回读核验。

## 8. 时间线

1. 数据/tokenizer 就位（已完成）；
2. 代码开发与单步冒烟；
3. 微尺度矩阵运行（5 seeds，后台批量）；
4. 分析、消融、榜单与全部文档；
5. 英文技术报告（≥6 页）与讲解视频脚本/录制；
6. 打包、CI 验证、交付桌面 `提交挑战/C2G_ParameterGolf`，并具备直接推 GitHub 的条件。
