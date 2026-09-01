# GFlowNet 遇见大语言模型

> 本文为应用专题，配合资料清单 LLM 论文阅读。
> 来源：GFlowNet 调研 2026-08 审查扩充（E15）。核心索引见 [README](README.md)。

---

> 起草日期：2026-08-14。本节所有 venue 均于当日联网逐篇核实（会议官方 program 页 / OpenReview / PMLR / ACM DL / 作者代码库 / arXiv），
> 并与 `docs/` 主 catalog 的 `A21`–`A35`、`T44`/`T54`/`T57` 编号对齐；catalog 未编号的论文用其 arXiv 标识引用。
> 遵循「只引用编号 + venue 准确」的声明纪律，不复制标题与链接堆叠。

把一个自回归大语言模型（LLM）看作在「前缀 → 下一个 token」的 DAG 上逐步构造终止序列的策略，GFlowNet（GFN）就有了天然落点：
它的训练目标不是最大化某个奖励 \(R\)，而是把采样分布对齐成 \(P(x)\propto R(x)\)，于是在「高质量」之外额外索取「多样、覆盖多模」。
这恰好对治 RLHF / PPO / GRPO 一类奖励最大化方法在后训练中反复出现的 mode collapse。
这条线自 `A21`（ICLR 2024）起步，到 2026 年已在推理、对齐、安全、推荐四个方向同时开花，并催生一批专门处理长序列信用与配分函数的训练目标。
下文按**推理、对齐/采样、红队安全、推荐/其他**四条线梳理，每条给出核心动机、代表作与相对非 GFN 基线的差异，最后收束到统一视角。

**两处比 catalog 卡片更精确的 venue（本节以官方日程为准）**：

- `A21` 实为 **ICLR 2024 Oral（Honorable Mention）**（官方 oral 页），catalog 卡片仅记 ICLR 2024。
- `A32` 的 ICML 官方 program 页（poster/64302，已当日核对）标注为 **Poster**；部分机构新闻稿称「Spotlight」，本节按 ICML 官方日程取 Poster。

---

## 一、推理：用 GFN 微调 LLM 生成多样推理路径

**核心动机**：一道多步推理题往往有多条都正确、但形态不同的解。
只模仿一条（SFT）或只追最高分（reward-max RL）都会牺牲解的多样性与鲁棒性；
GFN 把「从奖励诱导的后验里采样整条推理轨迹」当作训练目标，天然产出一组解而非一个解。

**代表作**

- **`A21` · Amortizing Intractable Inference in LLMs** — ICLR 2024 Oral（Honorable Mention），arXiv:2310.04363
  奠基之作：把思维链（CoT）解释为潜变量模型，用 GFN 微调 LLM 从 intractable 后验摊销采样，明确把 distribution-matching 定位为 MLE 与 reward-max RL 之外的第三条路。
- **`A25` · Flow of Reasoning** — ICML 2025（PMLR v267），arXiv:2406.05673
  把多步推理写成 DAG 上的 Markovian flow，仅用约 15 个可验证样例，就在 BlocksWorld、Game24、GSM8k 等六个任务上同时提升正确率与路径多样性。
- **Latent Thought Flow** — 预印本 2026-06，arXiv:2606.16222
  `A25` 的潜空间版：用连续 GFN（承接 `T12` 理论）在 LLM 隐状态里采变长「潜思维」轨迹，按「正确性 vs 计算代价」的后验分配概率，并以 Entropy-Weighted SubTB 应对稀疏答案监督。
- **`A33` · GFlowRL** — 预印本 2026（微软），arXiv:2607.13394
  把分布匹配 RL 扩到 dense 与 MoE（至 235B）：移除可学习的 prompt-conditional 配分函数、改用 rollout group 的 in-batch Monte Carlo 估计，再加重要性采样校正与非对称 flow-gap clipping，在 FlowRL 发散的噪声红队与 MoE 场景下仍稳定收敛。
- **`T54` · RapTB** — ICML 2026，arXiv:2603.00454
  面向长序列的训练组件：用 rooted-prefix 约束 + absorbed-suffix 回传提供 dense 的前缀级信用，配 submodular replay（SubM）抑制 prefix collapse 与长度偏置。

**与 RLHF / GRPO / PPO 的关系**：GFN 与最大熵 RL 有精确对应（`T14`/`T15`），relative TB 更被证明等价于 Trust-PCL（`T44`），
PPO 式 clipped update 也能搬到摊销离散采样（`T57`）。差别不在优化器而在目标：奖励最大化把概率质量堆到单一峰，GFN 的 \(P\propto R\) 保留整座多峰地形。
`A33` 的 in-batch Monte Carlo 基线与 GRPO 的 group 归一化在形式上很接近，却保留了分布匹配语义；`A33` 与 PACED-RL（见第四线）都报告在 math/code 上优于 GRPO。

**与非 GFN 基线差异**：SFT 需海量标注且只模仿单一路径；reward-max RL 样本效率高但多样性塌缩、易 reward hacking。
GFN 线用少样本 + 可验证奖励换来「一组高质量解」，代价是长轨迹信用分配与 \(Z\) 估计更棘手（见第五节）。

---

## 二、对齐 / 采样：从预训练 LLM 摊销后验

**核心动机**：许多对齐与受控生成任务本质是从 \(p_\text{post}(x)\propto p_\text{prior}(x)\,r(x)\) 采样——预训练模型是先验，约束/奖励是似然。
逐 query 跑 MCMC 太贵，GFN 提供「一次训练、处处采样」的摊销后验采样器。

**代表作**

- **`A22` · relative Trajectory Balance（RTB）** — NeurIPS 2024，arXiv:2405.20971
  提出渐近无偏、data-free 的 RTB 目标，把 diffusion / 离散扩散 LLM 先验微调到目标后验，覆盖视觉 classifier guidance、语言 infilling、文生图与离线控制。
- **`A21` · Amortized sampling from LLM** — ICLR 2024 Oral，arXiv:2310.04363
  自回归 LLM 版的摊销后验采样；RTB 可视为它在迭代式生成过程上的推广。
- **`A34` · PowerFlow** — ICML 2026，arXiv:2603.18363
  把无监督微调（RLIF）重构为分布匹配，用 GFN 作 amortized variational sampler 去匹配基模型的 \(\alpha\)-power 分布 \(p_\text{base}^{\alpha}/Z\)：\(\alpha>1\) 锐化增强推理、\(\alpha<1\) 平滑释放创造性多样性，并用 length-aware TB（LA-TB）消除自回归长度偏置。
- **Outsourced Diffusion Sampling** — ICML 2025（PMLR v267），arXiv:2502.06999
  把 RTB 式后验推断放到任意生成模型的外生噪声空间（噪声空间后验更平滑），是 `A22` 面向 GAN/VAE/flow 先验的规模化后续。

**与非 GFN 基线差异**：classifier guidance、best-of-\(n\)、reward-weighted SFT 往往只近似后验或偏向单点；RTB 给出可证明无偏的后验目标并支持 off-policy 训练。
`A34` 的论点尤其说明问题：power 变换是单调的，只改熵、不改相对排序与多模结构，因此不像 RLHF/GRPO 那样把质量「漂移」到基模型 support 之外——这是分布保真相较外部奖励对齐的独特优势。

---

## 三、红队 / 安全：发现多样攻击

**核心动机**：安全红队要的恰恰是「多种不同的攻击模式」而非一个最强 prompt。
RL 红队即便加多样性正则也常 mode collapse 或攻击失效，这正是 \(P\propto R\) 采样的主场。

**代表作（一条清晰的谱系）**

- **Learning Diverse Attacks** — ICLR 2025，arXiv:2405.18540
  **奠基论文**：两阶段——先用 GFN 微调 attacker LM 采多样高奖励攻击，再用 MLE 平滑；攻击跨目标模型可迁移，用其数据做安全微调后更抗未见攻击。
- **`A32` · Stable-GFlowNet** — ICML 2026 Poster，arXiv:2605.00553
  直面安全 reward 高噪与 \(Z\) 估计不稳：以 Contrastive Trajectory Balance（成对比值，\(Z\) 自动约掉）、Noisy Gradient Pruning、Min-K Fluency Stabilizer 三件套，在保持 GFN 最优策略的同时把独特攻击类型从 17 提到约 134、ASR 维持约 92%。
- **Generating Attacks for LLMs with GFlowNets** — 预印本 2026-08，arXiv:2608.10171
  自适应地以一模型攻另一模型并给出量化 robustness score，首次把 GFN 红队扩展到英语之外（土耳其语）。
- **旁支：Active Attacks / EraseFlow** — arXiv:2509.21947（作者代码库标 ICML 2026）；`A26`（NeurIPS 2025）
  Active Attacks 是可插拔模块，靠周期性安全微调 victim 使已开发区域奖励变平、逼 attacker 走 easy-to-hard 课程，可叠加在 GFN/PPO/REINFORCE 上；EraseFlow 用 GFN 探索多样「去概念/擦除」轨迹，属安全编辑一侧。

**与非 GFN 基线差异**：RL 红队优化「最伤害」的单点，覆盖窄、易被针对性安全微调堵死；GFN 红队优化「攻击分布」，覆盖更广、迁移更好，用作防御数据也更稳。
核心工程难点几乎都压在 \(Z\) 上：`A32` 用成对对比消 \(Z\)，与推理线 `A33` 用 in-batch Monte Carlo 顶替 \(Z\) 异曲同工。

---

## 四、推荐 / 其他

**核心动机**：推荐与 prompt 优化同样苦于「只推最热门 / 收敛到单一提示」。GFN 的多样性与离线支持约束在这里价值直接。

**代表作**

- **GFN4Rec** — KDD 2023，arXiv:2306.02239
  起点论文：列表级推荐用 log-scale reward matching + 自回归选择，内在地提升推荐多样性。
- **CFlower · Conservative GFN for LLM Recommenders** — ICML 2026 Poster
  形式化离线 LLM 推荐中 SubTB 的三类不可辨识失败（流高估、前向质量泄漏、后向补偿），提出惩罚「数据支持外前向流」的保守目标；其前身为 Flower（arXiv:2503.07377，token 级流做过程监督，预印本）。
- **PACED-RL · Beyond Normalization** — 预印本 2026-02，arXiv:2602.12642
  把 GFN 联合学习的 \(Z\) 反用为 per-prompt 期望奖励（在线准确率）信号，做难度感知的 prompt 选择与 accuracy-error 优先 replay；两个组件复用训练已产生的信息，几乎零额外开销，报告优于 GRPO。
- **相关：`A24` COFlowNet / `A35` GFlowPO** — ICLR 2025 / 预印本 2026
  `A24` 约束离线流不外推到数据 support 之外；`A35` 对离散 prompt/策略做分布式优化以保留多解。

**与非 GFN 基线差异**：交叉熵 / SFT 推荐会放大流行度偏置、多样性差；GFN 用流匹配对齐「生成概率 ↔ 奖励」，并能在离线约束下显式惩罚 support 外外推——
CFlower 正是把 `A24` 式 offline 支持失配问题在 LLM 推荐上的具体化与修补。

---

## 五、统一视角：LLM 场景下 GFN 的卖点与挑战

**独特卖点**

1. **多样性 / 反 mode collapse**：\(P\propto R\) 天生保留多峰，贯穿四条线——多解推理、多样攻击、多样推荐、创造性生成。
2. **分布保真**：只重塑概率质量而不越出基模型 support（`A34` 的 \(\alpha\)-power 单调性论证最典型），对齐时比外部奖励更少「漂移」与能力遗忘。
3. **unknown-\(Z\) 摊销**：GFN 从设计上就服务于「未归一化奖励 + 难算配分函数」的采样，一次训练即得可反复廉价采样的后验采样器（`A21`/`A22` 的摊销 Bayesian inference）。

**主要挑战**

1. **大动作空间**：token 词表巨大、组合爆炸，使绝对流匹配的梯度方差极高。
2. **长序列信用**：终点奖励要回传到早期前缀，易 prefix collapse 与长度偏置（`T54` 专治；Latent Thought Flow 用 EW-SubTB 缓解）。
3. **\(Z\) 估计**：可学习的 prompt-conditional \(Z\) 在大模型、长 rollout、噪声奖励叠加时，成为梯度不稳的主要来源。

**贯穿性洞见（最重要）**：四条线其实都在围着同一个对象打转——配分函数 \(Z\)。
它既是 GFN 相对普通 RL 的「额外负担」（要多学一个归一化量），又是它独有的「资产」。
2026 年的工作因此分成两派：一派**消灭 \(Z\)**——`A32` 用成对对比让 \(Z\) 约掉、`A33` 用 in-batch Monte Carlo 顶替可学习 \(Z\)、`A22` 的 RTB 用相对目标绕开绝对流；
另一派**善用 \(Z\)**——PACED-RL 把 \(Z\) 读成在线准确率当难度调度器、`A34` 把 \(Z\) 重参数化为 token 级能量以消长度偏置。
同一个量，一半人想删掉、一半人当宝藏，这个「\(Z\) 的双重角色」是理解 GFN×LLM 现状最省力的一根主线。

**给读者的一条实操路径**：先读 `A21` 建立「LLM 微调 = 摊销后验采样」的直觉，再按需求分叉——
要多样推理走 `A25`/`T54`，要对齐或受控生成走 `A22`/`A34`，要安全红队走 `A32` 及其前身（arXiv:2405.18540），要推荐或 prompt 优化走 CFlower / PACED-RL；
评测时务必同时报告分布指标（模式覆盖率、解多样性）与质量指标，切勿只看平均奖励——这正是 GFN 类方法区别于 reward-max RL 的评价底线。

