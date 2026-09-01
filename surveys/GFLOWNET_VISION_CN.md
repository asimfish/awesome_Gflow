# GFlowNet 在视觉与多模态生成中的应用

> 本文为应用专题，配合资料清单视觉论文阅读。
> 来源：GFlowNet 调研 2026-08 审查扩充（E16）。核心索引见 [README](README.md)。

---

> 论文核实：2026-08-14 逐篇核对会议官方 proceedings、OpenReview/CVF Open Access 与 arXiv 元数据。
> 引用纪律：catalog 已收录的 GFlowNet 论文只用编号（A21–A29）引用，不重复堆链接；对照用的纯扩散对齐方法和尚未收录的 TILDE 属 catalog 外工作，给出方法名、venue 与 arXiv，并在末尾对照表统一列出。venue 冲突时以正式 proceedings 为准。
> 交叉引用：diffusion sampler 与 relative TB 的连续路径测度理论详见 [E04](E04_diffusion_sampler_line.md)，本节聚焦视觉/多模态的应用形态，不重复其定理推导。

## 1. 核心定位：视觉任务里 GFlowNet 到底解决什么

视觉与多模态生成里，"最高 reward 的那一张图/那一条推理"往往不是我们真正想要的。三个反复出现的失败模式，恰好对应 GFlowNet 的三类用法：

1. **文生图 prompt 的多样性坍缩**：把用户 prompt 改写成"模型偏好 prompt"以提升美学/对齐分数时，RL 优化器会收敛到少数相似后缀（deterministic postfix），牺牲语义多样性。
2. **扩散模型 reward 微调的模式坍缩与先验损伤**：用 aesthetic/preference reward 微调 Stable Diffusion 时，reward 最大化会过拟合 reward、丢掉预训练先验、坍缩到少数模式（over-optimization / reward hacking）。
3. **视觉推理只走单条路径**：VLM 做多步规划/推理时，SFT 假设 IID、PPO 最大化累计回报，都压制了"多条同样正确的解题/规划路径"。

GFlowNet 把这三件事统一改写成同一个目标：**不最大化 reward，而是按 reward 归一化后的密度采样** \(P^\star(x)=R(x)/Z\)。于是"多样但高质量"从一个需要额外正则的副目标，变成训练目标本身。视觉场景的特殊性在于：构造过程通常不是从零离散搭建对象，而是**微调一个已经很强的预训练生成器**（扩散模型或 VLM），因此几乎所有视觉 GFN 工作都围绕"如何相对预训练先验做修正"展开——这条主线贯穿下面所有论文。

## 2. 关键论文线（venue 已核）

### 2.1 扩散模型 reward 微调：从 relative TB 到 ∇-GFlowNet（A22 → A23）

- **A22（NeurIPS 2024）** 把 GFlowNet 用于扩散**先验微调**：给定预训练扩散先验 \(p_\theta\) 与约束/奖励 \(r\)，用 relative TB（RTB）约束

\[
Z_\phi\,p^{\mathrm{post}}_\phi(x_T\!\to\!\cdots\!\to\!x_0)=r(x_0)\,p_\theta(x_T\!\to\!\cdots\!\to\!x_0)\quad\forall\tau
\ \Longrightarrow\
p^{\mathrm{post}}_\phi(x_0)\propto p_\theta(x_0)\,r(x_0),
\]

  即后验 sampler 采样正确（渐近无偏）。RTB 比较的是**两个生成方向过程**的比值，可严格解释为"以预训练路径测度为参考测度的 TB 特例"，loss 不需对采样链反传，因此天然支持 off-policy（从高密度样本倒走加噪轨迹、replay buffer）——这正是它相对 on-policy RL 微调在模式覆盖上的优势来源（详见 [E04](E04_diffusion_sampler_line.md)）。

- **A23（ICLR 2025）** 是这条线在视觉上最锋利的一刀：∇-GFlowNet（∇ 读作 nabla）指出标量 reward 信号太稀疏，改用**可微 reward 的梯度** \(\nabla r\)。它把 detailed balance 从标量层面提升到 **score/梯度层面**，提出 ∇-DB 目标，并给出 **residual ∇-DB** 变体——只学习相对预训练扩散模型 score 的**残差修正**，微调时同时平衡四个"力"：微调模型 score、预训练模型 score、（预测的）reward 梯度、学习到的残差 flow score。结果是在 Stable Diffusion 上以约 200 步更新即可完成 reward 微调，且在收敛速度、样本多样性、先验保持之间取得更好折中。

把 A22 与 A23 并置能看出一个统一模式：**二者都不直接学新分布，而是学"相对预训练先验的修正"**——RTB 在轨迹标量层面做相对参数化，∇-DB 在 score 梯度层面做残差参数化。这个"相对/残差"思想是后面所有下游应用能直接复用的接口。

### 2.2 文生图 prompt 适配：PAG（A29，CVPR 2025）

**A29** 处理的是**冻结文生图模型之外层的离散 prompt 搜索**：方法名 PAG（Prompt Adaptation with GFlowNets），把"prompt 改写"从 RL 的 reward 最大化改写成概率推断，用 GFlowNet 按未归一化密度采样一族有效且多样的 prompt。它的贡献不止是套用 GFN，而是诊断出一个此前被忽视的现象：朴素 GFN prompt optimizer 会经历**神经可塑性损失（plasticity loss）**，叠加序列 prompt 生成中低效的信用分配，导致 mode collapse。PAG 用 **flow reactivation + reward-prioritized sampling + reward decomposition** 系统性缓解，并展示跨 reward 函数的稳健性和跨文生图模型的迁移性。它是"GFN 在生成器外层做离散控制"的代表。

### 2.3 视觉语言多步推理：GFlowVLM（A28，CVPR 2025）

**A28** 把 GFlowNet 引入 VLM 的**多步推理/规划**：将观测与任务描述作为输入，用 chain-of-thought 推理引导动作选择，把整条推理轨迹当作 GFN 构造过程，并用 `[DONE]` token 显式建模终止状态。关键设计是把环境建模为**非马尔可夫决策过程**（历史进入状态），用 DB / SubTB 目标训练。在 NumberLine、BlackJack 与具身规划 ALFWorld 上，它相对 SFT 与 PPO 提升了训练效率、解的多样性以及分布内/分布外泛化——是视觉主会里最直接以 GFlowNet 为核心方法的工作之一。

### 2.4 概念擦除/遗忘与潜图条件化（A26、TILDE、A27）

"相对/残差参数化"最有说服力的验证，是它被安全场景直接复用：

- **A26 EraseFlow（NeurIPS 2025）**：把概念擦除（concept unlearning）视为**去噪路径空间的探索**，用 TB 目标对完整去噪轨迹重加权，让生成远离目标概念同时保留先验。它进一步证明**常数 reward + TB** 即可可靠擦除语义内容，从而摆脱脆弱、可被 hack 的外部 reward model，并泛化到未见概念。
- **TILDE（TILt-based Distributional Erasure，arXiv 2607.06432，预印本 2026-07-07）**：把概念遗忘写成**分布对齐**问题——目标是预训练模型在"遗忘约束"下的最小偏离条件分布（minimum-deviation、anchor-free），用一个 forget energy 对每个 prompt 条件分布做能量 tilt。它明确用 **residual ∇-GFlowNet 训练**（直接建立在 A23 之上）来学习 forget-energy tilt 相对预训练扩散模型诱导的 score 修正。这是 A23 的方法接口被下游安全工作原样继承的直接证据。
- **A27（NeurIPS 2025）**：先用 GFlowNet 对可解释的**离散潜在图**做多样后验式搜索，再条件化图像生成，以覆盖多种关系配置——代表"GFN 负责结构化潜变量、扩散负责像素"的分工形态。

## 3. 技术要点

### 3.1 目标函数：相对 TB 与 ∇-DB（相对经典 TB 的改动）

经典 TB 直接约束 \(Z\prod_n P_F=R(x)\prod_n P_B\)，需要估计全局 \(Z\) 且从零学 \(P_F\)。视觉微调的两种主流改法都改成"相对预训练先验"：

- **相对 TB（A22）**：把参考测度从"均匀/无信息"换成**预训练路径测度** \(p_\theta(\tau)\)，只学后验相对先验的比值与一个标量 \(Z_\phi\)；等价于以 \(p_\theta\) 为参考的 TB，因此保留 off-policy 训练的零点不变性。
- **∇-DB / residual ∇-DB（A23）**：把 DB 残差对状态求梯度，令 detailed balance 在 **score** 层面成立，从而把可微 reward 的 \(\nabla r\) 当作比标量 \(R\) 密得多的监督；residual 形式令网络只输出相对预训练 score 的修正项，天然做 prior preservation。

### 3.2 像素 / latent 空间里的构造过程

扩散去噪链天然就是 GFlowNet 轨迹：状态取 \((x_t,t)\)，前向策略 \(P_F\) 是**生成方向**（去噪）转移核，后向策略 \(P_B\) 是加噪核，终止状态 \(x_0\) 以边流进汇点、终点边流由 reward 决定。对 latent diffusion（如 SD），构造发生在 VAE 的 **latent 空间**，而 reward 通常在解码后的**像素图**上用 reward model 评估——构造空间与打分空间分离是视觉设定的固有特征。VLM 推理（A28）则把"离散 token/动作序列"当轨迹，`[DONE]` 决定终止，是离散构造；prompt 适配（A29）也在离散 token 空间构造，只是生成器被冻结。

### 3.3 reward 设计

- **可微 reward**（aesthetic score、偏好模型、CLIP 相似度）：可喂给 ∇-GFlowNet 用其梯度（A23）。
- **黑箱/不可微 reward**：relative TB / TB 只需标量即可（A22、A28）。
- **无 reward model**：概念擦除用**常数 reward + TB**（A26），或用 forget energy 定义 tilt 目标（TILDE），把"reward 设计"变成"目标分布设计"，规避 reward hacking。
- **reward 分解**：序列 prompt 生成中把终点 reward 分解到中间步以缓解信用分配与可塑性损失（A29）。

## 4. 与纯扩散对齐方法的对比

主流纯扩散对齐方法都以"reward/偏好最大化"为核心，GFlowNet 的差异在于把目标换成"按 \(p_{\text{prior}}\!\cdot\! r\) 采样"，因此多样性与先验保持是目标内生量而非事后正则：

| 方法（venue） | 优化目标 | 需可微 reward | 多样性/先验 | 数据 |
|---|---|---|---|---|
| DDPO（ICLR 2024） | policy-gradient RL（PPO），最大化标量 reward | 否 | 易 mode-seeking，需 KL 正则护先验 | on-policy 为主 |
| DRaFT（ICLR 2024） | 沿采样链反传 reward 梯度 | 是 | 收敛快但易 over-optimization、坍缩多样性 | on-policy |
| Diffusion-DPO（CVPR 2024, pp.8228–8238） | 成对偏好直接优化，无需显式 reward model | 否 | 隐式 reward 最大化，仍偏 mode-seeking | 离线偏好对 |
| **GFN：RTB（A22）/ ∇-GFN（A23）** | 按 \(p_\theta(x)\,r(x)\) 归一化采样 | ∇-GFN 用梯度；RTB 不需 | **多样性+先验保持为目标内生** | 支持 off-policy / replay |

要点：DDPO/DRaFT/Diffusion-DPO 都要靠 KL-to-prior 之类的外加正则来"防止跑偏"，而 GFN 把"贴近先验且按奖励重加权"直接写进最优解；代价是需要可靠估计相对配分量、且长去噪链上的信用分配与梯度方差更难控（见 §5）。A23 论文正是以 DDPO/DRaFT 一类为对照，主张在同等 reward 下取得更优的多样性—先验折中。

## 5. 开放问题

1. **正确性评估**：像素/latent 分布无法枚举，"采样正确"只能靠 recall/coverage、reward 分布、FID 等代理指标间接判断；低 loss 是否对应真实分布贴合（对应 T32 的诊断问题）在视觉上尤其难验证。
2. **长去噪链的信用分配与方差**：几十步去噪的 TB 乘积方差大，∇-DB 靠 score 监督缓解但依赖 reward 可微且梯度可靠。
3. **reward hacking 是否真被缓解**：GFN 的分布目标理论上抑制过优化，但在强 reward model 下是否仍被 hack，缺乏视觉基准的系统证据。
4. **外层离散控制与内层连续微调的统一**：PAG（A29，冻结生成器、离散 prompt）与 ∇-GFN（A23，微调 latent 去噪）是两种范式，能否在一个框架里协同尚不清楚。
5. **多模态推理的状态表示扩展性**：GFlowVLM（A28）的非马尔可夫状态把历史塞进 prompt，长程任务下的可扩展性与终止判定仍是瓶颈。
6. **概念擦除的双重保证**：遗忘彻底性与良性分布保真度的权衡，TILDE 的 minimum-deviation 目标是否可在部署中被验证，仍是开放的安全问题。
7. **与 Schrödinger bridge / OT 的连接**：视觉微调"贴近先验 + 满足约束"与熵正则 OT 结构同源（见 [E04](E04_diffusion_sampler_line.md) §与 SB 的衔接），但尚无定理说 GFN 微调等价于求某个 bridge。

## 6. venue 与 arXiv 对照表（catalog 外工作与对照方法）

| 工作 | 角色 | venue | arXiv |
|---|---|---|---|
| A22 RTB | catalog 内，编号引用 | NeurIPS 2024 | 2405.20971 |
| A23 ∇-GFlowNet | catalog 内，编号引用 | ICLR 2025 (Poster) | 2412.07775 |
| A26 EraseFlow | catalog 内，编号引用 | NeurIPS 2025（作者页标 Spotlight） | 2511.00804 |
| A28 GFlowVLM | catalog 内，编号引用 | CVPR 2025 (pp.3815–3825) | 2503.06514 |
| A29 PAG | catalog 内，编号引用 | CVPR 2025 | 2502.11477 |
| TILDE | catalog 外（本节新增核实） | 预印本 2026-07-07 | 2607.06432 |
| Diffusion-DPO | 纯扩散对齐对照 | CVPR 2024 (pp.8228–8238) | 2311.12908 |
| DDPO | 纯扩散对齐对照 | ICLR 2024 | 2305.13301 |
| DRaFT | 纯扩散对齐对照 | ICLR 2024 | 2310.03739 |

## 7. 声明纪律：哪些已证明，哪些是经验

**论文已证明（引用时注意各自条件）**：
- RTB 约束成立 ⟹ 后验边缘 \(p^{\mathrm{post}}(x)\propto p_\theta(x)r(x)\) 正确，且 RTB 是参考测度意义下的 TB（A22）。
- ∇-DB / residual ∇-DB 在连续 GFlowNet 框架下的目标定义与 prior-preserving 性质（A23）。
- 常数 reward + TB 可可靠擦除语义并保留先验的理论结论（A26）。

**经验规律 / 结构对应（不要当定理引用）**：
- "GFN 微调比 DDPO/DRaFT/Diffusion-DPO 更保多样性和先验"：A23 及各文的经验对照结论，依赖 reward、基线与调参。
- "concept unlearning 的最小偏离目标最优"：TILDE 的建模主张，属预印本，需正式评审与更强基线核验。
- "与 SB/OT 的衔接"：结构同源而非等价定理（精确条件另见 OT 分析文档）。

