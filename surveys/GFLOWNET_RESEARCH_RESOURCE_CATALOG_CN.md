# GFlowNet 论文、课程与代码清单

> 最后核验：2026-07-31（扩充部分检索至 **2026-08-25**）；**2026-08 修订**：经 10 个质检 agent + 20 个扩充 agent 复核，修正链接/标题/发表状态勘误约 30 处，并在[第 11 节](#11-2026-08-扩充收录n-系列)扩充收录 106 篇（含并行审查流水线补录并入 18 篇、2026-08 追补 1 篇、顶会查全补录 6 篇）。  
> 配套正文：[《GFlowNet 理论调研与学习指南》](GFLOWNET_THEORY_GUIDE_CN.md)　｜　修订说明：[REVISION_SUMMARY](../REVISION_SUMMARY.md)  
> 收录规模：**206 篇独立论文** = 第 1–3 节原始主目录 100 篇（理论 T01–T57 / 最优传输 O01–O08 / 应用 A01–A35）+ 第 11 节扩充 106 篇（N001–N106）。每篇只出现一次并配简介；推荐路线、顶会审计和专题分析只引用编号，避免重复堆叠。

---

## 0. 先看这里

### 0.1 标记说明

| 标记 | 含义 | 阅读动作 |
|---|---|---|
| **P0** | 建立理论骨架不可缺少 | 精读、推导核心公式 |
| **P1** | 进入一条研究线的重要工作 | 按方向精读 |
| **P2** | 应用案例、补充视角或尚待验证的前沿 | 先扫读摘要、实验和限制 |
| **主会** | 已在会议官方 proceedings 或主会日程中出现 | 可以按正式会议论文引用 |
| **Workshop** | 会议同期研讨会论文，不是主会论文 | 引用时写清 workshop |
| **预印本** | 截止核验日未见正式接收信息 | 对理论和实验结论保留审慎 |

### 0.2 按读者类型选路线

这里仅列论文编号；完整简介和链接都在后面的唯一主目录中。前三条按**读者类型**分，后两条是理论读者的深入支线。

| 读者 | 建议顺序 | 说明 |
|---|---|---|
| **新手入门**（无 RL/VI 背景） | 先做 §0.1 的 90 分钟热身 → T01 → 亲手在小 HyperGrid 上跑通 FM/TB（见 §7）→ 回读 T02 → T03 → T05 | 不要一上来啃 Foundations（JMLR，测度论味重）。先有"流守恒 + 采样正确性"的手感，再补形式化 |
| **理论研究者** | T01 → T02 → T03 → T05 → T07 → T08 → T09 → T14 → T17 → T32 → T49 | 两周建立骨架；结尾 T32/T49 是"loss 到底保证了什么"的当前答案 |
| **应用研究者** | A01（综述，先建版图）→ 按分支各挑 3–5 篇 + 配套代码库：<br>· 分子/生物：A02 → A09 → A11 → N010 分支代码用 Recursion gflownet<br>· LLM/推理：A21 → A25 → N040 系（分布匹配 RL）→ T54（长序列信用）<br>· 安全/红队：N063（ICLR 2025 奠基）→ A32（ICML 2026 Spotlight）<br>· 组合优化：A19 → T41 → N079（GFACS） | 应用线不必先读完理论骨架，但**必须**读 T07 与 T32——否则容易只看 mean reward 就以为学对了分布 |
| *（支线）* 训练、稳定性与探索 | T05 → T07 → T17 → T24 → T32 → T34 → T45 → T47 → T51 → T49 → T50 | 理论读者深入训练动力学 |
| *（支线）* GFlowNet × OT | T02 → T19 → T36 → O01 → O02 → O07 → **O08** → 再看 §11.8/§11.9 的 SB/神经 OT 对照组 | 门槛较高：O08 代码 TBA，需自实现，宜按 6 周而非 4 周计划 |

开始读论文前，建议先用 90 分钟完成：

1. [The GFlowNet Tutorial](https://milayb.notion.site/The-GFlowNet-Tutorial-95434ef0e2d94c24aab90e69b30be9b3)；
2. [GFlowNet Playground](https://gfn-playground.caleydoapp.org/)；
3. 本地指南的[统一数学框架](GFLOWNET_THEORY_GUIDE_CN.md#2-统一数学框架)和[目标函数对照表](GFLOWNET_THEORY_GUIDE_CN.md#38-目标函数对照表)。

完成标准：能区分终止对象 \(x\)、构造轨迹 \(\tau\)、边流 \(F(s\to s')\)，并解释 GFlowNet 的目标是
\[
P^\star(x)=\frac{R(x)}{Z},
\]
而不是单纯寻找最大回报对象。

---

## 1. 理论与通用方法论文

### 1.1 奠基、训练目标与信用分配

| 编号 | 论文与状态 | 简介 | 建议 |
|---|---|---|---|
| **T01** | [Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation](https://proceedings.neurips.cc/paper/2021/hash/e614f646836aaed9f89ce58e837e2310-Abstract.html) · NeurIPS 2021 | GFlowNet 的原始论文，用 DAG 上的流匹配把终止对象的采样概率训练成与奖励成比例，并以分子设计和主动学习说明“多样高奖励采样”不同于最大化奖励。阅读重点是 flow matching、合流路径和摊销生成的最初动机。 | **P0 精读** |
| **T02** | [GFlowNet Foundations](https://jmlr.org/papers/v24/22-0364.html) · JMLR 2023 | 建立 Markovian flow、state/edge/trajectory flow、前向与后向策略、reward matching 等统一数学框架，并系统说明 FM、DB 等约束为何导出正确终止分布。它也是理解内部流非唯一性、条件 GFN 和后续 OT 连接的核心理论起点。 | **P0 精读** |
| **T03** | [Trajectory Balance: Improved Credit Assignment in GFlowNets](https://proceedings.neurips.cc/paper_files/paper/2022/hash/27b51baca8377a0cf109f6ecc15a0f70-Abstract.html) · NeurIPS 2022 | 提出轨迹级恒等式，将一条完整构造路径上的前向概率、后向概率、配分函数 \(Z\) 和终点奖励联系起来。TB 缩短信号传播路径，但也引入长轨迹乘积、方差和 \(Z\) 估计等训练问题。 | **P0 精读** |
| **T04** | [Generative Flow Networks for Discrete Probabilistic Modeling](https://proceedings.mlr.press/v162/zhang22v.html) · ICML 2022 | 从离散概率建模角度研究 Markov flow 的参数化和学习，澄清多条生成路径如何共同决定对象概率及其熵结构。适合在 Foundations 之后补足 GFlowNet 作为离散生成模型的统计视角。 | **P1** |
| **T05** | [Learning GFlowNets from Partial Episodes for Improved Convergence and Stability](https://proceedings.mlr.press/v202/madan23a.html) · ICML 2023 | 提出 Subtrajectory Balance，用任意子轨迹上的 balance 约束连接一步 DB 与完整 TB，并用 \(\lambda\) 控制信用分配尺度。它是理解 GFlowNet loss 的偏差—方差、局部—全局监督折中的关键论文。 | **P0 精读** |
| **T06** | [Better Training of GFlowNets with Local Credit and Incomplete Trajectories](https://proceedings.mlr.press/v202/pan23c.html) · ICML 2023 | 利用中间状态的局部能量或奖励信息，并允许从不完整轨迹学习，从而缓解只有终点奖励时的稀疏信用问题。论文的价值在于展示如何把领域可分解结构转化为训练信号。 | **P1** |
| **T07** | [Towards Understanding and Improving GFlowNet Training](https://proceedings.mlr.press/v202/shen23a.html) · ICML 2023 | 系统研究有限训练下 loss、终止分布误差和内部流之间的鸿沟，指出常用目标可能在欠拟合、稀疏奖励和长轨迹环境中产生误导。论文还提出 PRT、SSR、GTB 等改进，是从“零损失理论”走向实际训练诊断的必读工作。 | **P0 精读** |

### 1.2 VI、RL、Markov chain 与状态空间扩展

| 编号 | 论文与状态 | 简介 | 建议 |
|---|---|---|---|
| **T08** | [GFlowNets and Variational Inference](https://openreview.net/forum?id=uKiE0VIluA-) · ICLR 2023 | 比较 GFlowNet 与层次变分推断，说明特定条件下两者梯度估计之间的联系，同时强调采样分布、后向策略和 off-policy 数据会造成实质差异。它能防止把 GFN 和 VI 过度简化为无条件等价。 | **P0 精读** |
| **T09** | [A Variational Perspective on Generative Flow Networks](https://openreview.net/forum?id=AZ4GobeSLq) · TMLR 2023 | 在轨迹空间中构造前向与反向分布，并用 KL 目标统一解释 TB 及若干 GFlowNet 训练规则。阅读时要区分“对象边缘分布正确”与“选定的轨迹分布匹配”这两个层次。 | **P0 精读** |
| **T10** | [GFlowNet-EM for Learning Compositional Latent Variable Models](https://proceedings.mlr.press/v202/hu23c.html) · ICML 2023 | 把 GFlowNet 用作组合离散潜变量的摊销后验，并嵌入 EM 式参数学习循环。它展示了 GFN 不仅能做候选发现，也能服务于结构化潜变量模型中的后验推断。 | **P1** |
| **T11** | [Generative Flow Networks: A Markov Chain Perspective](https://arxiv.org/abs/2307.01422) · 预印本 2023 | 用 Markov chain 的语言重新表述 GFlowNet，帮助连接平稳分布、路径测度和循环状态空间。其主要价值是概念统一；使用其中等价关系时仍应逐项检查吸收性和边界条件。 | **P1** |
| **T12** | [A Theory of Continuous Generative Flow Networks](https://proceedings.mlr.press/v202/lahlou23a.html) · ICML 2023 | 将离散 DAG 上的流守恒推广到一般测度空间，处理连续及混合状态、可测流和密度等问题。它是研究连续动作、几何对象以及连续 GFlowNet–OT 扩展的理论入口。 | **P0 精读** |
| **T13** | [Stochastic Generative Flow Networks](https://proceedings.mlr.press/v216/pan23a.html) · UAI 2023 | 把环境转移本身具有随机性的情况纳入 GFlowNet，而不是假设动作唯一决定下一状态。该工作适合研究实验结果不确定、随机模拟器或带噪科学发现任务。 | **P1** |
| **T14** | [Generative Flow Networks as Entropy-Regularized RL](https://proceedings.mlr.press/v238/tiapkin24a.html) · AISTATS 2024 | 将 GFlowNet 写成特定的最大熵强化学习问题，连接 soft value、policy consistency 和 reward-proportional sampling。它为复用 RL 优化工具提供接口，但不能据此忽略多路径奖励校正。 | **P0 精读** |
| **T15** | [Discrete Probabilistic Inference as Control in Multi-path Environments](https://proceedings.mlr.press/v244/deleu24a.html) · UAI 2024 | 从控制角度推导离散概率推断，明确多路径结构中必须怎样校正奖励才能得到目标对象分布。论文有助于理解 GFlowNet、最大熵控制和普通“按终点奖励做 RL”的差别。 | **P0** |
| **T16** | [GFlowNet Training by Policy Gradients](https://proceedings.mlr.press/v235/niu24c.html) · ICML 2024 | 提出用 policy-gradient 形式训练 GFlowNet，并讨论前向与后向策略的联合设计。它拓宽了回归式 balance loss 之外的优化接口，适合与 PPO、actor–critic 路线对照。 | **P1** |
| **T17** | [On Divergence Measures for Training GFlowNets](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8aaae73117c9266b35eb977e70fcf95f-Abstract-Conference.html) · NeurIPS 2024 | 系统研究不同 divergence 如何诱导 GFlowNet 训练目标，并指出实践失败往往来自梯度估计方差而非 divergence 本身。论文进一步构造 control variate，是连接 loss 设计、VI 和稳定优化的重要工作。 | **P0 精读** |
| **T18** | [Expected Flow Networks in Stochastic Environments and Two-Player Zero-Sum Games](https://iclr.cc/virtual/2024/poster/17581) · ICLR 2024 | 用期望意义下的流约束扩展确定性 GFlowNet，使其能够处理随机环境和二人零和博弈。阅读重点是“逐转移守恒”放宽为“条件期望守恒”后，正确性需要哪些假设。 | **P1** |
| **T19** | [A Theory of Non-Acyclic Generative Flow Networks](https://ojs.aaai.org/index.php/AAAI/article/view/28989) · AAAI 2024 | 建立允许环的一般 GFlowNet 理论，用 expected visit flow 和吸收条件取代 DAG 中的简单路径计数，并揭示循环可能造成的 flow explosion。它是最短路、持续规划和 GFlowNet–OT 方向的第一块理论基石。 | **P0 精读** |
| **T20** | [Diffusion Generative Flow Samplers](https://arxiv.org/abs/2310.02679) · ICLR 2024 | 将 GFlowNet 的部分轨迹训练信号引入连续迭代式 diffusion sampler，以学习未归一化目标分布。它代表离散构造流与连续随机路径生成的一条早期交叉路线。 | **P1** |

### 1.3 训练、探索、泛化与效率：2024–2025

| 编号 | 论文与状态 | 简介 | 建议 |
|---|---|---|---|
| **T21** | [Learning Energy Decompositions for Partial Inference in GFlowNets](https://iclr.cc/virtual/2024/poster/18721) · ICLR 2024 | 学习把终点能量分解到中间状态的势函数，为部分轨迹提供局部信用。它特别适合 reward 可评价但天然分解未知的任务。 | **P1** |
| **T22** | [Pre-Training and Fine-Tuning Generative Flow Networks](https://iclr.cc/virtual/2024/poster/17406) · ICLR 2024 | 先进行 outcome-conditioned、弱依赖具体奖励的预训练，再对下游目标微调。论文探索 GFN 能否像通用生成模型一样复用表示和构造策略。 | **P1** |
| **T23** | [Local Search GFlowNets](https://iclr.cc/virtual/2024/poster/19387) · ICLR 2024 | 在已有样本附近执行回退和重构，将局部搜索轨迹用于 off-policy 训练，以更快进入高奖励区域。它改善发现效率，但评估时必须同时检查分布误差而非只看最佳奖励。 | **P1** |
| **T24** | [Embarrassingly Parallel GFlowNets](https://proceedings.mlr.press/v235/silva24a.html) · ICML 2024 | 并行训练多个具有不同经验或偏好的子 GFN，再组合其覆盖能力。论文聚焦多峰目标下的 mode coverage 和可扩展训练，是 ensemble/exploration 路线的重要基线。 | **P1** |
| **T25** | [Pessimistic Backward Policy for GFlowNets](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c1ab28d0fe0bfb53067a1af7e578cd7d-Abstract-Conference.html) · NeurIPS 2024 | 设计更“悲观”的后向策略来改变内部轨迹分配，使训练更集中于高奖励区域。它说明 \(P_B\) 不是无关紧要的辅助量，而会实质影响优化难度和探索—利用行为。 | **P1** |
| **T26** | [QGFN: Controllable Greediness with Action Values](https://proceedings.neurips.cc/paper_files/paper/2024/hash/948d8ba4e30c8c3a800cf436b31f376e-Abstract-Conference.html) · NeurIPS 2024 | 引入动作价值信息，在推断阶段连续调节采样的 greediness，而无需为每种温度重新训练模型。适合需要在“忠实分布采样”和“更高奖励搜索”之间动态切换的场景。 | **P1** |
| **T27** | [Streaming Bayes GFlowNets](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2fb57276bfbaf1b832d7bfcba36bb41c-Abstract-Conference.html) · NeurIPS 2024 | 研究新数据批次不断到达时如何增量更新离散后验 GFN，避免每次从头训练。它把 GFN 推向在线 Bayesian inference，并暴露旧后验、重加权和遗忘之间的权衡。 | **P1** |
| **T28** | [Learning to Scale Logits for Temperature-Conditional GFlowNets](https://proceedings.mlr.press/v235/kim24s.html) · ICML 2024 | 用单个条件模型学习跨温度的 logit 缩放，从而覆盖从多样采样到集中搜索的一族目标分布。论文适合研究 reward tempering 的校准和条件泛化。 | **P1** |
| **T29** | [On Generalization for GFlowNets](https://arxiv.org/abs/2407.03105) · 预印本 2024 | 从统计学习角度讨论仅见到部分状态或轨迹时，局部 balance 拟合能否推广到未见区域。其价值是提出泛化问题和初步理论视角；结论需结合后续正式论文与实证工作阅读。 | **P2** |
| **T30** | [Action Abstractions for Amortized Sampling](https://arxiv.org/abs/2410.15184) · ICLR 2025 | 通过宏动作或动作抽象缩短有效生成深度，减少长轨迹中的信用传播负担。它对层次化构造、规划和长序列 GFN 很有启发，但抽象动作是否保持支持覆盖是关键。 | **P1** |
| **T31** | [Optimizing Backward Policies in GFlowNets via Trajectory Likelihood Maximization](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f3efbcfe76bed022a37c5aeb1daf2326-Abstract-Conference.html) · ICLR 2025 | 通过轨迹似然最大化和 entropy-RL 视角显式优化 \(P_B\)，而不是把它固定为均匀分布。论文显示后向策略可用来选择更易学习的内部流，并改善前向采样。 | **P1** |
| **T32** | [When Do GFlowNets Learn the Right Distribution?](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a48f8928f78a58399ef0049453c14b02-Abstract-Conference.html) · ICLR 2025 Spotlight | 分析局部 balance 误差、函数表示限制与最终对象分布误差之间的关系，并提出比平均 loss 更可靠的 correctness 评价。它直接回答“loss 很低是否代表采样正确”，是 2025 年最重要的理论诊断论文之一。 | **P0 精读** |
| **T33** | [Generalization and Distributed Learning of GFlowNets](https://iclr.cc/virtual/2025/poster/29760) · ICLR 2025 | 给出数据依赖的泛化分析，并提出异步分布式 SAL 训练方法。它把“训练轨迹覆盖不足”和“多工作器扩展”放进同一研究框架。 | **P1** |
| **T34** | [Beyond Squared Error: Exploring Loss Design for Enhanced Training of Generative Flow Networks](https://proceedings.iclr.cc/paper_files/paper/2025/hash/353ec686503cd7020460d2829578ee4e-Abstract-Conference.html) · ICLR 2025 | 系统比较平方误差之外的回归损失，研究尾部惩罚和鲁棒性如何改变探索—利用及梯度行为。它提醒读者：同一个零点条件不意味着有限训练性质相同。 | **P1** |
| **T35** | [Towards Improving Exploration through Sibling Augmented GFlowNets](https://iclr.cc/virtual/2025/poster/30233) · ICLR 2025 | 用一个探索 sibling 与主 GFN 协作，将发现新模式和保持目标分布的职责部分解耦。它适合稀疏奖励环境，也为 2026 年 ACE 一类双网络方法提供背景。 | **P1** |
| **T36** | [Revisiting Non-Acyclic GFlowNets in Discrete Environments](https://proceedings.mlr.press/v267/morozov25a.html) · ICML 2025 | 在有限离散非无环环境中给出更简洁的存在性、唯一性和稳定性分析，并刻画固定 \(P_B\) 与 minimum flow 的关系。它把抽象非无环理论推进到可计算形式，是 O07/O08 的直接前置。 | **P0 精读** |
| **T37** | [Random Policy Evaluation Uncovers Policies of GFlowNets](https://proceedings.mlr.press/v267/he25a.html) · ICML 2025 | 把 GFlowNet flow 与普通随机策略的 policy evaluation 联系起来，提供分析策略结构的新工具。该视角有助于复用 RL 的值函数方法理解内部流。 | **P1** |
| **T38** | [Symmetry-Aware GFlowNets](https://proceedings.mlr.press/v267/kim25s.html) · ICML 2025 | 处理多个轨迹或表示对应同一对称对象时产生的系统采样偏差，并在训练中显式利用等价类结构。图、分子和组合对象中若忽略对称性，模型可能学到错误的对象边缘分布。 | **P1** |
| **T39** | [Ergodic Generative Flows](https://proceedings.mlr.press/v267/brunswic25a.html) · ICML 2025 | 以遍历变换和较弱的 flow-matching 条件推广 GFN，覆盖循环、连续变换及模仿学习等情形。它与非无环 GFN 相关，但采用不同的动力系统和遍历性视角。 | **P1** |
| **T40** | [Secrets of GFlowNets' Learning Behavior: A Theoretical Study](https://arxiv.org/abs/2505.02035) · 预印本 2025 | 分析 GFlowNet 的训练动力学和策略演化，试图解释模型为何会经历特定的模式发现与再分配过程。适合作为假设生成材料，关键结论仍应在可枚举任务上独立核验。 | **P2** |
| **T41** | [Hybrid-Balance GFlowNet for Solving Vehicle Routing Problems](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1898054a9392207250ad9cfde5286b2c-Abstract-Conference.html) · NeurIPS 2025 Spotlight | 面向车辆路径问题（VRP/CVRP/TSP），自适应结合 TB 的全局监督与 DB 的局部监督，并为含仓库节点的 CVRP 设计专门推断策略；以插件形式嵌入 AGFN、GFACS 等 GFlowNet 求解器。宜归入组合优化应用而非通用信用分配方法。 | **P1** |
| **T42** | [Adaptive Quantization in Generative Flow Networks for Probabilistic Sequential Prediction](https://proceedings.neurips.cc/paper_files/paper/2025/hash/dc63026c74191032dff373ed9d2d038a-Abstract-Conference.html) · NeurIPS 2025 | 为概率序列预测学习非均匀离散化，使 GFN 能把建模容量分配到更重要的数值区域。论文说明状态空间离散方式本身也是可学习的建模选择。 | **P1** |
| **T43** | [Flow Factorization for Efficient Generative Flow Networks](https://ojs.aaai.org/index.php/AAAI/article/view/34887) · AAAI 2025 | 将边流分解为状态流与动作分配项，以减少直接参数化所有边的成本。该结构化分解有利于大动作空间中的计算效率和可解释性。 | **P1** |
| **T44** | [Relative Trajectory Balance Is Equivalent to Trust-PCL](https://arxiv.org/abs/2509.01632) · NeurIPS 2025 FPI Workshop | 给出 relative TB 与 path-consistency RL 方法 Trust-PCL 的对应关系，进一步收紧 GFN 与最大熵 RL 的理论连接。阅读时应检查两边使用的路径分布、基准量和边界条件是否完全一致。 | **P1** |

### 1.4 2026 前沿方法

| 编号 | 论文与状态 | 简介 | 建议 |
|---|---|---|---|
| **T45** | [Boosted GFlowNets: Improving Exploration via Sequential Learning](https://arxiv.org/abs/2511.09677) · AISTATS 2026 | 用一系列 residual-reward GFN 逐步修补当前模型欠覆盖的区域，再组合成 boosted sampler。它把 boosting 思想引入模式覆盖，适合与 sibling/ACE 双网络探索方法比较。 | **P1** |
| **T46** | [Controlling Exploration-Exploitation in GFlowNets via Markov Chain Perspectives](https://arxiv.org/abs/2602.01749) · 预印本 2026 | 从 Markov-chain 视角引入可调参数 \(\alpha\)（方法名 \(\alpha\)-GFN），显式控制探索与利用强度。其吸引力在于统一调节行为，但需要进一步确认不同任务上的分布校准和理论保证。 | **P2** |
| **T47** | [Evaluating GFlowNet from Partial Episodes](https://iclr.cc/virtual/2026/poster/10007783) · ICLR 2026 | 提出 evaluation balance 和 partial-episode evaluator，在不完整轨迹上估计或诊断策略，并连接 value-based 与 policy-based GFN。它提供了训练 loss 之外的新评价接口。 | **P1** |
| **T48** | [Loss-Guided Auxiliary Agents for Overcoming Mode Collapse in GFlowNets](https://ojs.aaai.org/index.php/AAAI/article/view/39613) · AAAI 2026 | 训练辅助 agent 专门访问主模型高 loss、低覆盖的区域，再把经验回流给主 GFN。方法把“哪里最需要数据”直接绑定到局部训练误差。 | **P1** |
| **T49** | [\(f\)-Trajectory Balance](https://icml.cc/virtual/2026/poster/61247) · ICML 2026 | 建立 translation-invariant trajectory loss 与 \(f\)-divergence 的系统对应，使 TB loss 设计从经验试错转向 divergence 选择。它是 2026 年训练目标理论中最值得优先精读的工作。 | **P0 精读** |
| **T50** | [Avoid What You Know: Divergent Trajectory Balance for GFlowNets](https://icml.cc/virtual/2026/poster/62783) · ICML 2026 主会 | ACE 同时维护 canonical GFN 和 exploration GFN，让后者针对主模型欠覆盖的高奖励区域收集数据。它代表将“目标分布拟合”和“主动找盲区”解耦的最新探索路线；arXiv 元数据若仍写 under review，应以 ICML 官方主会页为准。 | **P1** |
| **T51** | [Stable GFlowNets with Probabilistic Guarantees](https://arxiv.org/abs/2605.01729) · 预印本 2026 | 尝试把可观测训练误差转化为全局分布距离或稳定性证书，直接回应 low-loss/high-TV 的风险。潜力很高，但使用其保证时必须核对有限状态、覆盖和误差界假设。 | **P1** |
| **T52** | [GFlowState: Visualizing the Training of Generative Flow Networks Beyond the Reward](https://arxiv.org/abs/2604.21830) · 预印本 2026 | 提出可视化和描述 GFlowNet 训练状态的诊断方法，避免只用平均奖励或单一 loss 判断模型。它更像分析工具，价值取决于指标能否预测真实分布误差。 | **P2** |
| **T53** | [Beyond the Proxy: Trajectory-Distilled Guidance for Offline GFlowNet Training](https://icml.cc/virtual/2026/poster/62632) · ICML 2026 | TD-GFN 用逆强化学习从离线优质轨迹中蒸馏引导信号，同时让最终 GFN 更新仍依赖真实终点奖励。它试图利用 proxy 或专家轨迹而不永久继承其偏差。 | **P1** |
| **T54** | [Rooted Absorbed Prefix Trajectory Balance with Submodular Replay for GFlowNet Training](https://icml.cc/virtual/2026/poster/65366) · ICML 2026 | RapTB 用 rooted/absorbed 前缀轨迹平衡为长序列提供前缀信用，并用 submodular replay 保持缓存多样性，针对长度偏置和稀疏奖励。它是 LLM/序列 GFN 当前很重要的训练组件。 | **P1** |
| **T55** | [Routing by Reaching: Composition of Pre-trained GFlowNets for Multi-Objective Generation](https://icml.cc/virtual/2026/poster/61403) · ICML 2026 | 研究如何无需从头重训就组合多个已训练 GFN，以满足新的多目标奖励。它把模型复用和组合泛化带入 GFlowNet，是“foundation sampler”方向的早期探索。 | **P1** |
| **T56** | [Spectral Flow Matching: Stabilizing Stochastic GFlowNets via Frequency-Domain Regularization](https://icml.cc/virtual/2026/poster/64150) · ICML 2026 | 用频域（谱）正则稳定随机 GFlowNet，提升噪声环境下的稳定性与稀疏奖励探索。阅读时应重点核对频谱约束如何改变目标分布和计算开销。 | **P1** |
| **T57** | [Proximal Policy Optimization for Amortized Discrete Sampling](https://arxiv.org/abs/2606.15793) · ICML 2026 SPIGM Workshop | 从最大熵 RL 连接出发，将 PPO 式 clipped policy update 用于摊销离散采样。它可能带来成熟 RL 优化工具，但 clipping 是否引入难以控制的分布偏差是核心问题。 | **P1** |

---

## 2. GFlowNet × Optimal Transport 论文

这一组把 OT 基础、神经 OT、非无环 minimum-flow GFN 和本调研重点关注的 `arXiv:2606.06272` 放在同一条连续路线中。

| 编号 | 论文与状态 | 简介 | 建议 |
|---|---|---|---|
| **O01** | [Optimal Transport for Machine Learners](https://arxiv.org/abs/2505.06589) · 课程讲义/预印本 2025 | 面向机器学习读者系统讲解 Monge、Kantorovich、对偶、动态 OT、Wasserstein 几何和梯度流。它不是 GFlowNet 论文，但能快速补齐理解 coupling、cost 和正则化所需的 OT 语言。 | **P0 先修** |
| **O02** | [A Framework for Wasserstein-1-Type Metrics](https://arxiv.org/abs/1701.01945) · J. Convex Anal. 2019 | 给出 Wasserstein-1 型距离的统一框架，将其推广到不同质量的非负测度之间（非平衡 OT），保持凸性与可计算性并涵盖多种已有度量。可作 O08 的 OT 前置，重点体会运输成本、（不）平衡质量与最小费用流/W1 对偶之间的关系。 | **P1** |
| **O03** | [GeONet: A Neural Operator for Learning the Wasserstein Geodesic](https://proceedings.mlr.press/v244/gracyk24a.html) · UAI 2024 | 用 neural operator 学习分布对之间的 Wasserstein geodesic，目标是跨任务摊销求解动态 OT。它与 GFN–OT 的互补点是连续分布和 geodesic，而非离散图上的构造路径。 | **P1** |
| **O04** | [Schrödinger Bridge Flow for Unpaired Data Translation](https://arxiv.org/abs/2409.09347) · NeurIPS 2024 Spotlight | 研究带熵正则的动态 transport 和随机路径桥接，把起点分布逐步推向终点分布。它可用于比较“路径熵正则”与 GFlowNet 中“多条构造路径分配”的相同点和不同点。 | **P1** |
| **O05** | [Universal Neural Optimal Transport](https://proceedings.mlr.press/v267/geuter25a.html) · ICML 2025 | 学习可条件化、可摊销的 neural OT map 或 plan，希望一个模型处理一族源—目标分布。它是评估 conditional GFlowNet–OT 是否真正具有跨任务泛化优势的重要强基线。 | **P1** |
| **O06** | [Computing High-Dimensional Optimal Transport by Flow Neural Networks](https://proceedings.mlr.press/v258/xu25f.html) · AISTATS 2025 | 用神经流处理连续高维 OT，重点解决传统离散 coupling 在维度和样本数上的扩展困难。注意这里的 flow neural network 不是 GFlowNet，但可作为连续扩展的相邻方法。 | **P1** |
| **O07** | [Learning Shortest Paths with Generative Flow Networks](https://arxiv.org/abs/2603.01786) · ICML 2026 SPIGM Workshop | 研究在满足目标终止分布的多个可行内部流中，minimum-flow 准则为何偏向最短生成路径。它把“采样正确”之外的内部流选择变成优化问题，是从非无环 GFN 走向图 OT 的直接桥梁。 | **P0 精读** |
| **O08** | [Your GFlowNet Secretly Learns an Optimal Transport Plan](https://arxiv.org/abs/2606.06272) · **ICML 2026 SPIGM Workshop；非主会** | 论文考虑非无环图、给定源分布与奖励诱导目标分布，并在固定初始流边缘下最小化总流量；在图最短路诱导的 transport cost 下，最优 GFlowNet 边流编码一个 Kantorovich 最优 coupling。它的重要性在于为终点分布相同但内部流不唯一的问题提供 OT 选择原则；它的边界也很明确：结论不是“任意 DAG 上的标准 TB 自动求任意 OT”，而依赖 minimum-flow、图 cost、吸收性和边缘约束等条件。 | **P0 特别精读** |

### 2.1 对 O08 的精读问题

读这篇论文时，建议逐一回答：

1. 源分布怎样进入初始流约束，目标分布怎样由 terminal reward 归一化得到？
2. minimum-flow 的目标究竟是总 expected visits、总 edge flow，还是带权路径长度？
3. 从边流恢复 coupling 时，质量沿多条最短路分裂是否影响 coupling 的唯一性？
4. 定理保证的是最优 transport cost、某个 transport plan，还是 plan 的唯一恢复？
5. 神经参数化、balance 残差和有限采样误差怎样传递到边缘违反与 OT cost gap？
6. 算法与 network simplex、min-cost flow、Sinkhorn、neural OT 的时间和内存比较是否公平？

本地文件：

- [O08 PDF：2606.06272](literature/core/2606.06272.pdf)
- [Foundations PDF：2111.09266](literature/core/2111.09266.pdf)
- [Shen 2023 PDF：2305.07170](literature/core/2305.07170.pdf)

---

## 3. 应用论文

应用论文主要用来学习状态、动作、终止条件、reward 和约束如何设计。不能只按最高或平均 reward 判断是否学到了正确分布。

### 3.1 科学发现、分子、蛋白与材料

| 编号 | 论文与状态 | 简介 | 建议 |
|---|---|---|---|
| **A01** | [GFlowNets for AI-Driven Scientific Discovery](https://pubs.rsc.org/en/content/articlelanding/2023/dd/d3dd00002h) · Digital Discovery 2023 | 系统回顾 GFlowNet 在分子、蛋白、材料、因果发现和主动学习中的作用，并解释为什么科学发现往往需要一组多样候选而非单个最优解。适合先建立应用版图，再按具体领域下钻。 | **P0 综述** |
| **A02** | [Biological Sequence Design with GFlowNets](https://proceedings.mlr.press/v162/jain22a.html) · ICML 2022 | 将序列逐步构造为 GFN 环境，并与主动学习结合，在昂贵 oracle 下寻找多样高适应度生物序列。它是 reward proxy、批量候选和 mode diversity 的经典案例。 | **P1** |
| **A03** | [Multi-Objective GFlowNets](https://proceedings.mlr.press/v202/jain23a.html) · ICML 2023 | 用偏好向量条件化 GFN，在一次训练后为不同 Pareto 权衡采样多样解。论文说明条件 reward 和温度如何表示用户偏好，适合多指标分子与工程设计。 | **P1** |
| **A04** | [Multi-Fidelity Active Learning with GFlowNets](https://arxiv.org/abs/2306.11715) · TMLR 2024 | 联合选择候选对象与评估保真度，在低成本近似和高成本真实实验之间分配预算。它把 GFN 从候选生成扩展到实验决策策略。 | **P1** |
| **A05** | [Towards Equilibrium Molecular Conformation Generation with GFlowNets](https://pubs.rsc.org/en/content/articlelanding/2024/dd/d4dd00023d) · Digital Discovery 2024, 3(5):1038–1047 | 以分子构象能量定义近似 Boltzmann 目标，探索 GFN 对连续/几何构象分布的建模。重点关注坐标参数化、对称性和能量 oracle 误差。 | **P1** |
| **A06** | [Crystal-GFN](https://arxiv.org/abs/2310.04925) · 预印本 2023 | 将晶体结构按组成、空间群和几何参数逐步生成，并纳入有效性及材料性质约束。它展示了复杂结构先验如何进入动作空间和 mask。 | **P1** |
| **A07** | [PhyloGFN](https://iclr.cc/virtual/2024/poster/18107) · ICLR 2024 | 把系统发育树的逐步合并过程建模为 DAG，在给定序列数据下摊销采样树后验。该任务有明确 Bayesian 目标，是检验 GFN 多路径和结构后验能力的代表案例。 | **P1** |
| **A08** | [Genetic-guided GFlowNets](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4b25c000967af9036fb9b207b198a626-Abstract-Conference.html) · NeurIPS 2024 | 将遗传算法产生的结构化搜索经验蒸馏到 off-policy GFN 中，以加速高奖励模式发现。论文代表经典启发式搜索与分布学习的混合路线。 | **P1** |
| **A09** | [RGFN: Synthesizable Molecular Generation Using GFlowNets](https://proceedings.neurips.cc/paper_files/paper/2024/hash/53704142f230054140418ecd8857f391-Abstract-Conference.html) · NeurIPS 2024 | 直接在化学反应和可用构件空间中生成分子，使样本天然附带合成路径。它把可合成性从后验筛选转为生成过程约束。 | **P1** |
| **A10** | [GFlowNet Assisted Biological Sequence Editing](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c14760740573001c0d18d58879a6a305-Abstract-Conference.html) · NeurIPS 2024 | 从给定 seed sequence 出发进行少量但多样的编辑，而不是完全从头生成。适合研究局部动作、编辑距离约束和保持原序列先验。 | **P1** |
| **A11** | [SynFlowNet: Design of Diverse and Novel Molecules with Synthesis Constraints](https://iclr.cc/virtual/2025/poster/27946) · ICLR 2025 | 以化学反应和可购买 reactant 为动作，保证生成分子可由显式路径合成。相较原子级生成，它更强调现实可执行性与反应空间覆盖。 | **P1** |
| **A12** | [Pretraining Generative Flow Networks with Inexpensive Rewards for Molecular Graph Generation](https://proceedings.mlr.press/v267/pandey25b.html) · ICML 2025 | 先用廉价 proxy reward 预训练原子级分子 GFN，再适配昂贵性质目标。它研究 proxy 预训练何时提高样本效率、何时会把偏差带入下游。 | **P1** |
| **A13** | [Synergy of GFlowNet and Protein Language Model Makes a Diverse Antibody Designer](https://ojs.aaai.org/index.php/AAAI/article/view/34370) · AAAI 2025 | 将蛋白语言模型先验与效力、可开发性等多项 reward 组合，用 GFN 生成多样抗体序列。它是 product-of-experts reward 和生物先验融合的代表应用。 | **P1** |
| **A14** | [LeakGFN](https://icml.cc/virtual/2026/poster/63060) · ICML 2026 | 区分真正有效分子流和因截断、无效动作产生的“泄漏”终止流，减少化学任务中的目标分布污染。该方法直面现实环境中大量无效 terminal 的建模问题。 | **P1** |
| **A15** | [Synthesizable Molecular Generation via Soft-constrained GFlowNets with Rich Chemical Priors (S3-GFN)](https://icml.cc/virtual/2026/poster/64424) · ICML 2026 | 结合软约束和大规模 SMILES 先验，提高生成分子的有效性与可合成率，同时保留 GFN 的多样性目标。它代表将 foundation chemistry prior 注入 GFN 的最新路线。 | **P1** |
| **A16** | [A Distributional Framework for Generative Modeling of Molecular Crystals](https://arxiv.org/abs/2607.05266) · 预印本 2026 | 面向分子晶体建立分布式而非单点优化的生成框架，以覆盖多个稳定候选结构。论文很新，重点核查晶体对称性、能量模型和分布评价是否充分。 | **P2** |

### 3.2 结构学习与组合优化

| 编号 | 论文与状态 | 简介 | 建议 |
|---|---|---|---|
| **A17** | [Bayesian Structure Learning with GFlowNets](https://proceedings.mlr.press/v180/deleu22a.html) · UAI 2022 | 将有向无环图逐边构造，并按 Bayesian score 近似采样因果结构后验。它是 GFN 处理组合对象后验、多条构造路径与配分函数的经典应用。 | **P1** |
| **A18** | [Joint Bayesian Inference of Graphical Structure and Parameters with a Single GFlowNet](https://neurips.cc/virtual/2023/poster/70228) · NeurIPS 2023 | 在单一 GFN 中联合表示图结构及其连续或离散参数后验，避免结构学习后再单独拟合参数。论文展示混合状态和分层生成在 Bayesian inference 中的价值。 | **P1** |
| **A19** | [Let the Flows Tell: Solving Graph Combinatorial Problems with GFlowNets](https://proceedings.neurips.cc/paper_files/paper/2023/hash/27571b74d6cd650b8eb6cf1837953ae8-Abstract.html) · NeurIPS 2023 | 把若干图组合优化问题写成逐步构造过程，并用 GFN 生成多样近优解。阅读时应区分优化指标提升与 reward-proportional 分布拟合。 | **P1** |
| **A20** | [Robust Scheduling with GFlowNets](https://arxiv.org/abs/2302.05446) · ICLR 2023 | 用 GFN 生成一组对不确定扰动具有不同权衡的调度方案，从而提高决策鲁棒性。它说明 GFN 在需要备选方案而非唯一计划的运筹任务中的价值。 | **P2** |

### 3.3 LLM、推理、视觉与 diffusion

| 编号 | 论文与状态 | 简介 | 建议 |
|---|---|---|---|
| **A21** | [Amortizing Intractable Inference in Large Language Models](https://proceedings.iclr.cc/paper_files/paper/2024/hash/bc667ac84ef58f2b5022da97a465cbab-Abstract-Conference.html) · ICLR 2024 | 用 GFlowNet 微调语言模型，使其按后验权重采样潜在推理或文本变量，而不是依赖逐查询 MCMC。它是“LLM 作为摊销离散推断器”的关键早期论文。 | **P1** |
| **A22** | [Amortizing Intractable Inference in Diffusion Models for Vision, Language, and Control](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8b21a7ea42cbcd1c29a7a88c444cce45-Abstract-Conference.html) · NeurIPS 2024 | 用 relative TB 学习 diffusion 模型中的难后验，覆盖视觉、语言和控制任务。论文展示 GFN 可以学习随机生成过程的条件路径分布，而不仅是离散对象。 | **P1** |
| **A23** | [Efficient Diversity-Preserving Diffusion Alignment via Gradient-Informed GFlowNets (Nabla-GFlowNet)](https://iclr.cc/virtual/2025/poster/30600) · ICLR 2025 | 利用可微 reward 的梯度引导 diffusion/GFN alignment，在提升目标性质时尽量保持预训练生成先验和输出多样性。它适合与仅使用标量 reward 的方法比较样本效率。 | **P1** |
| **A24** | [COFlowNet](https://iclr.cc/virtual/2025/poster/28047) · ICLR 2025 | 在没有在线 oracle、只能依赖离线数据时约束流进入数据支持之外的区域，以减少虚假的高奖励外推。论文直面 offline GFN 的 support mismatch。 | **P1** |
| **A25** | [Flow of Reasoning: Training LLMs for Divergent Reasoning with Minimal Examples](https://proceedings.mlr.press/v267/yu25k.html) · ICML 2025 | 用少量可验证样例训练语言模型生成多条高质量推理路径，强调 divergent reasoning 而非单一路径模仿。评估时应同时看正确率、路径多样性和 reward hacking。 | **P1** |
| **A26** | [EraseFlow](https://proceedings.neurips.cc/paper_files/paper/2025/hash/66c9de41210338c9581d5313125b7486-Abstract-Conference.html) · NeurIPS 2025 | 用 GFN 探索 diffusion 模型中多样的去概念/擦除轨迹，以避免单一路径造成能力损伤。它把多样轨迹搜索用于模型编辑与安全。 | **P1** |
| **A27** | [Discovering Latent Graphs with GFlowNets](https://proceedings.neurips.cc/paper_files/paper/2025/hash/96c6f409a374b5c81d2efa4bc5526f27-Abstract-Conference.html) · NeurIPS 2025 | 学习可解释的潜在图结构，再条件化图像生成，以覆盖多种关系配置。GFlowNet 在这里负责对离散潜图进行多样后验式搜索。 | **P1** |
| **A28** | [GFlowVLM: Enhancing Multi-step Reasoning in Vision-Language Models with Generative Flow Networks](https://openaccess.thecvf.com/content/CVPR2025/html/Kang_GFlowVLM_Enhancing_Multi-step_Reasoning_in_Vision-Language_Models_with_Generative_Flow_CVPR_2025_paper.html) · CVPR 2025 | 将视觉语言模型的多步推理轨迹作为 GFN 构造过程，训练模型覆盖多条高回报 reasoning/planning 路径。论文报告具身规划和 OOD 泛化，是视觉主会中最直接的 GFlowNet 核心应用之一。 | **P1** |
| **A29** | [Learning to Sample Effective and Diverse Prompts for Text-to-Image Generation](https://openaccess.thecvf.com/content/CVPR2025/html/Yun_Learning_to_Sample_Effective_and_Diverse_Prompts_for_Text-to-Image_Generation_CVPR_2025_paper.html) · CVPR 2025 | 用 GFlowNet 学习一族有效且多样的文本提示，并通过 reward decomposition 缓解 prompt optimizer 的 plasticity loss。它展示 GFN 如何在冻结的文生图模型外层做离散 prompt 搜索。 | **P1** |
| **A30** | [Flow of Spans](https://iclr.cc/virtual/2026/poster/10007998) · ICLR 2026 | 使用动态 span vocabulary，使同一文本可通过不同粒度的 token/span 路径构造，从树式自回归生成提升为 DAG。它利用多路径结构改善语言生成的信用分配和效率。 | **P1** |
| **A31** | [AlphaSAGE](https://iclr.cc/virtual/2026/poster/10006456) · ICLR 2026 | 将 GFN 用于结构化量化 alpha 因子挖掘，目标是得到高质量且彼此低相关的一组策略。它是金融组合发现中“多样解集”价值的代表案例。 | **P2** |
| **A32** | [Stable-GFlowNet for LLM Red-Teaming](https://icml.cc/virtual/2026/poster/64302) · ICML 2026 Spotlight | 用 pairwise contrastive TB 和噪声 reward masking 训练稳定、多样的红队提示生成器，并避免显式估计 \(Z\)。它针对安全 reward 高噪、模式稀疏和提示多样性三个实际问题。 | **P1** |
| **A33** | [GFlowRL](https://arxiv.org/abs/2607.13394) · 预印本 2026 | 将 distribution-matching RL 扩展到 dense 与 MoE 语言模型，希望让策略按奖励诱导分布覆盖多种高质量响应。论文非常新，需要重点检查它与经典 GFlowNet 的目标、支持和归一化是否完全一致。 | **P1/P2** |
| **A34** | [PowerFlow: Unlocking the Dual Nature of LLMs via Principled Distribution Matching](https://arxiv.org/abs/2603.18363) · ICML 2026 | 研究 LLM 分布匹配训练中的能力—多样性权衡，尝试用 GFlowNet 式目标控制响应覆盖。建议把它与 PPO、DPO 和 GFlowRL 放在同一评价协议下比较。 | **P2** |
| **A35** | [GFlowPO](https://arxiv.org/abs/2602.03358) · 预印本 2026 | 用 GFlowNet 对离散 prompt 或策略候选进行分布式优化，强调保留多个有效解而非收敛到单一提示。它适合作为 prompt optimization 应用线索，尚需正式接收状态和更强基线验证。 | **P2** |

---

## 4. GFlowNet × OT 是否很有潜力

结论：**有潜力，但当前更像“结构性理论突破 + 早期算法原型”，还不是成熟的通用 OT 求解器。**

> 完整版逐节论证（定理重述与证明思路、真正新颖点、四大不足、8 个后续方向潜力评级、决定性实验协议）见 [GFlowNet × OT 方向潜力深度分析](GFLOWNET_OT_POTENTIAL_ANALYSIS_CN.md)。

| 判断维度 | 为什么看好 | 主要限制 |
|---|---|---|
| **理论连接** | O08 给出可证明的 coupling/最优 transport 关系，不只是因为两者都叫“flow” | 依赖非无环图、minimum-flow、固定源边缘和图最短路 cost |
| **摊销能力** | 条件 GFN 有机会一次学习一族 source–target transport problems | 尚缺跨图、跨边缘分布的强泛化证据 |
| **大图扩展** | 用神经网络表示边流，可能避开显式存储巨大 coupling 矩阵 | balance 训练和探索可能仍比经典稀疏 min-cost-flow 慢 |
| **内部流选择** | OT 为“相同终点分布对应多个内部流”提供成本最小的规范化原则 | 最短/最低成本内部流可能降低路径多样性和探索能力 |
| **连续与高维** | 可与 continuous GFN、neural OT、Schrödinger bridge 连接 | 测度、Jacobian、吸收和误差传播理论尚不成熟 |

### 最值得做的四个研究课题

下表按**可执行性**口径选题（要求有最小可行实验与明确报告指标）；OT 分析 §5 是按**方向潜力**口径给 8 个方向评级，两表口径不同、不必一一对应。"撞车风险"来自 2026-08 的独立复核。

| 课题 | 撞车风险 | 核心假设 | 最小可行实验 | 必须报告 |
|---|---|---|---|---|
| **Balance 残差到 OT 误差的界** | **低——最推荐** | 局部 flow 误差可控制 coupling 和 cost 误差 | 可枚举小图，主动构造低 loss/高 cost-gap 反例 | TV、边缘误差、cost gap、界的松紧度 |
| **条件 GFN 学习一族图 OT** | **高** | 重复查询时，摊销成本可超过一次性求解器 | 网格图和随机稀疏图；训练多组 source/target，测试未见边缘 | marginal violation、OT cost gap、wall-clock、显存 |
| **熵正则 GFlowNet–OT / SB-GFN** | **高** | 路径熵与 coupling entropy 可形成可控的双层正则 | 比较 min-flow、Sinkhorn、path-entropy 和两者联合 | plan entropy、path entropy、cost、mode coverage |
| **GFN proposal + classical OT correction** | 中（更像工程组合） | GFN 负责摊销发现稀疏支持，经典 solver 做精确修正 | 先预测候选 transport edges，再在子图上跑 min-cost flow | support recall、最终精度、总时间、失败案例 |

**撞车依据与差异化建议**

- 课题「条件 GFN 学一族图 OT」：[ULOT（NeurIPS 2025）](https://papers.nips.cc/paper_files/paper/2025/file/873fd89b3e4db1f6242c2333673e104d-Paper-Conference.pdf) 已实现图上、条件化、含 unbalanced 的摊销 OT plan 预测（比经典 solver 快约两个数量级，还能给 solver 做 warm-start），另有 CONDOT、CVFM 与已收录的 O05。卖点大量被占。
- 课题「熵正则 / SB-GFN」：离散与图上 Schrödinger 桥生态已拥挤（[DDSBM](https://arxiv.org/abs/2410.01500)、[GSBoG](https://arxiv.org/abs/2602.04675)、[MadSBM](https://arxiv.org/abs/2601.22408)、[离散 SB/EOT benchmark](https://arxiv.org/abs/2509.23348)，均见 §11.8）。这些方法多为 CTMC + iterative Markovian fitting、并非 GFlowNet，但占同一生态位；其中 GSBoG 与 O08 的设定最接近。
- 课题 4 新颖性偏弱（ULOT 已演示"神经预测 → warm-start 经典 solver"范式），建议**降为课题 1/2 的对照基线**而非独立课题。
- 若坚持做课题 1 或 3，**必须锁死差异化定位**：隐式、无法显式构造 cost matrix 的巨大组合图 + 只能走合法局部动作——这是 GNN / CTMC / 连续 neural-OT 都不覆盖的唯一护城河。
- 另可考虑 OT 分析 §5 判为低撞车的 **primal-dual GFlowNet OT**（Kantorovich potential 作 critic，可直接给出 primal-dual gap 证书）。

短期最有希望的定位不是替换所有 Sinkhorn 或 network simplex，而是：

> **面向巨大结构化图、重复 source–target 查询、条件化分布和昂贵边代价的摊销 OT。**

---

## 5. 2024–2026 顶会覆盖审计

论文已经在主目录中给出简介；这里仅用编号审计 venue 覆盖，不再重复论文卡片。

| Venue | 已核验的主会覆盖 | 说明 |
|---|---|---|
| **NeurIPS 2024–2025** | T17、T25–T27、T41–T42；A08–A10、A22、A26–A27 | NeurIPS 2026 截止 2026-07-31 尚未举办，不能提前计入 |
| **ICML 2024–2026** | T16、T24、T28、T36–T39、T49–T50、T53–T56；A12、A14–A15、A25、A32、A34 | 2026 是 loss、探索、长序列与组合方法最密集的一批；A34 经复核为 ICML 2026 主会（原标预印本） |
| **ICLR 2024–2026** | T18、T20–T23、T30–T35、T47；A07、A11、A21、A23–A24、A30–A31 | 2025 特别集中在 correctness、loss、\(P_B\) 与探索；T20（ICLR 2024）、T30（ICLR 2025）经复核由预印本更正为主会 |
| **AAAI 2024–2026** | T19、T43、T48；A13 | 形成非无环理论、flow factorization、loss-guided exploration 方法线 |
| **CVPR 2025** | A28、A29 | 两篇都以 GFlowNet 为核心；2024、2026 官方主会列表未找到同等直接工作 |
| **ICCV 2025** | 无 | 官方 proceedings 未检索到以 GFlowNet 为核心方法的主会论文 |
| **ECCV 2024/2026** | 无可确认项 | ECCV 2024 未检索到核心工作；ECCV 2026 截止核验日尚无最终 proceedings |

审计口径：

- 只纳入 GFlowNet 是核心模型、训练目标或主要分析对象的论文；
- 不把 normalizing flow、flow matching、rectified flow、optical flow 或名字中偶然出现 “FlowNet” 的工作算入；
- Workshop 不计作主会，因此 **O08 不能写成 ICML 2026 主会论文**；
- 会议官网与 arXiv 元数据冲突时，以正式 proceedings 或主会日程为准。

### 5.1 补录的主会覆盖（N 系列，2026-08 查全复核）

独立复核确认原 §5 属**零误报但查全不足**：只按标题含 “GFlowNet” 检索，会整批跳过标题不含该词、但以 GFlowNet 为核心的论文（Delta-AI、Learning Diverse Attacks、Adaptive Teachers 等即属此类）。下表把第 11 节中符合本节口径的主会论文按 venue 归位，与上表合并才是完整覆盖。

| Venue | 补录的主会编号 | 说明 |
|---|---|---|
| **NeurIPS 2024** | N010 | 扩散采样器统一基准，与 T20 同谱系 |
| **NeurIPS 2025** | N101、N104 | 可合成分子生成规模化；扩散采样器可扩展训练 |
| **ICML 2024** | N062 | 对称回放训练（组合优化样本效率） |
| **ICML 2025** | N057、N106 | 3D 分子与合成路径共设计；生物序列 off-policy 稳健训练 |
| **ICML 2026** | N082（**Oral**）、N097、N105（Spotlight） | 路径依赖摊销推断（挑战 Markov flow 前提）；保守 GFN 做 LLM 推荐；SMC × 摊销采样 |
| **ICLR 2024** | N008、N009 | Order-Preserving GFlowNets；Delta-AI 稀疏图模型局部目标 |
| **ICLR 2025** | N014、N024、N056、N063、N102 | 目标条件后向合成；自适应 teacher；RxnFlow；红队多样攻击（奠基）；对抗 GFN 解 VRP |
| **ICLR 2026** | N019、N103 | FlowRL（TB ⇔ 反向 KL，胜 GRPO/PPO）；通用 soft operator 统一离散组合生成 |
| **UAI 2026** | N083 | Particle GFlowNets——补上原表的 UAI 空缺 |
| **AISTATS 2024–2025** | N003、N078 | 最大熵 GFN × soft Q-learning；蚁群采样 GFACS |
| **EMNLP 2024** | N020 | GDPO 多样性对齐 |

> 口径未变：workshop 一律不计主会（O08、O07、T57、T44 等仍按 Workshop 标注）；仅 GFlowNet 为核心模型/训练目标/主要分析对象的论文计入——第 11 节中的神经 OT、Schrödinger 桥、扩散采样器等**对照组**论文即便发表于主会也不计入本表。

---

## 6. 课程、教程与博客

这些不是论文，因此单独收纳，避免与正式研究成果混在一起。

### 6.1 系统课程

| 优先级 | 资源 | 简介 |
|---|---|---|
| **P0** | [Probabilistic Inference with GFlowNets — IFT 6760B A25](https://alexhernandezgarcia.com/teaching/gflownets25/) · [Slides](https://alexhernandezgarcia.com/teaching/gflownets25/slides) · [Bibliography](https://alexhernandezgarcia.com/teaching/gflownets25/bibliography) | 2025 秋季研究生课程，覆盖基础、loss、连续 GFN、训练评估、VI/RL、条件/多目标 GFN 和科学发现；最适合作为系统学习主线。 |
| **P0** | [Mila GFlowNet Workshop 2023](https://www.gflownet.org/) · [日程、讲座和 Colab](https://www.gflownet.org/schedule) | 三天集中课程，从概率推断先修到理论、训练、连续空间、应用和开放问题；时间有限可优先看理论与训练日。 |
| **P1** | [GFlowNets: Introduction and Applications to AI-Driven Scientific Discovery](https://alexhernandezgarcia.github.io/slides/mmiccs-mar23) | 一套结构完整的入门与科学发现幻灯片，适合快速复习或准备组会报告。 |
| **P2** | [GFlowNets and System 2 Deep Learning](https://www.microsoft.com/en-us/research/video/gflownets-and-system-2-deep-learning/) | 从研究愿景解释主动学习、推理和 System 2 连接；适合理解动机，不应作为定理来源。 |

### 6.2 教程、博客与交互材料

| 优先级 | 资源 | 简介 |
|---|---|---|
| **P0** | [The GFlowNet Tutorial](https://milayb.notion.site/The-GFlowNet-Tutorial-95434ef0e2d94c24aab90e69b30be9b3) | 作者团队维护的高层教程，从水流直觉逐步进入 balance objectives 和多种扩展，是最好的第一入口。 |
| **P0** | [Emmanuel Bengio: Introduction to GFlowNets](https://folinoid.com/w/gflownet/) | 用水流、合流路径和分子设计解释 reward-proportional sampling，适合第一次接触。 |
| **P0** | [Sungsoo Ahn: Generative Flow Networks](https://sungsoo-ahn.github.io/blog/2026/generative-flow-networks/) | 2026 更新的进阶长文，从概率机器学习角度串联 TB、MaxEnt RL、VI 和 diffusion。 |
| **P1** | [GFlowNets and Amortized Marginalization](https://milayb.notion.site/01755ca312834e15ab0ae9ef46bcb1bb) | 解释 GFN 为什么既能摊销采样，也能估计难求和的边缘量。 |
| **P1** | [Mila: What Do GFlowNets and Variational Inference Have in Common?](https://mila.quebec/en/article/what-do-gflownets-and-variational-inference-have-in-common) | 在阅读 T08/T09 前建立轨迹空间 VI 的直觉。 |
| **P0** | [GFlowNet Playground](https://gfn-playground.caleydoapp.org/) | 交互观察构造路径、流守恒、奖励比例采样和模式覆盖；适合直觉，不替代数值实验。 |

### 6.3 先修材料

| 主题 | 资源 | 最低掌握要求 |
|---|---|---|
| 概率与信息论 | [Deep Learning Book，Part I](https://www.deeplearningbook.org/) | log probability、KL/TV、未归一化密度、数值稳定 |
| 概率图模型 | [Stanford CS228](https://cs.stanford.edu/~ermon/cs228/index.html) | Bayesian network、采样、VI、结构学习 |
| 深度生成模型 | [Stanford CS236](https://deepgenerativemodels.github.io/syllabus.html) | 自回归、VAE、normalizing flow、EBM |
| 强化学习 | [David Silver / DeepMind RL Course](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PL7-jPKtc4r78-wCZcQn5IqyuWhBZ8fOxT) | MDP、TD、policy gradient；另补 maximum-entropy RL |
| 最优传输 | [Computational Optimal Transport](https://optimaltransport.github.io/book/) | 离散 Kantorovich、dual、Sinkhorn、Wasserstein |
| OT 实践 | [Python Optimal Transport](https://pythonot.github.io/) | 精确离散 OT、Sinkhorn、barycenter 和基线实现 |

---

## 7. 代码与复现

| 优先级 | 资源 | 适合任务 | 注意 |
|---|---|---|---|
| **P0** | [torchgfn](https://github.com/GFNOrg/torchgfn) · [文档](https://gfn.readthedocs.io/en/latest/) · [PyPI](https://pypi.org/project/torchgfn/) | HyperGrid、实现新 loss、教学复现 | 环境、sampler、module 和 loss 解耦，适合第一份实验。**注意版本**：截至 2026-04 为 v2.4.1，v2 相对初版是大重构（模块化 estimator/sampler/loss、新增 Chip Design 环境、custom log_rewards、更难的 HyperGrid），`pip install torchgfn` 装到的最新版与早期 tutorial/notebook API 不一致——跟教程时请锁定版本。库论文见 N099（注意是 Workshop/预印本，非 JMLR） |
| **P1** | [Recursion/Valence GFlowNet](https://github.com/recursionpharma/gflownet) · [文档](https://gflownet.readthedocs.io/en/latest/) | 图、分子、多目标及 online/offline 混合训练 | 依赖较重，进入真实科学任务后再用 |
| **P1** | [gfnx](https://github.com/d-tiapkin/gfnx) · [文档](https://gfnx.readthedocs.io/en/latest/) | JAX、可扩展 benchmark、单文件基线 | 可扩展 **benchmark 套件**（不只是基线集合），适合性能实验和大批量比较 |
| **P1** | [gfn-diffusion](https://github.com/GFNOrg/gfn-diffusion) | 连续 GFN / diffusion sampler 训练与统一评测 | 配套 N010（NeurIPS 2024 统一基准），做连续采样器选型时的对照实现；原清单缺此条 |
| **P0** | [Mila Workshop notebooks](https://github.com/josephdviviano/gflownet-tutorials) | Colab 实操和基础习题 | 与 Workshop 课程配套（仓库已改名，旧名 `torchgfn-tutorials` 靠 GitHub 转发） |
| **P1** | [Awesome-GFlowNets](https://github.com/zdhNarsil/Awesome-GFlowNets) | 继续发现论文与代码 | 社区索引；发表状态必须回原始来源核验。**已停更于 2024-10-01**（2026-09-01 核实），不可作为 2025-2026 的文献来源 |
| **P2** | [ARIS](https://github.com/wanshuiyin/auto-claude-code-research-in-sleep) | 周期性检索、下载、去重和生成候选报告 | 适合自动更新清单，不代替人工审阅定理与会议状态 |

#### 代码资源勘误（2026-09-01 经 GitHub API 核实，详见 `insights/trends_applications.md`）

- `milaforscience/gflownet` 是 0 star / 0 提交的**空镜像**，勿引用；面向科学发现的 Mila 库正确地址是 [alexhernandezgarcia/gflownet](https://github.com/alexhernandezgarcia/gflownet)（344 star，最近提交 2026-08-29，Crystal-GFN 出处）。
- [GFNOrg/gflownet](https://github.com/GFNOrg/gflownet)（686 star）停更于 2023-02-28，是 2021 原始实现的**历史归档**，不是活跃实现。
- [recursionpharma/gflownet](https://github.com/recursionpharma/gflownet) 近一年仅 2 次提交且均为 Dockerfile/tox 杂务，处于**停滞**状态；上表 P1 评级仅针对其代码完备度，不代表当前维护活跃度。
- `torchgfn` 的 GitHub license 字段为 `NOASSERTION`，商用集成前需人工确认授权。
- `gfnx` 的 PyPI 只有 0.0.1（2025-11-16 上传后未发新版），当前是「作者自用 + 论文配套」，尚未形成社区，不建议作为唯一依赖。

### 第一份复现实验

1. 选择可以精确枚举 \(P^\star(x)=R(x)/Z\) 的有限 HyperGrid。
2. 比较 FM、DB、TB、SubTB，固定网络容量和采样预算。
3. 同时报告：
   - 终止分布 L1/TV 或 JS；
   - mode coverage；
   - reward correlation 与平均 reward；
   - 轨迹长度；
   - loss 的 median、tail quantile 和 spike；
   - wall-clock 和采样数。
4. 加入 off-policy/replay 后，单独检查支持覆盖和数据分布偏移。
5. 只有 toy task 的分布误差可信后，再迁移到分子、LLM 或 OT。

---

## 8. 论文精读模板

```markdown
### 编号 · 论文标题

- 状态：会议/期刊/Workshop/预印本；版本日期：
- 一句话问题：它解决哪个具体失败模式？
- 空间：DAG / 非无环 / 连续 / 随机；终止条件：
- 目标分布：明确写出 p*(x)：
- 参数：PF、PB、F、Z 中哪些固定、哪些学习？
- 数据：on-policy / off-policy / replay / offline？
- 核心 loss：
- 定理：
  - 假设：
  - 结论：
  - 没有覆盖：
- 实验：
  - 是否可枚举真实分布？
  - 分布指标：
  - 优化/发现指标：
  - 最强基线：
- 与 T02 Foundations 的关系：
- 我认为最可能失败的地方：
- 最小复现实验：
```

### 填好的样例：O08

模板容易写成走过场，这里给一份到位的深度作参照。

- **状态**：ICML 2026 SPIGM **Workshop** Poster（非主会）；arXiv:2606.06272 v1（2026-06-04）。
- **一句话问题**：reward matching 只固定终止分布，内部流严重欠定；"该选哪一个内部流"此前没有原则性答案。
- **空间**：非无环有向图（允许环，flow = expected visit counts）；终止条件为吸收到 \(s_f\)。
- **目标分布**：\(P_T(x)=R(x)\)，且额外固定首步源分布 \(F(s_0\to u)=L(u)\)，并要求 \(\sum_u L(u)=\sum_x R(x)=1\)。
- **参数**：概念上直接对边流 \(F\) 做线性规划（非神经训练）；\(P_F,P_B\) 由 \(F\) 归一化诱导，\(Z\) 因 \(\sum R=1\) 而固定为 1。
- **数据**：无——本文是 LP/理论刻画，不是训练算法。
- **核心 loss**：不是 loss，而是约束优化 \(\min_{F\ge0}\sum_{e\in E^\circ}F(e)\) s.t. \(\operatorname{div}F(s)=L(s)-R(s)\)。
- **定理**（Thm 3.2）
  - 假设：Assumption 3.1 全部成立——非无环、源集与目标集可达、单位边长、\(\sum L=\sum R=1\)、取全局最优。
  - 结论：在图最短路 \(d_G\) 诱导的 transport cost 下，最小总流的最优值等于 Kantorovich OT 的最优值，且最优边流编码一个最优 coupling。
  - 没有覆盖：① 一般（非最短路）cost；② 未知 \(Z\)、未归一化 \(R\) 的常规 GFlowNet 场景；③ 神经参数化下的近似解——LP 最优 ≠ 训练收敛点；④ coupling 的**唯一性**（质量可沿多条等长最短路分裂）；⑤ 有限采样与 balance 残差如何传导到边缘违反与 cost gap。
- **实验**：是否可枚举真实分布——是（小图可精确解 LP）；分布指标——OT cost gap、marginal violation；优化指标——总流量；最强基线——**network simplex / min-cost-flow（多项式时间、精确、自带对偶证书）**，这是本文最强的反方论点，论文对此正面回应不足。
- **与 T02 的关系**：T02 指出内部流非唯一是 Foundations 框架的固有性质；本文给这个自由度提供了 OT 意义下的规范化选择原则，属于"在 T02 之上加一层选择准则"，而非修改 T02。
- **我认为最可能失败的地方**：Assumption 3.1 中"单位边长最短路 cost"是很强的限制——真实任务的 cost 通常不来自图距离；一旦换成一般 cost，等价性大概率失效。其次，最短内部流可能**降低**路径多样性与探索能力，与 GFlowNet 的核心卖点冲突。
- **最小复现实验**：在可枚举网格图上（a）用 POT/LP 解 Kantorovich OT，（b）解 minimum-flow LP，核对两者最优值是否相等；（c）再用神经 TB + flow 正则训练，报 marginal violation、OT cost gap、wall-clock，看神经解离 LP 最优有多远。这一步就是把定理从"LP 层面成立"推进到"训练层面成立"的关键实验。

---

## 9. 前沿追踪与本地索引

### 9.1 每周追踪入口

- [arXiv：按最新日期搜索 GFlowNet](https://arxiv.org/search/?query=GFlowNet&searchtype=all&abstracts=show&order=-announced_date_first&size=50)
- [IFT 6760B Bibliography](https://alexhernandezgarcia.com/teaching/gflownets25/bibliography)
- [Awesome-GFlowNets](https://github.com/zdhNarsil/Awesome-GFlowNets)
- [OpenReview](https://openreview.net/)：核验 ICLR/NeurIPS 接收状态
- [PMLR](https://proceedings.mlr.press/)：核验 ICML/AISTATS/UAI 正式版本

核验顺序：会议或期刊正式页 → OpenReview venue 状态 → arXiv comments/journal reference → 作者页 → 课程或社区索引。二手博客只用于发现材料，不用于确认定理或发表状态。

### 9.2 本地文件

| 资料 | 路径 | 用途 |
|---|---|---|
| 理论调研与六周学习指南 | [GFLOWNET_THEORY_GUIDE_CN.md](GFLOWNET_THEORY_GUIDE_CN.md) | 理论正文、推导、练习和学习计划 |
| 当前统一清单 | [GFLOWNET_RESEARCH_RESOURCE_CATALOG_CN.md](GFLOWNET_RESEARCH_RESOURCE_CATALOG_CN.md) | 唯一论文目录、课程、代码和前沿导航 |
| 核心 PDF | [literature/core/](literature/core/) | 离线精读 |
| 核心论文抽取文本 | [research/text/](research/text/) | 全文搜索和引用定位 |
| arXiv 原始检索 | [research/raw/](research/raw/) | 追踪新论文 |
| 论文标识核验 | [research/verified_papers.json](research/verified_papers.json) | 机器可读的论文 ID 记录 |
| 调研工具 | [.research-tools/aris/](.research-tools/aris/) | 后续自动化更新 |

---

## 10. 最小完成标准

不再单独复制论文标题；按编号勾选即可。**本最小集只覆盖理论与 OT 主线**；应用版图请先读 A01（P0 综述），再按 §0.2 的应用者路线下钻。

**4 周版**（对应理论指南前 4 周；砍掉连续 GFN 与 OT 线）

- [ ] **理论骨架**：T01、T02、T03、T05、T07
- [ ] **统一视角**：T08、T09、T14、T15、T17
- [ ] **正确性诊断**：T32、T49
- [ ] 复现：可枚举 HyperGrid 上比较 FM/DB/TB/SubTB，报 TV/mode/logZ（见 §7）

**6 周完整版**（在上面基础上追加）

- [ ] **空间扩展**：T12（连续，测度论）、T19、T36（非无环）
- [ ] **2026 前沿**：T47、T50、T51、T54
- [ ] **OT 专题**：O01、O02、O07、**O08**（配 §8 的填好样例）

> 时间预算提醒：本清单与理论指南的分周计划口径一致——每周 6–10 小时，数学基础强可压缩。**OT 线不要排进 4 周**：O08 代码 TBA，最小实验需要自己实现 minimum-flow LP 与 GFN 的映射，本身接近一个独立小研究。

完成 4 周版并复现一个可枚举任务后，就具备独立评审多数 GFlowNet 理论论文的基本框架；6 周版再加上非无环与 OT 这条线的判断力。

---

## 11. 2026-08 扩充收录（N 系列）

> 本节由 20 个扩充 agent 分车道调研产出，经机械去重后并入。**去重口径**：候选卡片 104 张 → 剔除与第 1–3 节已收 100 篇重复 0 张、剔除跨车道重复 23 张 → 净新增 81 篇；另并入并行审查流水线补录中本节所缺的 18 篇（见 §11.21）→ 共 99 篇；再加 2026-08 增量检索追补 1 篇 → 共 100 篇；再加顶会查全复核补录 6 篇（§11.23）→ **共 106 篇（N001–N106）**。

> 编号与前面的 T/O/A 系列平行，不改动原编号。每篇的发表状态由所属车道 agent 独立核验（渠道：arXiv API、OpenReview、PMLR/proceedings、dblp）；逐车道明细与被剔除清单见 `reports/expand/E*.md`。


### 11.1 理论：收敛性与样本复杂度（6 篇）

| 编号 | 论文与状态 | 简介 | 建议 |
|---|---|---|---|
| **N001** | [Convergences guarantees of GFlowNets](https://openreview.net/forum?id=JmsgmkdIkk) · NeurIPS 2025 FPI Workshop Poster | 针对"TB loss 下降是否意味着分布靠近目标"这一长期悬置的直觉给出正面定理:证明最小化 Trajectory Balance loss 时,学到分布与目标分布的 KL 散度被所最小化的量上界控制,并在小型采样任务上验证。它是目前对 TB 目标最直接的收敛性背书,填补 loss→KL 一环。 | 优先读:与 T32(loss→对象分布误差)、T51(loss→TV)构成完整链条,篇幅短可当天读完 <sub>来源 E01</sub> |
| **N002** | [From discrete-time policies to continuous-time diffusion samplers: Asymptotic equivalences and faster training](https://arxiv.org/abs/2501.06148) · TMLR | 研究无目标样本时训练神经 SDE/扩散采样器,证明离散步长趋零的极限下,GFlowNet 式熵正则 RL 目标与连续时间对象(PDE、路径空间测度)之间的一族渐近等价,并据此设计更快的训练方案(附官方代码)。它把离散 GFN 收敛分析接入连续时间数学工具箱,是桥梁性工作。 | 优先读:做连续/扩散 GFN 理论必读;与 T12、T20 连读 <sub>来源 E01/E06/E07</sub> |
| **N003** | [Maximum entropy GFlowNets with soft Q-learning](https://proceedings.mlr.press/v238/mohammadpour24a.html) · AISTATS 2024 主会 | 通过构造修正奖励,把 GFlowNet 与最大熵 RL 建立为精确对应而非特例式近似,并导出用 soft Q-learning 训练 GFN 的算法。它使 soft Q 迭代的收缩性与收敛理论可整体搬运到 GFN 训练,是收敛性分析最重要的 RL 接口之一,却一直不在清单里。 | 优先读:与 T14(Tiapkin 等,同年 AISTATS)对照,注意两者奖励修正方式不同 <sub>来源 E01/E02/E05</sub> |
| **N004** | [Investigating Generalization Behaviours of Generative Flow Networks](https://arxiv.org/abs/2402.05309) · TMLR 2025 | 系统实证检验"GFN+深度网络泛化良好"这一流行假设:构造奖励难度可调、\(p(x)\) 可精确计算、含未见测试集的图环境。发现 GFN 学到的函数确有利于泛化的隐式结构,但出人意料地对 offline/off-policy 训练敏感,而隐式学到的奖励对训练分布变化鲁棒。 | 建议读:做泛化界前先读它校准直觉;注意与 T29(arXiv 2407.03105)是两篇不同论文 <sub>来源 E01</sub> |
| **N005** | [Information-Geometric Forward Policy Training in GFlowNets](https://arxiv.org/abs/2608.03967) · 预印本 2026-08 | 把前向策略视为轨迹采样器,证明其一阶内蕴几何由轨迹族的 Fisher-Rao 度量给出、自然梯度是规范局部更新;给出轨迹 Fisher 到逐步条件二阶矩的精确分解,并划分精确计算/Monte Carlo/利用目标局部结构三种可计算范式。把"目标结构→优化几何"引入 GFN 训练动力学分析。 | 建议读:关心训练动力学与预条件优化者读;尚未同行评审,实验规模小,结论待验 <sub>来源 E01/E02</sub> |
| **N006** | [Analyzing GFlowNets: Stability, Expressiveness, and Assessment](https://openreview.net/forum?id=B8KXmXFiFj) · ICML 2024 SPIGM Workshop Poster | 从稳定性、表达能力、评估三维度分析 GFN:证明 balance 违反的影响在状态图上不均匀、节点影响力与其后代奖励挂钩,据此提出加权 balance loss 加速收敛;并证明合适状态图下 GFN 可精确表示任意树上分布,同时构造出 balance 不可达的失败案例。 | 选读:T32 同团队的 workshop 前身,"balance 不可达"表达能力反例在正式版之外仍独有;与 T32 连读 <sub>来源 E01/E18</sub> |

### 11.2 理论：训练目标与损失设计（4 篇）

| 编号 | 论文与状态 | 简介 | 建议 |
|---|---|---|---|
| **N007** | [Distributional GFlowNets with Quantile Flows](https://openreview.net/forum?id=vFSsRYGpjW) · TMLR 2024 · arXiv:2302.05793 | 把每条边的流从标量改为分布并用分位数函数参数化，提出 quantile matching（QM）这一分布式 TD 风格 balance 目标：既能处理随机奖励，又可经失真风险度量得到风险敏感策略；确定性基准上也因更强训练信号优于 FM/DB/TB。是"balance 条件分布化"的代表作。 | P1；研究随机奖励或分布式目标可升 P0，与 T13（随机环境）对读 <sub>来源 E02/E03/E04/E05</sub> |
| **N008** | [Order-Preserving GFlowNets](https://openreview.net/forum?id=VXDPXuq4oG) · ICLR 2024 · arXiv:2310.00386 | 不再拟合给定标量奖励，而是学习与候选（偏）序一致的奖励并按其采样；理论证明训练过程逐步稀疏化奖励景观，天然实现"先探索后利用"，免去奖励指数 \(\beta\) 调参，并直接适配多目标 Pareto 前沿。改变了"训练目标到底拟合什么"这一层设计。 | P1；做多目标或 reward tempering（T28）方向应精读 <sub>来源 E02/E10/E20</sub> |
| **N009** | [Delta-AI: Local objectives for amortized inference in sparse graphical models](https://proceedings.iclr.cc/paper_files/paper/2024/hash/710445227fa8c1b6a9ceada902dd4741-Abstract-Conference.html) · ICLR 2024 · arXiv:2310.02423 | 针对稀疏概率图模型的摊销推断，利用变量 Markov blanket 上的条件分布匹配构造 GFlowNet 式局部损失：每次参数更新无需实例化全部变量或完整轨迹，信用分配完全局部化，训练显著加速且支持 off-policy。展示了"用目标结构换局部 balance 目标"的设计范式。 | P1；与 T05/T06 的局部信用路线及 T10 对读 <sub>来源 E02/E06</sub> |
| **N010** | [Improved off-policy training of diffusion samplers](https://openreview.net/forum?id=vieIamY2Gi) · NeurIPS 2024 poster · arXiv:2402.05098 | 建立统一代码库与基准，系统比较连续 GFlowNet 目标（TB、VarGrad 型 Z-free 估计、FL-SubTB、Langevin 参数化）与 PIS 等模拟式变分目标训练 diffusion sampler 的表现，并提出目标空间局部搜索加回放缓冲的探索策略；修正了此前关于鲁棒性与样本效率的部分结论。 | P1；做连续 GFN/神经采样器的损失选型前必读，可升 P0，与 T20 对读 <sub>来源 E02/E04/E06/E07/E18</sub> |

### 11.3 理论：探索、信用分配与回放（4 篇）

| 编号 | 论文与状态 | 简介 | 建议 |
|---|---|---|---|
| **N011** | [Generative Augmented Flow Networks](https://arxiv.org/abs/2210.03308) · ICLR 2023（notable top 25%） · arXiv:2210.03308 | GAFlowNet 把 intrinsic motivation（Random Network Distillation）作为中间奖励注入 flow：同时用 edge-based 与 state-based 内在奖励增广，把只有终点奖励的稀疏信号变成沿轨迹的密集反馈，并证明增广目标渐近无偏于原 GFlowNet。它是"intrinsic reward 改善 GFlowNet 探索"的奠基论文，也是后续几乎所有 exploration bonus 工作的对照基线。 | **P0**：与 T06/T21（已知能量的 local credit）对照，区分"内在好奇心"与"已知势函数分解"两条信用来源。 <sub>来源 E03</sub> |
| **N012** | [Thompson Sampling for Improved Exploration in GFlowNets](https://arxiv.org/abs/2306.17693) · ICML 2023 SPIGM Workshop · arXiv:2306.17693 | TS-GFN 把"训练时选哪条轨迹"视作主动学习问题，用贝叶斯/多臂老虎机思想解决：将前向策略网络最后一层参数化为 ensemble，维护对策略的近似后验，每步采一个 ensemble 成员并按其策略采样轨迹，从而优先探索不确定区域。在 grid-world 与序列生成上比传统 off-policy（ε-noisy、tempering）收敛更快、样本效率更高，且仅增约 15% 计算。 | **P1**：作为温度/ε-greedy 之外的第三类探索范式；与 QGFN(T26)、α-GFN(T46) 比较"如何决定行为策略"。 <sub>来源 E03</sub> |
| **N013** | [An Empirical Study of the Effectiveness of Using a Replay Buffer on Mode Discovery in GFlowNets](https://arxiv.org/abs/2307.07674) · ICML 2023 SPIGM Workshop · arXiv:2307.07674 | 系统实证 replay buffer 对 GFlowNet 模式发现的作用：对比"无缓冲 / 随机采样缓冲 / R-PRS（Reward Prioritized Replay Sampling，仿 PER 的按奖励优先）"三种配置，在 Hypergrid 与分子合成环境上显示带缓冲尤其是 R-PRS 显著加速模式发现、提升多样性；关键结论是"提升来自更频繁访问高奖励轨迹，而非缓冲本身的存在"。 | **P1**：replay buffer 关键词最直接的实证基线；读 T54（submodular replay）前的必要背景。 <sub>来源 E03</sub> |
| **N014** | [Looking Backward: Retrospective Backward Synthesis for Goal-Conditioned GFlowNets](https://arxiv.org/abs/2406.01150) · ICLR 2025 · arXiv:2406.01150 | RBS 换一个方向用 backward policy：对未达标（零奖励）的前向轨迹，从目标状态出发用 P_B 合成一条"必然成功"的后向轨迹，反转后作为高质量正样本注入训练缓冲，从而在目标条件（goal-conditioned）、极稀疏奖励下制造大量可学习信号；配合 age-based 采样、P_B 正则（惩罚与均匀分布的 KL）与强化终点奖励反馈，样本效率大幅超过 HER、OC-GAFN 等强基线。 | **P1**：与 T25/T31（P_B 的参数化/优化）互补——展示 P_B 的第三种用法：hindsight 式后向数据增强。 <sub>来源 E03</sub> |

### 11.4 理论：连续、非无环与随机扩展（3 篇）

| 编号 | 论文与状态 | 简介 | 建议 |
|---|---|---|---|
| **N015** | [CFlowNets: Continuous Control with Generative Flow Networks](https://openreview.net/forum?id=yAYHho4fATa) · ICLR 2023（正式） | 首个把 GFlowNet 推向**连续控制/连续动作空间**的方法：用重要性采样近似连续状态的入流与出流，改写 flow-matching 损失，并给出流近似误差界（随采样数增大而衰减）。它是 T12（连续理论）在方法侧的**前身与被修正对象**——T12 明确指出其可达性、密度参数化等假设不严谨。作为连续 GFN 的历史起点与常用对照基线，必读但需带批判视角。 | **P0 精读**（与 T12 对照读） <sub>来源 E04/E20</sub> |
| **N016** | [MetaGFN: Exploring Distant Modes with Adapted Metadynamics for Continuous GFlowNets](https://openreview.net/forum?id=dtyNeemB7A) · TMLR 2025 | 针对"连续 GFN 的探索几乎无人研究"的空白，提出 Adapted Metadynamics：借 molecular dynamics 的 metadynamics 思想，对任意黑盒奖励在连续域上施加历史偏置势以逃离已访问模式，作为连续 GFN 的 off-policy 探索器，在多个连续/流形（球面、环面）环境上比既有探索策略更快收敛、发现更远模式。它填补 T23（离散 local search）在连续侧的对应空白。 | **P1**（连续域探索方法） <sub>来源 E04</sub> |
| **N017** | [Torsional-GFN: a conditional conformation generator for small molecules](https://arxiv.org/abs/2507.11759) · 预印本 2025 · arXiv:2507.11759 | 在**超环面流形** \([0,2\pi]^m\) 上采样分子扭转角的连续 GFN，前/后向策略用 von Mises 混合参数化，并以 GNN 将策略**条件化**于分子图与局部结构，实现单一模型跨多分子的摊销采样（Vargrad 损失、Boltzmann 目标）。相较 A05（逐分子连续构象）它是"连续流形 + 条件化 amortization"的代表，示范混合/流形状态空间上条件化连续 GFN 的可行路径。 | **P2**（连续流形+条件化范例，偏应用） <sub>来源 E04</sub> |

### 11.5 理论：与强化学习的统一视角（4 篇）

| 编号 | 论文与状态 | 简介 | 建议 |
|---|---|---|---|
| **N018** | [Trajectory Balance with Asynchrony (TBA)](https://arxiv.org/abs/2503.18929) · NeurIPS 2025（已接收，海报 #1115） | 把 **off-policy TB 目标**嵌入异步分布式 RL：多个 searcher 并行生成轨迹写入 replay buffer，单个 trainer 按 reward/recency 优先级异步采样更新，解耦"探索"与"学习"。在数学推理、偏好调优、自动红队三类任务上，对 Online DPO、Dr. GRPO 等强基线取得 4× 以上 wall-clock 加速并保持/提升精度。是"用 GFN 目标替代 on-policy PPO/GRPO"最有力的系统级证据。 | **P0/P1**（现代 RL 优化器视角必读） <sub>来源 E05</sub> |
| **N019** | [FlowRL: Matching Reward Distributions for LLM Reasoning](https://arxiv.org/abs/2509.15207) · ICLR 2026 | 主张用"匹配完整奖励分布"取代 PPO/GRPO 的奖励最大化：用**可学习配分函数** \(Z_\phi(x)\) 把标量奖励归一化为目标分布，最小化策略与目标的**反向 KL**，并证明其在期望梯度上**等价于 GFlowNet 的 TB 损失**。以 flow-balanced 优化促进多样探索，数学推理较 GRPO 平均 +10%、较 PPO +5.1%，缓解长链推理的模式坍缩。是 GFN↔GRPO/PPO 统一最重要的实证工作。 | **P0**（本车道最高优先） <sub>来源 E05</sub> |
| **N020** | [GDPO: Learning to Directly Align Language Models with Diversity Using GFlowNets](https://aclanthology.org/2024.emnlp-main.951/) · EMNLP 2024 Main，pp. 17120-17139 | 把**离线**偏好对齐视为贝叶斯推断，用 GFlowNet 直接从离线偏好数据学习"按奖励分布采样"的前向策略，得到多样性寻优的 DPO 变体（GFlowNet-DPO）。相比 DPO 过拟合奖励、坍缩到局部模式，GDPO 在对话生成与摘要上产生更多样且仍与偏好对齐的响应；引入 token-wise reference log-reward 与 tempering 系数。是 offline RL / DPO 与 GFN 的结合点，区别于 A33/A34/A35。 | **P1**（与 A33/A34/A35 LLM-RL 群对照） <sub>来源 E05/E19</sub> |
| **N021** | [Beyond Normalization: Rethinking the Partition Function as a Difficulty Scheduler for RLVR (PACED-RL)](https://arxiv.org/abs/2602.12642) · 预印本 2026 | 在 GFlowNet 式 LLM 后训练（如 FlowRL）之上重新诠释配分函数：它在最优处等于某 prompt 所有补全的奖励质量之和，可直接当作**在线准确率/难度估计器**，用于难度感知的自适应 prompt 选择；并利用 GFN 目标的 **off-policy 容忍度**做误差优先 replay，提升 RLVR 样本效率。两个组件都复用标准 GFN 训练已产生的信息，定位为 GRPO 的 GFN 式替代。 | **P2**（预印本，结论待复核） <sub>来源 E05</sub> |

### 11.6 理论：变分推断、MCMC 与摊销推断（7 篇）

| 编号 | 论文与状态 | 简介 | 建议 |
|---|---|---|---|
| **N022** | [Probabilistic Inference in Language Models via Twisted Sequential Monte Carlo](https://proceedings.mlr.press/v235/zhao24c.html) · ICML 2024 · arXiv:2404.17546 | 把 RLHF、自动红队、填充等 LLM 任务统一为对未归一化目标分布的后验采样，用学习的 twist 函数估计部分序列的未来价值、聚焦推断期算力，并给出双向 SMC 配分函数界以评估各类推断方法的 KL 误差。与 GFlowNet 微调（A21）构成"摊销训练 vs 推断时计算"的直接对照。 | **P1**。非 GFN 论文但同一问题谱系；与 A21 对读，关注 twist 学习与 soft RL/GFN 子轨迹目标的对应，其双向界也可用于评估 GFN 微调质量。 <sub>来源 E06</sub> |
| **N023** | [Outsourced Diffusion Sampling: Efficient Posterior Inference in Latent Spaces of Generative Models](https://proceedings.mlr.press/v267/venkatraman25a.html) · ICML 2025 · arXiv:2502.06999 | 把任意生成模型写成外生高斯噪声的确定性变换，在噪声空间训练扩散采样器（RL/GFN 式目标）以采样约束后验，使 GAN、(H)VAE、流模型等先验下的条件采样成为可能；噪声空间后验通常更平滑，更适合摊销推断。实验覆盖条件图像生成、RLHF 与蛋白结构生成。 | **P1**。A22（RTB）的直接延续，从数据空间转入潜空间；适合关注"冻结大先验+后验采样器"路线的读者。 <sub>来源 E06</sub> |
| **N024** | [Adaptive teachers for amortized samplers](https://openreview.net/forum?id=BdmVgLMvaf) · ICLR 2025 · arXiv:2410.01432 | 为摊销采样器训练引入自适应行为策略"教师"：教师专门采样学生模型高损失区域且能泛化到未探索模式，形成高效课程以提升模式覆盖与样本效率；在探索困难的合成环境、两类扩散采样任务与四类生化发现任务上验证。 | **P1**。与 T35/T48/T50 双网络探索簇同源且更早（ICLR 2025），先读它可理清该簇脉络；注意其教师目标与 T48 辅助 agent 的差异。 <sub>来源 E06</sub> |
| **N025** | [GFlowOut: Dropout with Generative Flow Networks](https://proceedings.mlr.press/v202/liu23r.html) · ICML 2023 · arXiv:2210.12928 | 把 dropout mask 视为潜变量，用 GFlowNet 学习其高度多峰的后验分布，取代独立固定分布采样或标准变分推断，并利用样本相关信息改进后验估计；实证改善分布外泛化与不确定性估计。是 GFN 服务于贝叶斯深度学习近似推断的代表工作。 | **P1/P2**。与 T10 的摊销后验一脉相承；阅读时关注 mask 后验质量如何评估及额外计算开销。 <sub>来源 E06</sub> |
| **N026** | [DynGFN: Towards Bayesian Inference of Gene Regulatory Networks with GFlowNets](https://papers.nips.cc/paper_files/paper/2023/hash/eb5254c4ee813d05af9c098f2d9c5708-Abstract-Conference.html) · NeurIPS 2023 · arXiv:2302.04178 | 利用 RNA velocity 把基因调控网络推断转为动力系统稀疏辨识，再用 GFlowNet 在含环依赖结构的组合空间上摊销贝叶斯后验，同时解决"调控结构天然含环、不能建成 DAG"与"观测噪声导致大等价类、需刻画不确定性"两个难题。 | **P1**。与 A17/A18 对读，是"非 DAG 结构后验"的代表；关注其对后验分布质量（而非单图准确率）的评估方式。 <sub>来源 E06/E12</sub> |
| **N027** | [Learning Decision Trees as Amortized Structure Inference](https://arxiv.org/abs/2503.06985) · 预印本 2025 · arXiv:2503.06985 | 把决策树构造写成顺序规划问题，训练 GFlowNet 策略从贝叶斯后验采样决策树（DT-GFN），采样集成即得随机森林；在表格数据分类、分布偏移鲁棒性与异常检测上超过主流树模型和深度方法，且模型描述长度更短、可解释，集成规模上表现一致扩展。 | **P2**。GFNOrg 出品的"离散结构后验"新形态，作观察项收录；截至 2026-08 仍是预印本，正式引用时勿标会议。 <sub>来源 E06/E12/E20</sub> |
| **N028** | [Stop the Sampler! Classifier-Based Adaptive Stopping for Sampling Kernels](https://arxiv.org/abs/2606.16073) · ICML 2026 SPIGM Workshop · arXiv:2606.16073 | 把 MCMC 轨迹的终止时机当作可学习组件：在非无环 GFlowNet 理论框架内训练状态依赖分类器决定链何时停止，经 detailed balance 建立最优分类器与目标密度的联系，并用多层级训练方案辅助复杂几何下的探索；实证缩短平均轨迹长度并改善混合与模式覆盖。 | **P2**。GFN×MCMC 融合的探路工作，与 T19/T36 非无环理论连读；目前仅 workshop 发表，引用时务必注明。 <sub>来源 E06</sub> |

### 11.7 交叉：扩散采样器与随机最优控制（8 篇）

| 编号 | 论文与状态 | 简介 | 建议 |
|---|---|---|---|
| **N029** | [Improved sampling via learned diffusions](https://proceedings.iclr.cc/paper_files/paper/2024/hash/2c9f31add52c28c2db39329b464bce91-Abstract-Conference.html) · ICLR 2024 poster · arXiv:2307.01198 | Richter 与 Berner 把 PIS、DIS、DDS、Schrödinger bridge 统一为"广义 Schrödinger 桥"问题，在时间反转路径测度之间定义变分散度族；提出 log-variance loss，避免对 SDE 求解器求导、显著抑制 mode collapse。该 loss 与 GFN 的 VarGrad/TB 目标同构，是两社区目标函数层面的交汇点。 | **P0**。SOC 阵营的统一框架论文；读时对照 T08（TB=VarGrad 关系）能看清"同一目标、两套语言"。 <sub>来源 E07</sub> |
| **N030** | [Beyond ELBOs: A Large-Scale Evaluation of Variational Methods for Sampling](https://proceedings.mlr.press/v235/blessing24a.html) · ICML 2024 · arXiv:2406.07423 | Blessing、Vargas、Neumann 等建立神经采样器统一基准：标准化任务套件覆盖 SMC、AFT、CRAFT、FAB、GMMVI 及各类 diffusion sampler，系统研究 ELBO/EUBO 等指标何时掩盖 mode collapse，并提出熵模式覆盖等新度量。结论之一：高 ELBO 不代表模式覆盖，diffusion 类方法在高维更抗塌缩。 | **P1**。评估方法论必读；GFN 社区"多样性/模式覆盖"话语与采样器社区"mode collapse 度量"在此对接，可与 T47（evaluation balance）互参。 <sub>来源 E07/E18</sub> |
| **N031** | [Iterated Denoising Energy Matching for Sampling from Boltzmann Densities](https://proceedings.mlr.press/v235/akhound-sadegh24a.html) · ICML 2024 poster · arXiv:2402.06121 | Mila 团队（含 Bengio、Malkin、Tong）提出 iDEM：只用能量函数及其梯度、不需数据样本的迭代式随机 score matching，内环模拟自由、外环用模型自采样探索，凭 diffusion 的快速模式混合平滑能量面，首次在 LJ-55（165 维）粒子系统上用纯能量训练成功，训练快 2–5 倍。 | **P1**。GFN 作者群给出的"绕开轨迹 RL"的另一条路线；与 NEW-E07-1 对读可看到同一批人对 on/off-policy 与模拟自由两条技术路线的取舍。 <sub>来源 E07</sub> |
| **N032** | [Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models with Memoryless Stochastic Optimal Control](https://openreview.net/forum?id=xQBRrtQM8u) · ICLR 2025 **Spotlight** · arXiv:2409.08861 | Domingo-Enrich、Chen（Meta）把 reward 微调严格表述为 SOC 问题：证明必须使用 memoryless 噪声调度才能无偏收敛到 reward-tilted 分布，并把 SOC 化为回归式 Adjoint Matching 目标。任务设定与 A22（relative TB 微调 diffusion 后验）完全同构，是 SOC 阵营对同一问题的答案。 | **P0**。与 A22、A23 三方对照读：同是"从 \(p_{\text{pre}} \cdot r\) 采样"，GFN 用 off-policy 平衡目标、本文用 memoryless SOC 回归，比较偏差与样本效率的差异。 <sub>来源 E07</sub> |
| **N033** | [Adjoint Sampling: Highly Scalable Diffusion Samplers via Adjoint Matching](https://proceedings.mlr.press/v267/havens25a.html) · ICML 2025 · arXiv:2504.11713 | Meta FAIR 将 Adjoint Matching 特化为从未归一化密度采样：Reciprocal Adjoint Matching 加 replay buffer，使梯度更新次数远超能量评估次数，成为首个如此可扩展的 on-policy 方法；支持 SE(3) 对称与周期边界，扩展到神经能量函数上的摊销构象生成并开源基准。 | **P1**。目前神经采样器"规模化"的代表作；GFN 读者应关注其"每次能量评估多次更新"的效率论证，与 GFN replay buffer 探索（NEW-E07-1）异曲同工。 <sub>来源 E07</sub> |
| **N034** | [Feynman-Kac Correctors in Diffusion: Annealing, Guidance, and Product of Experts](https://proceedings.mlr.press/v267/skreta25a.html) · ICML 2025 spotlight poster · arXiv:2503.02819 | Skreta、Akhound-Sadegh、Doucet、Brekelmans、Tong、Neklyudov 等基于 Feynman–Kac 公式推导加权模拟方案：不重训即可从退火、几何平均或专家乘积分布精确采样，用 SMC 重采样做推断时扩展，应用于温度退火摊销采样、多目标分子生成与 CFG 改进。 | **P1**。"推断时校正"路线的核心论文；GFN 的温度条件化（T28）与多目标组合（A03、T55）可与之直接对话。 <sub>来源 E07</sub> |
| **N035** | [Sequential Controlled Langevin Diffusions](https://openreview.net/forum?id=dImD2sgy86) · ICLR 2025 poster · arXiv:2412.07081 | Chen、Richter、Berner、Blessing 等在路径测度上统一 SMC 与学习式 diffusion sampler：SMC 的重采样/MCMC 提供稳健性与渐近保证，学习的控制漂移提供适应性，SCLD 常以先前 diffusion sampler 约 10% 的训练预算达到更好性能。正文明确引用 GFN 文献并对比离散时间与连续时间视角。 | **P1**。"退火 + 学习传输"的集大成；与 GFN 的 replay/局部搜索探索（NEW-E07-1）互为补充，训练同样用 log-variance loss。 <sub>来源 E07</sub> |
| **N036** | [Proximal Diffusion Neural Sampler](https://openreview.net/forum?id=XTHQqS7ObC) · ICLR 2026 | Guo、Choi、Tao、Y. Chen（Georgia Tech）用路径测度上的 SOC 统一连续（SDE）与离散（CTMC）神经采样器，指出一次性全局优化会加剧 mode collapse，改用近端点法把学习拆成一串 KL 约束子问题，在分子动力学与 Ising/Potts 等连续和离散基准上达 SOTA。注意：论文未把 GFN 作为 baseline，对话在问题层面——SOC 路线已伸入 GFN 的离散主场。 | **P1**。2026 前沿信号：离散采样不再是 GFN 专属；其近端退火与 GFN 温度/退火课程可比较，两社区基准仍未打通是显著研究空档。 <sub>来源 E07</sub> |

### 11.8 交叉：Schrödinger 桥与熵正则 OT（10 篇）

| 编号 | 论文与状态 | 简介 | 建议 |
|---|---|---|---|
| **N037** | [Discrete Diffusion Schrödinger Bridge Matching for Graph Transformation (DDSBM)](https://arxiv.org/abs/2410.01500) · ICLR 2025 | 用连续时间马尔可夫链把 Iterative Markovian Fitting 推广到高维离散/图空间求解 SB 并证明收敛；关键是把"独立修改节点与边"的参考动力学对应到 **以图编辑距离(GED)为 cost 的熵正则 OT(EOT)**，落地于分子优化（最小结构改动达成目标性质）。正中**课题3**生态位，但用 CTMC+IMF 而非 GFlowNet。 | P1｜精读第 3 节 GED↔EOT 对应与离散 IMF 收敛证明；对照 O07/O08 的图 cost 与 minimum-flow。 <sub>来源 E08</sub> |
| **N038** | [Generalized Schrödinger Bridge on Graphs (GSBoG)](https://arxiv.org/abs/2602.04675) · 预印本 2026-02（v2） | 首个**数据驱动**的"图上广义 Schrödinger bridge"：以受控 CTMC + 迭代比例拟合 + TD 目标，学习满足源—目标边缘、并在 state-dependent running cost 下优化中间轨迹的**可执行图传输策略**，避免稠密全局 solver、可扩展到大稀疏图。与**课题3**高度撞车，其"可执行 trajectory-level 策略"正是 O08 主打卖点（兼及课题1/4）。 | P0｜必读，最直接的"可执行图传输策略"竞品；核对它与 minimum-flow GFN 在边缘约束、cost 形式、可扩展性上的异同。 <sub>来源 E08</sub> |
| **N039** | [Entering the Era of Discrete Diffusion Models: A Benchmark for Schrödinger Bridges and Entropic Optimal Transport](https://arxiv.org/abs/2509.23348) · 预印本 2025-09（v2） | 首个**离散空间 SB/EOT 基准**：用 CP 参数化构造"解析已知 SB 解"的分布对，可严格评测 solver 是否真解 EOT/SB（而非只看 FID/MSE）；副产品给出 DLightSB、DLightSB-M、α-CSBM 三个求解器并在高维离散设置比较。为**课题2**提供"有真值可比"的评测基座，同时补齐**课题3**的评测标准。 | P1｜作为课题2/3 评测基座；重点看解析 SB 解构造与 CP 参数化能否移植到 GFN 边流/coupling 的误差评测。 <sub>来源 E08</sub> |
| **N040** | [Minimal-Action Discrete Schrödinger Bridge Matching for Peptide Sequence Design (MadSBM)](https://arxiv.org/abs/2601.22408) · 预印本 2026-01（v1） | 把肽序列生成建模为**氨基酸编辑图上的受控 CTMC**，以预训练蛋白语言模型 logits 作参考过程，学习时间相关控制场以走"最小作用量/低成本"传输路径；并首次给离散 SB 加**分类器引导**。属离散 SB 的生物序列应用，占据**课题3**在序列空间的生态位；"最小作用量=最短/低成本路径"与 O07/O08 思路呼应。 | P2｜扫读；关注"最小作用量路径 + pLM 参考过程 + 分类器引导"能否搬到序列/编辑型 GFN。 <sub>来源 E08</sub> |
| **N041** | [Unsupervised Learning for Optimal Transport plan prediction between unbalanced graphs (ULOT)](https://papers.nips.cc/paper_files/paper/2025/hash/873fd89b3e4db1f6242c2333673e104d-Abstract-Conference.html) · NeurIPS 2025 主会 · arXiv:2506.12025 | 用 GNN + cross-attention、并**以 FUGW 权衡超参为条件**，无监督地预测两图之间的（不平衡）OT plan，比经典 solver 快约两个数量级，且预测 plan 可为经典 solver 提供 **warm-start**、对输入与超参可微。直接占据**课题1**（条件/摊销图 OT）核心卖点，并完整演示**课题4**（神经 proposal→经典修正）范式。 | P0｜必读，课题1/4 头号竞品；核对 FUGW 条件化、warm-start 协议与"2 个数量级加速"的评测口径。 <sub>来源 E08</sub> |
| **N042** | [Modeling Stochastic Conditional Dynamics from Sparse Observations via Kernel-Stabilized Flow Matching (CVFM)](https://arxiv.org/abs/2411.08314) · TMLR 2026 | 提出 **Conditional Variable Flow Matching**：在**连续条件密度空间**上摊销学习条件分布之间的流；用条件 Wasserstein 距离 + "条件失配核"抑制稀疏/不配对数据下 mini-batch 条件耦合的方差爆炸，并可扩展逼近**条件 Schrödinger bridge**。与**课题1**（条件/摊销 OT）撞车，但作用于连续空间而非图——保留了 GFN 在"离散/图"上的差异化余地。 | P1｜看"条件失配核"如何稳定条件 OT；对照条件 GFN 的跨 (L,R) 泛化设计。 <sub>来源 E08</sub> |
| **N043** | [Diffusion Schrödinger Bridge Matching (DSBM)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/c428adf74782c2092d254329b6b02482-Abstract.html) · NeurIPS 2023 · arXiv:2303.16852 | 提出 **Iterative Markovian Fitting (IMF)** 与 DSBM：用一系列"简单回归"迭代逼近 SB，而 **SB 恰恰恢复熵正则版 OT**，避免旧 DSB 的时间离散化与"遗忘"误差累积。它是 DDSBM/CSBM/MadSBM 的**方法学母本**，也是理解"路径熵正则 + 首末边缘约束"的连续基线。**课题3**方法奠基（与已收录 O04《Schrödinger Bridge Flow》同源，O04 是其 α-IMF/flow 变体）。 | P0｜精读 IMF 与"SB=EOT"的等价，理解所有离散 SB 方法的方法学根。 <sub>来源 E08</sub> |
| **N044** | [Generalized Schrödinger Bridge Matching (GSBM)](https://arxiv.org/abs/2310.02233) · ICLR 2024 | 广义 SB matching：在固定首末边缘下支持**依赖状态与分布的一般 running cost**，用变分/条件模拟分解逐步求解。它是 GSBoG 的**连续母本**，示范"如何把任务相关的中间成本塞进 SB"，正对应**课题3**里"路径熵 + coupling 成本双层正则"的连续实现。 | P1｜看"如何把中间 running cost 塞进 SB"，对照课题3 的双层正则设计。 <sub>来源 E08</sub> |
| **N045** | [Light Schrödinger Bridge (LightSB)](https://openreview.net/forum?id=WhZoCLRWYJ) · ICLR 2024 · arXiv:2310.01174 | 轻量、**免模拟**的 SB/EOT 求解器：用 sum-exp 二次型参数化 Schrödinger potential + 能量视角，中等维度几分钟即可求 SB；并证明其为 **SB 的通用逼近器**、给出**泛化误差分析**。它是基准(2509.23348)里 DLightSB 的母本；其"通用逼近 + 泛化误差界"思路对**课题2**（残差→OT 误差界）最具借鉴价值（兼**课题3**基线）。 | P1｜看通用逼近 + 泛化误差分析，思路可迁移到课题2 的界证明。 <sub>来源 E08</sub> |
| **N046** | [Categorical Schrödinger Bridge Matching (CSBM)](https://proceedings.mlr.press/v267/ksenofontov25a.html) · ICML 2025 · arXiv:2502.01416 | 为**离散时间/离散空间 SB** 提供理论与算法基座：证明**离散时间 IMF (D-IMF) 在有限空间、一般 Markov 参考过程下收敛到唯一 SB**，据此给出可"少步生成"的 Categorical SB Matching（VQ 图像/合成数据验证）。论文明确把 DDSBM 列为"唯一已有离散 SB 方法"并加以改进——是**课题3**"离散 SB 已有收敛性证明"的最强反方证据。 | P0｜精读 D-IMF 收敛定理（有限空间 + 一般 Markov 参考），课题3 的硬证据。 <sub>来源 E08</sub> |

### 11.9 交叉：神经最优传输与图上 OT（8 篇）

| 编号 | 论文与状态 | 简介 | 建议 |
|---|---|---|---|
| **N047** | [Neural Optimal Transport](https://openreview.net/forum?id=d8CBRlWNkqH) · ICLR 2023 Spotlight | Korotin 等提出基于鞍点对偶的神经 OT 算法，可同时求强/弱 cost 下的确定性 map 与随机 plan，并证明神经网络是 transport plan 的通用逼近器。它是当代 neural OT solver 谱系（含 O05）的方法学源头。 | **P1**：作为 neural OT 基线谱系入口；读 O05/O06 前先读它，理解 max-min 目标与 fake solution 风险 <sub>来源 E09</sub> |
| **N048** | [Neural Optimal Transport with General Cost Functionals](https://openreview.net/forum?id=gIiz7tBtYZ) · ICLR 2024 poster | 把 NOT 推广到一般 cost functional（不限 \(\ell^1/\ell^2\)），可编码类别保持、成对约束等任务信息，并给出恢复 plan 的误差分析。对 GFN×OT 主线的直接启发：图最短路诱导的非欧 cost 正属于"一般 cost"，其鞍点框架与误差分析可作对照。 | **P1**：与 O08 的图 cost 假设对读，思考"一般 cost + 边缘约束"在神经参数化下如何训练 <sub>来源 E09</sub> |
| **N049** | [The Monge Gap: A Regularizer to Learn All Transport Maps](https://proceedings.mlr.press/v202/uscidda23a.html) · ICML 2023（PMLR 202:34709-34733） | 提出 Monge gap 正则子：度量任意映射 \(T\) 偏离 c-OT 最优性的程度，摆脱 ICNN 架构约束与平方欧氏 cost 限制，用"拟合损失 + 最优性正则"学任意 cost 的 transport map。其"用一个可计算的最优性偏差项做正则"的思路，与 GFN 里"balance 残差 + minimum-flow 正则"结构同型。 | **P1**：方法短小清晰；重点看 Monge gap 定义与 O08 的 min-flow 目标在"最优性度量"上的类比 <sub>来源 E09</sub> |
| **N050** | [Progressive Entropic Optimal Transport Solvers](https://openreview.net/forum?id=7WvwzuYkUq) · NeurIPS 2024 poster | Apple/NYU 提出 ProgOT：借动态 OT 的时间离散把质量位移拆成多步，每步用调度好的 \(\varepsilon\) 跑 Sinkhorn，大规模下比标准 EOT 更快更稳，甚至优于部分神经求解器，并证明 map 估计的统计一致性。是"经典求解器仍很能打"的代表。 | **P1**：做 GFN-OT 实验时的必备强基线；关注其 \(\varepsilon\) 调度与分步思想能否映射到 GFN 的逐步构造 <sub>来源 E09</sub> |
| **N051** | [Semidefinite Relaxations of the Gromov-Wasserstein Distance](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8189d86a5d8dea0694d43bb90e01c14d-Abstract.html) · NeurIPS 2024 | GW 距离本质是非凸二次规划，既有求解器只能到局部最优。本文给出 SDP 松弛：可多项式时间求 transport plan，且能对任意 plan 计算到全局最优的 optimality gap，实验中常直接命中全局最优并附证书。对"GFN 学到的 coupling 离最优多远"这类问题提供了可计算的检验思路。 | **P1**：重点看 optimality gap 证书的构造——GFN×OT 的"balance 残差→OT gap"界可借鉴此范式 <sub>来源 E09</sub> |
| **N052** | [Any2Graph: Deep End-To-End Supervised Graph Prediction With An Optimal Transport Loss](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b81a352c156ca123c30c740f147a4496-Abstract.html) · NeurIPS 2024 | 提出部分掩码融合 GW（PMFGW）损失：置换不变、可微、尺寸无关，让深度模型端到端地从任意模态预测整图（卫星图→路网、指纹→分子）。它展示了"以 OT/GW 为损失在组合对象空间上学习"的完整工程路线，与 GFN 逐步构造图的路线互为对照。 | **P1**：与 A19（GFN 解图组合问题）对读：一次性 OT 损失回归 vs 逐步流式构造，各自的尺寸泛化与代价 <sub>来源 E09</sub> |
| **N053** | [A Convergent Single-Loop Algorithm for Relaxation of Gromov-Wasserstein in Graph Data](https://openreview.net/forum?id=0jxPyVWmiiF) · ICLR 2023 poster | 提出 BAPG：首个有收敛保证的单循环 GW 近似算法，用放松耦合可行性换取效率，基于 Luo-Tseng 误差界给出不动点集与 GW 临界点集的距离界，在图对齐/图划分上快且好。是"图上 GW 的经典可扩展求解器"代表。 | **P2**：作大规模图对齐基线用；其"放松可行性 + 误差界"的分析套路对放松边缘约束的 GFN 训练有参考价值 <sub>来源 E09</sub> |
| **N054** | [Estimating Barycenters of Distributions with Neural Optimal Transport](https://proceedings.mlr.press/v235/kolesov24a.html) · ICML 2024 poster | 把 NOT 的对偶鞍点法从两边缘推广到 Wasserstein barycenter：用双层对抗目标取代既有方法的三层优化，支持一般 cost，并给出理论误差界。barycenter 即"同时满足多个边缘约束的聚合传输"，其对偶参数化与误差分析可为多目标/条件 GFN 的多边缘约束设计提供模板。 | **P2**：与 A03（多目标 GFN）对读，比较"多边缘聚合"的 OT 对偶实现与偏好条件化 GFN 实现 <sub>来源 E09</sub> |

### 11.10 应用：分子与药物发现（4 篇）

| 编号 | 论文与状态 | 简介 | 建议 |
|---|---|---|---|
| **N055** | [TacoGFN: Target-conditioned GFlowNet for Structure-based Drug Design](https://openreview.net/forum?id=N8cPv95zOU) · TMLR 2024 | 首个把 SBDD 表述为"以蛋白口袋为条件、按 affinity×药性×可合成度 reward 成比例采样"的 GFlowNet：不拟合有限 protein-ligand 复合物数据分布，而对全体口袋诱导的 reward 分布建模，并用基于 pharmacophore 的快速对接预测器压低 reward 成本、支持训练中评估数百万分子。CrossDocked2020 上 median Vina Dock −8.44（微调后 −10.93）、success rate 56%→88.8%，生成速度较优化式方法快数个量级。 | **P1**：SBDD×GFlowNet 的奠基读物；重点看 target-conditioning 的 reward 设计与 pharmacophore 代理对接器如何压低 reward 成本 <sub>来源 E10</sub> |
| **N056** | [Generative Flows on Synthetic Pathway for Drug Design（RxnFlow）](https://openreview.net/forum?id=pB1XSj2y4X) · ICLR 2025 | 在"反应模板 + 可购买 building block"的合成路径空间上训练 GFlowNet，天然保证可合成性；核心创新 action space subsampling 让其能在 120 万 building block × 71 反应模板的超大动作空间上训练而无显著开销，且不重训即可更换/扩充 building block 或新增目标（如溶解度）。在 CrossDocked2020 pocket-conditional 上取得 SOTA（平均 Vina −8.85、可合成率 34.8%），全面超越反应/片段/原子级基线。 | **P1**：与 A09/A11 并读，是 reaction-space 合成路线的规模化代表；关注大动作空间子采样与 action embedding <sub>来源 E10</sub> |
| **N057** | [Compositional Flows for 3D Molecule and Synthesis Pathway Co-design（CGFlow）](https://proceedings.mlr.press/v267/shen25b.html) · ICML 2025 | 提出 Compositional Generative Flows：把 flow matching 扩展到"逐步构造组合对象 + 建模连续状态"，并接入 GFlowNet 理论实现 reward-guided 采样；据此的 3DSynthFlow 同时共设计分子的合成路径与 3D 结合构象。LIT-PCBA 全部 15 个靶点取得 SOTA 结合亲和、采样效率较 2D 合成基线提升 4.2×；CrossDocked 上 Vina Dock −9.42、AiZynth 成功率 36.1%，为首个两项同时 SOTA。 | **P1**：SBDD 的"3D 姿态 + 合成路径"联合建模前沿；关注 flow matching 与 GFlowNet 的融合方式及连续态处理 <sub>来源 E10</sub> |
| **N058** | [Sample-efficient Multi-objective Molecular Optimization with GFlowNets（HN-GFN）](https://proceedings.neurips.cc/paper_files/paper/2023/hash/fbc9981dd6316378aee7fd5975250f21-Abstract.html) · NeurIPS 2023 | 将多目标分子优化纳入多目标贝叶斯优化（MOBO），以单个 preference-conditioned 的 hypernetwork-GFlowNet 作为 acquisition 优化器，从近似 Pareto front 采样一批多样候选；并提出 hindsight 式 off-policy 策略在不同偏好间共享高分子以加速学习。在 gsk3b/jnk3/qed/sa 等真实 MOBO 设定下于候选质量与样本效率上领先，并能在偏好上泛化。 | **P1**：多目标分子优化代表作；与 A03 对读"偏好条件 + hypernetwork + MOBO 采集"路线，关注昂贵 oracle 下的样本效率与多样性 <sub>来源 E10</sub> |

### 11.12 应用：结构学习与因果发现（3 篇）

| 编号 | 论文与状态 | 简介 | 建议 |
|---|---|---|---|
| **N059** | [Expert-Aided Causal Discovery of Ancestral Graphs](https://doi.org/10.1016/j.ins.2026.123816) · Information Sciences, Vol. 756, Art. 123816 · arXiv:2309.12032 | 首个在潜在混杂（不假设因果充分性）下做分布式推断的 GFlowNet 因果发现方法：按 BIC 类得分成比例采样祖先图（AG），用最优实验设计主动向专家或 LLM 提问，再以重要性加权把带噪反馈并入后验而无需重训，并证明反馈足够准确时收敛到真实祖先图。 | **P1**。把车道从"DAG 后验"扩展到"潜混杂下的祖先图后验 + 人在环"；建议与 A17 对照读，注意期刊版与 arXiv 早期版标题不同。 <sub>来源 E12</sub> |
| **N060** | [Learning Equivalence Classes of Bayesian Network Structures with GFlowNet](https://openreview.net/forum?id=FAcc7oAdaa) · TMLR 2025 | 提出 CPDAG-GFN：不在 DAG 空间、而直接在 Markov 等价类（CPDAG）空间上用 GFlowNet 学习后验并抽取高分候选，配合偏稀疏过滤器改进与真图的对齐。它正面回应"观测数据至多识别到等价类"的根本限制，避免 DAG 后验把概率质量摊到同一 MEC 内的冗余图上。 | **P1/P2**。与 A17 对照，理解"后验定义在 DAG 还是 CPDAG 空间"对评价指标与可解释性的影响。 <sub>来源 E12</sub> |
| **N061** | [Generative Flow Networks: Theory and Applications to Structure Learning](https://arxiv.org/abs/2501.05498) · 博士论文 · arXiv:2501.05498 | DAG-GFlowNet（A17）与 JSP-GFN（A18）一作 Deleu 的博士论文：上篇系统建立 GFlowNet 数学基础及其与变分推断、强化学习的联系和连续空间扩展；下篇完整展开贝叶斯结构学习——在观测与干预数据下对 DAG 结构及机制参数做联合后验近似。是本车道最系统的整合文献。 | **P0 工具书式使用**。作为 §3.2 的导读与统一记号来源，不必逐页精读；也可放学习资源节。 <sub>来源 E12</sub> |

### 11.13 应用：组合优化（1 篇）

| 编号 | 论文与状态 | 简介 | 建议 |
|---|---|---|---|
| **N062** | [Symmetric Replay Training: Enhancing Sample Efficiency in Deep RL for Combinatorial Optimization (SRT)](https://arxiv.org/abs/2306.01276) · **ICML 2024 主会** | 提出对称回放训练：利用 CO 解空间的对称性（多条 (partial) 轨迹通向同一状态，类似 GFN 后向策略 \(P_B\)），周期性用最大似然对**对称轨迹**做直接信用分配，显著提升样本效率。**边界说明**：这是"DRL for CO"的通用样本效率方法，**GFlowNet 是其两个 backbone 之一**（论文 §4.4 专门讨论与 GFN 的关系，其 GFN 实例化在分子优化 PMO 基准；路由 TSP/CVRP/PCTSP 用 POMO）。与 GFACS/AGFN 同属 KAIST（Kim/Park）团队的 GFN-for-CO 方法脉络，收录以补齐"主会级样本效率方法"节点。 | **P2**（边界条目；若整合者偏好车道纯度，可移至训练/探索或分子车道。价值在把 GFN 的 \(P_B\)/对称性与 CO 样本效率显式联系） <sub>来源 E13</sub> |

### 11.14 应用：LLM 推理与后训练（6 篇）

| 编号 | 论文与状态 | 简介 | 建议 |
|---|---|---|---|
| **N063** | [Learning Diverse Attacks on Large Language Models for Robust Red-Teaming and Safety Tuning](https://arxiv.org/abs/2405.18540) · ICLR 2025 Poster | Lee、Kim、Malkin、Jain、Bengio 等（KAIST/Mila）提出两阶段红队方法：先以毒性×似然为奖励做 GFlowNet 微调采样攻击提示，再收集高奖励提示做 MLE 平滑。相比带新颖性正则的 RL 红队，攻击更多样、跨目标模型迁移更好；用其数据安全微调后的模型对其他 RL 攻击更稳健。 | **P1**（A32 的直接前驱，安全线必读；与 A32 对照看 TB 稳定化改进） <sub>来源 E14/E19</sub> |
| **N064** | [GFlowNet Fine-tuning for Diverse Correct Solutions in Mathematical Reasoning Tasks](https://arxiv.org/abs/2410.20147) · 预印本 2024-10 · arXiv:2410.20147 | 首个在 GSM8K/MATH 上系统对比 GFlowNet 微调与 PPO/DPO/RFT 的实证研究：GFlowNet 在 Pass@8 与各基线相当的前提下，采 8 个解时"不同正确解"数量最高（GSM8K 上 2.34 对 PPO 的 1.73），说明分布匹配微调能从多样中间推理步收敛到同一正确答案。 | **P2**（实证小论文；作为"GFN vs 奖励最大化 RL"在数学推理上的基线证据引用，FoR 将其列为并行工作） <sub>来源 E14</sub> |
| **N065** | [Accurate and Diverse LLM Mathematical Reasoning via Automated PRM-Guided GFlowNets](https://arxiv.org/abs/2504.19981) · 预印本 2025-04 · arXiv:2504.19981 | 用 MCTS + 相似度数据增强自动训练过程奖励模型（PRM），再把 GFlowNet（SubTB 变体）从 token 级提升到**推理步级**：状态为部分解、动作为完整推理步。Llama3.2-3B 在 MATH Level 5 提升 +2.59%，泛化到 SAT MATH +9.4%，解间语义相似度显著低于 PPO。 | **P2**（步级 GFN + 自动 PRM 的代表配方；未过审，结论以 arXiv 版为准） <sub>来源 E14</sub> |
| **N066** | [Proof Flow: Preliminary Study on Generative Flow Network Language Model Tuning for Formal Reasoning](https://arxiv.org/abs/2410.13224) · NeurIPS 2024 Workshop | 把 GFlowNet 微调引入 Lean 神经定理证明（NTP）：以 tactic 为动作、可验证证明为奖励，基于 ReProver（350M）初始化。受限算力下解题数从基线 4/20 提升到 8–9/20，与 SFT（9/20）相当，并给出奖励塑形与 replay 的消融。展示 GFN 对搜索期探索的价值。 | **P2**（工作坊初步研究，形式化推理方向目前唯一的 GFN 条目；结论谨慎引用） <sub>来源 E14</sub> |
| **N067** | [Latent Logic Tree Extraction for Event Sequence Explanation from LLMs](https://proceedings.mlr.press/v235/song24j.html) · ICML 2024 Poster，PMLR 235:46238-46258 | LaTee 用摊销 EM 从 LLM 中抽取解释事件序列的潜逻辑树：E 步以 LLM 先验 × 时序点过程似然定义后验，用 GFlowNet 微调 LLM 采样多样逻辑树；M 步更新点过程参数并精调 LLM 先验。行为数据上未来事件预测较注意力 TPP 相对提升约 20%。 | **P1/P2**（"GFN×LLM 四元组"中目录缺失的一篇；与 T10 GFlowNet-EM、A21 对照读） <sub>来源 E14</sub> |
| **N068** | [Latent Thought Flow: Efficient Latent Reasoning in Large Language Models](https://arxiv.org/abs/2606.16222) · 预印本 2026-06 · arXiv:2606.16222 | LTF 把推理建模为**变长连续潜思维轨迹**，用 continuous GFlowNet（随机潜转移）训练采样器匹配"答案质量 × 计算成本"诱导的后验；提出熵加权 SubTB 处理稀疏监督，并用参考先验正则稳定探索。较 CoLaR/ReGuLaR 等潜推理基线平均提速且准确率 +9.5%、推理长度 −27.2%。 | **P1/P2**（潜空间 CoT × 连续 GFN（T12）的交叉点，方向重要但极新，需复核实验与后续录用状态） <sub>来源 E14</sub> |

### 11.16 应用：材料与物理科学（3 篇）

| 编号 | 论文与状态 | 简介 | 建议 |
|---|---|---|---|
| **N069** | [Efficient Symmetry-Aware Materials Generation via Hierarchical Generative Flow Networks（SHAFT / CHGlowNet）](https://pubs.rsc.org/en/content/articlelanding/2026/dd/d4dd00392f) · Digital Discovery（RSC） · arXiv:2411.04323 | 在 Crystal-GFN 仅生成空间群/组成/晶格参数、不含原子坐标的基础上，首次用 GFlowNet 生成**含原子坐标的完整晶体**：先建 flat GFlowNet 基线，再提出分层且对称性感知的 SHAFT，按空间群→晶格→原子逐层分解为子目标，缩短长程轨迹并利用对称性；在有效性、稳定性、多样性上超过 flat GFN、CDVAE 与 DiffCSP。 | **P1**：完整晶体生成的代表作；与 A06（Crystal-GFN）、T38（对称性感知 GFN）对读，重点看分层子目标如何缓解长轨迹信用分配，并核对 validity/stability 指标口径 <sub>来源 E16</sub> |
| **N070** | [Catalyst GFlowNet for electrocatalyst design: A hydrogen evolution reaction case study](https://arxiv.org/abs/2510.02142) · 预印本 arXiv:2510.02142（2025-10） | 首个把 GFlowNet 用于**电催化剂设计**：基于 Crystal-GFN 构造周期性晶体并切出催化表面，用 FAENet(GNN) 预测吸附能、并做 ML 结构弛豫，以形成能与吸附能构造奖励，从而采样一组多样催化剂候选而非单一最优。概念验证针对析氢反应(HER)，成功重新发现已知最优催化剂铂(Pt)。 | **P2**：催化剂车道开山但仍是 proof-of-concept；注意当前仅限少量单元素组成、单一吸附质(H)、小晶胞；评估应看分布覆盖与多样性，而非只看"能否找到 Pt" <sub>来源 E16</sub> |
| **N071** | [Collective Variable Free Transition Path Sampling with Generative Flow Network（TPS-GFN）](https://arxiv.org/abs/2405.19961) · 预印本 arXiv:2405.19961 | 把分子亚稳态间的**过渡路径采样(TPS)**重构为对分子轨迹的摊销式能量采样，用 GFlowNet(one-step TB) 训练偏置势，无需依赖昂贵的集体变量(CV)，并借 replay buffer 做 off-policy 以避免模式坍缩。在 Alanine Dipeptide、Polyproline、Chignolin 上比既有无 CV 的 ML 方法生成更真实、更多样的过渡路径，属物理系统/稀有事件采样。 | **P2**：物理系统稀有事件采样的 GFN 切入点；关注 one-step TB 与相对 TB/log-variance 的关系（对读 T20 扩散采样、A22 相对 TB），务必核对最终版本标题与发表状态 <sub>来源 E16</sub> |

### 11.17 应用：多目标与条件生成（3 篇）

| 编号 | 论文与状态 | 简介 | 建议 |
|---|---|---|---|
| **N072** | [Goal-conditioned GFlowNets for Controllable Multi-Objective Molecular Design](https://arxiv.org/abs/2306.04620) · **ICML 2023 Workshop** | 针对偏好标量化在**凹 Pareto 前沿**上把解推向极端点的缺陷，改用"目标区域"硬约束条件化 GFN：仅当样本目标向量与目标方向的余弦相似度超过阈值（focus region 圆锥）才给奖励，并用可学习的表格式目标采样器 Tab-GS + hindsight replay 缓解奖励稀疏与不可行目标。实证在多目标分子任务上比偏好条件更均匀、更高熵地覆盖整条前沿。 | **P1**：与 A03(MOGFN-PC) 对读"软偏好 vs 硬目标区域"，做条件生成/凹前沿覆盖的必读；注意 workshop 状态与仅 3 seeds 的实证规模 <sub>来源 E17</sub> |
| **N073** | [Global-Order GFlowNets](https://arxiv.org/abs/2504.02968) · **预印本 2025** | 指出 OP-GFN 按批内 Pareto 支配施加的"局部序"会产生**相互冲突的训练目标**、导致优化不一致；提出用 global-rank 或最近邻等方式把局部偏序提升为与 Pareto 支配相容的"全局（弱）全序"，据此定义学习型奖励再训练 GFN。多基准上取得小而稳定的改进，是 OP-GFN 免标量化路线的直接修补。 | **P2**：紧随 OP-GFN 阅读，关注局部序冲突的构造性反例与全局化的计算代价；预印本结论需在可枚举任务上独立复核 <sub>来源 E17</sub> |
| **N074** | [Amortized Active Generation of Pareto Sets (A-GPS)](https://neurips.cc/virtual/2025/poster/116473) · **NeurIPS 2025 主会 Poster** | 面向在线离散黑盒多目标优化，学习一个支持"事后偏好条件"的 Pareto 集生成模型（建立在 VSD 之上）：用类别概率估计器 CPE 判别**非支配**关系来引导生成，并证明该非支配 CPE 隐式估计 hypervolume 改进概率（PHVI）；再以偏好方向向量做摊销变分推断，实现免重训的偏好可控采样，绕开显式 hypervolume 计算与标量化。 | **P1（相邻方法）**：作为评估 preference-conditioned GFN（A03/HN-GFN）跨偏好泛化与前沿覆盖的**非 GFN 强基线**，重点比 PHVI、hypervolume 与偏好对齐；勿记为 GFN 核心 <sub>来源 E17</sub> |

### 11.18 应用：安全、红队与对齐（2 篇）

| 编号 | 论文与状态 | 简介 | 建议 |
|---|---|---|---|
| **N075** | [GFlowNets with Human Feedback (GFlowHF)](https://arxiv.org/abs/2305.07036) · **ICLR 2023 Tiny Papers Track**（两页、**非归档短文、非主会**） | 已知最早的"GFlowNet + 人类反馈"框架：用人类评分拟合奖励，学习严格**正比于评分**的策略，而非像 RLHF 只追逐最高分，从而获得更强探索与更高多样性；并论证对约 10% 噪声标签比 RLHF 更鲁棒。仅为 Tiny Papers、实验为玩具环境，价值在**历史定位**与"分布匹配抗噪"的直觉，不宜作方法基线。 | **P2 / P3** <sub>来源 E19</sub> |
| **N076** | [Generating Attacks for LLMs with GFlowNets](https://arxiv.org/abs/2608.10171) · **预印本 2026-08** · arXiv:2608.10171 | 直接沿用 NEW-E19-1 的两阶段 GFlowNet+MLE 红队框架，把攻击提示生成扩展到**英语与土耳其语的多语言场景**，用 attacker / victim / evaluator 三模型闭环训练并给出定量鲁棒性分数。报告 GFN+MLE 相较仅 SFT 大幅提升成功率与毒性、降低两两余弦相似度（即更高多样性）。属工程性外推，低参量设置、相似度偏高，结论需谨慎。 | **P3** <sub>来源 E19</sub> |

### 11.19 应用：其他领域（3 篇）

| 编号 | 论文与状态 | 简介 | 建议 |
|---|---|---|---|
| **N077** | [Generative Flow Network for Listwise Recommendation (GFN4Rec)](https://arxiv.org/abs/2306.02239) · KDD 2023 | 面向**列表级推荐**：把"为用户生成一个物品列表"建模为 GFN 的逐步构造过程，用 log-scale reward matching 损失让列表生成概率与其整体效用对齐，并用自回归选择模型刻画列表内物品互相影响。相比交叉熵 ranking 更能缓解低多样性问题，在模拟在线环境与两个真实离线数据集上验证了主动探索时的质量—多样性权衡。它是 GFlowNet 进入**推荐系统**主会的最早代表。 | **P1**（推荐系统首选入口；与 A03 多目标、A19 组合优化对照，理解"多样候选集"价值） <sub>来源 E20</sub> |
| **N078** | [Ant Colony Sampling with GFlowNets for Combinatorial Optimization (GFACS)](https://proceedings.mlr.press/v258/kim25a.html) · AISTATS 2025 | 把 GFlowNet 与蚁群优化（ACO）分层结合：先用 GFN 摊销一个覆盖高奖励且多样解的**多峰先验**，再以 ACO 式并行随机搜索迭代更新为逼近近优解的后验。相比 RL 预训练得到的单峰先验，多峰先验更利于后续迭代改进；论文在 TSP、CVRP、OP、PCTSP 等**七类组合优化问题**上稳定超过经典 ACO 与 DeepACO。它代表"GFN 作为可学习先验驱动元启发式求解器"的路线。 | **P1**（组合优化求解器方向最强代表；与 A19"Let the Flows Tell"互补——A19 是通用图 CO 的 GFN 构造，GFACS 聚焦路由类问题 + ACO 混合） <sub>来源 E20</sub> |
| **N079** | [Discovery of Diverse and Realistic Financial Tail-Risk Using GFlowNets (GRID)](https://openreview.net/forum?id=YHiS8knV3s) · **投稿 ICLR 2026（under review） | 把**金融尾部风险情景生成**建模为在**连续状态空间**上逐步构造宏观经济轨迹：每步由 GFN 预测动作分布参数（可用高斯/Beta 混合等灵活分布族）并采样转移，终点由预测 oracle 给标量 reward，用 flow-matching 训练。相比 SMC（多样但不聚焦高风险）与深度 RL（聚焦但收敛到少数模式），GRID 在真实金融数据上同时提升情景**多样性与真实性**，捕捉非线性依赖。 | **P2**（投稿中，需待接收状态；金融方向与 A31 AlphaSAGE 的 alpha 因子挖掘互补——本篇是宏观风险情景生成） <sub>来源 E20</sub> |

### 11.20 生态：评测基准与软件（2 篇）

| 编号 | 论文与状态 | 简介 | 建议 |
|---|---|---|---|
| **N080** | [Evaluating Generalization in GFlowNets for Molecule Design](https://openreview.net/forum?id=JFSaHKNZ35b) · ICLR 2022 MLDD Workshop（**奠基作，2022**） | 针对分子设计"生成一批多样高分候选"的目标，系统比较多种候选评价指标，提出 **TopKDiverse**（Tanimoto 多样性约束下取 top-K 平均分）刻画下游搜索性能，并发现 **GFNEvalS**（对齐采样概率与目标奖励分布）比 flow error / top-k 更能预测泛化。是 GFlowNet 专属"多样性/覆盖"指标的历史起点，被后续广泛沿用。 | **P2（奠基参照）**；作为 mode coverage/diversity 指标的源头，与评测陷阱一并读 <sub>来源 E18</sub> |
| **N081** | [Benchmarking GFlowNets against MCMC: The Role of Peak Sharpness and Dimensionality](https://jac.ut.ac.ir/article_106220_565b5b56aeb6a2e1813f39d7ffebcd62.pdf) · J. of Algorithms and Computation 57（2） | 在 HyperGrid 上系统对比 TB/DB/FM 与 Metropolis–Hastings 对"奖励地形几何（峰宽/尖锐）"与维度的敏感性，挑战"学习式采样器普遍更优"的默认叙事：尖峰/低维/近 Dirac 目标上 MCMC 更稳且快约 500×，宽峰多模态时 GFlowNet-DB 可多发现约 4.75× 模式。结论是"该不该上 GFlowNet 取决于地形"。 | **P2（方向性证据）**；venue 小众、规模仅 HyperGrid，宜作定性参考，须与 Sendera/Blessing 的大规模横评交叉验证 <sub>来源 E18</sub> |


### 11.21 审查流水线补录并入（18 篇）

> 本项目另有一条并行审查流水线（`review/R01–R10` + `docs/GFLOWNET_SUPPLEMENT_PAPERS_CN.md`，以 T58–T63 / A36–A54 编号）。两批扩充经比对：补录 25 篇中 7 篇与本节重复（已合并），**18 篇为本节此前所缺**，在此并入并统一为 N 编号。原编号对照见文末。

| 编号 | 论文与状态 | 简介 | 建议 |
|---|---|---|---|
| **N082** | [Path-dependent Discrete Amortized Inference](https://arxiv.org/abs/2608.08644) · ICML 2026 **Oral** | 证明 GFlowNet 类离散摊销采样器的 Markov 假设会阻碍训练中的信号传播，且因 state aliasing 灾难性地限制可表达的终止分布；提出用可学习潜动力系统提升 MDP，使策略依赖整条历史轨迹（路径依赖），并把现有摊销采样器训练算法可证地扩展到该非 Markov 设定，实验显示更快收敛与更好探索。它直接挑战 T02 奠基框架中的 Markovian flow 前提，是 2026 表达力理论的新支柱；与 T62 同一首尾作者团队（da Silva、Lahlou）。 | **P0 精读** <sub>来源 补录T61</sub> |
| **N083** | [Particle GFlowNets: Rethinking Generative Marginalization Models](https://proceedings.mlr.press/v337/silva26a.html) · UAI 2026 主会（PMLR 337:6366–6383） | 证明生成边缘化模型（MaM，为任意阶自回归离散建模同时学习边缘与条件概率）与 GFlowNet 等价——此前两者被视为不同范式；进而把 MaM 的采样策略推广到非自回归生成过程，用由 Gelman–Rubin 统计量导出的自动准则对持久 Gibbs 采样器做全状态重启（rejuvenation），显著加速大组合空间中的训练收敛。经全卷检索，它是 UAI 2026 有且仅有的一篇 GFlowNet 核心论文，直接修复 §5 审计表的 UAI 空缺。 | **P1** <sub>来源 补录T62</sub> |
| **N084** | [Interpreting GFlowNets for Drug Discovery: What Probes Can and Cannot Show](https://arxiv.org/abs/2511.19264) · NeurIPS 2025 WiML Workshop / MoML 2025 | 对 SynFlowNet（合成感知分子 GFN）做控制严谨的可解释性研究：结合梯度显著性、反事实编辑、因子分析与稀疏自编码器，并用打乱标签、未训练同构网络等对照。关键结论是许多"可解码性"其实来自图架构与原子特征而非策略学习——高探针分数不等于学到化学，给出可复用的分子模型解释对照协议。 | **P2** <sub>来源 补录A41</sub> |
| **N085** | [Designing the Haystack: Programmable Chemical Space for Generative Molecular Discovery (SpaceGFN)](https://arxiv.org/abs/2603.00614) · 预印本 2026 | 把"化学空间本身"提升为可编程对象：用户用构件与反应规则显式构造化学/合成自洽的分子宇宙，GFN 在其中做性质偏置采样。Discovery 模式支持类天然产物空间与进化启发空间，Editing 模式支持反应一致的先导优化；在 96 个药靶上兼顾优化性能与合成约束下的结构多样性。 | **P2** <sub>来源 补录A43</sub> |
| **N086** | [Curriculum-Augmented GFlowNets for mRNA Sequence Generation (CAGFN)](https://arxiv.org/abs/2510.03811) · 预印本 2025 | 针对 mRNA 设计的稀疏长时程奖励与多目标权衡，把课程学习与多目标 GFN 结合：按序列长度渐进的课程从易到难引导探索，并提供新的 mRNA 设计环境。相比无课程的随机采样 GFN，Pareto 性能与生物合理性更好、更快达到高质量解并能泛化到分布外序列。 | **P2** <sub>来源 补录A45</sub> |
| **N087** | [FlowPipe: LLM-Enhanced Conditional GFlowNets for Data Preparation Pipeline Construction](https://arxiv.org/abs/2606.24679) · SIGMOD 2027（作者标注已接收） | 把数据准备流水线合成表述为 DAG 上的条件概率流生成，用**条件 GFlowNet + TB** 把终端验证奖励连回早期决策；再用 FiLM 深度语义调制注入 LLM 逻辑先验，并把失败感知写入流目标以避开无效状态。74 个真实数据集上较 SOTA 平均准确率 +11.96%、收敛快 12.5×。venue 已从预印本升级为 SIGMOD 2027。 | **P2** <sub>来源 补录A49</sub> |
| **N088** | [Generative Learning for Quantum Measurement Design (FlowMeas)](https://arxiv.org/abs/2608.11396) · 预印本 2026-08 | 把有限 shot 预算下的量子测量协议设计重述为生成学习问题，用 GFN 直接采样浅层 Clifford 测量电路集合并满足硬件约束。零纠缠深度即匹配/超过主流 product-measurement，允许 1–2 层纠缠门时能量估计误差再降至多 27%；策略可跨相关哈密顿量复用，规模达 20-qubit（并演示 54-qubit 编码模型）。 | **P2** <sub>来源 补录A38</sub> |
| **N089** | [GFlowNets for Model Adaptation in Digital Twins of Natural Systems](https://arxiv.org/abs/2604.20707) · 预印本 2026（Under Review） | 把自然系统数字孪生的模型适应视为模拟推断问题：稀疏间接观测常不能唯一标定参数，GFN 在完整模拟器配置上按"模拟-观测一致度"奖励比例采样多个可信参数化。以受控环境农业番茄机理模型为例，能恢复适应景观的主导区域并在不确定性下保留多个合理配置。 | **P2** <sub>来源 补录A46</sub> |
| **N090** | [Exploration through Generation: Applying GFlowNets to Structured Search](https://arxiv.org/abs/2510.21886) · 预印本 2025 | 教学式地把 GFN 用 TB loss 应用到三个经典图优化问题——旅行商(TSP)、最小生成树(MST)、最短路——顺序选边/选点/选城市构造解。生成解与经典算法（Dijkstra、Kruskal、精确 TSP 求解器）的最优解一致，卖点是通过训练摊销计算以换取对更大实例的可扩展性。 | **P2** <sub>来源 补录A47</sub> |
| **N091** | [Structurally Valid Log Generation using FSM-GFlowNets](https://arxiv.org/abs/2510.26197) · 预印本 2025 | 把有限状态机(FSM)与 GFN 结合生成结构合法且行为多样的合成事件日志：FSM 由专家轨迹推导、编码领域规则，GFN 用 flow matching + FSM 合规/统计保真的混合奖励、经动态动作掩码与引导采样保证句法有效。UI 交互日志上 KL/χ² 显著优于 GPT-4o 与 Gemini，下游意图分类用纯合成日志亦具竞争力。 | **P2** <sub>来源 补录A52</sub> |
| **N092** | [Transform-Invariant Generative Ray Path Sampling for Efficient Radio Propagation Modeling](https://arxiv.org/abs/2603.01655) · 预印本 2026 | 用 GFN 替代射线追踪的穷举路径搜索做智能采样，缓解高阶交互下有效路径稀少导致的稀疏奖励：经验回放缓冲留存稀有有效路径、均匀探索策略防过拟合、基于物理的动作掩码先滤除不可能路径。理想街道峡谷上比穷举 GPU 快 10×/CPU 快 100× 并保持高覆盖精度（真实曼哈顿几何的 OOD 泛化仍需增强）。**此即 R07 标注"需人工确认 GFN 核心性"的条目——已确认 GFN 是其核心采样器。** | **P2** <sub>来源 补录A54</sub> |
| **N093** | [CounterFlowNet: From Minimal Changes to Meaningful Counterfactual Explanations](https://arxiv.org/abs/2602.17244) · 预印本 2026 | 把反事实解释(CF)生成建模为条件 GFlowNet 上的顺序特征修改，按用户自定义奖励（有效性、稀疏性、邻近性、可信度）成比例采样多样 CF。顺序建模天然产生高稀疏编辑，统一动作空间同时支持连续与类别特征，且可在推断时用 action masking 施加不可变/单调等可执行性约束而无需重训。 | **P2** <sub>来源 补录A40</sub> |
| **N094** | [AlphaG-OPD: Reliability-Gated Sibling Counterfactuals for On-Policy Distillation in Symbolic Alpha Factor Discovery](https://arxiv.org/abs/2608.01303) · 预印本 2026-08 | 在符号 alpha 因子挖掘中，GFN 保持对完整表达式的多样奖励比例分布，但其轨迹级目标不比较中间状态未选的兄弟动作。本文用"可靠性门控的兄弟反事实"把终端因子评估转成局部动作指导（何处教/何者可靠可教/教多强多久），终端奖励、TB、后向策略与语法均不变。CSI300/500/1000 与标普 500 跨市场表现稳健（A31 AlphaSAGE 同赛道）。 | **P2** <sub>来源 补录A50</sub> |
| **N095** | [TILDE: TILt-based Distributional Erasure for Concept Unlearning](https://arxiv.org/abs/2607.06432) · 预印本 2026-07 | 文生图 diffusion 的概念遗忘：把遗忘表述为"在遗忘约束下与预训练模型偏差最小的条件分布"这一分布对齐目标，用**残差 ∇-GFlowNet** 学习由遗忘能量相对预训练模型诱导的 score 修正。在物体、艺术风格、角色上兼顾强遗忘与更好的保留度和分布保真（A23 Nabla-GFlowNet、A26 EraseFlow 同线）。 | **P2** <sub>来源 补录A51</sub> |
| **N096** | [Meta-Learning-Driven GFlowNets for 3D Directional Modulation in Mobile Wireless Systems (Meta-GFlowNet)](https://arxiv.org/abs/2511.06188) · 预印本 2025（投 IEEE ICC 2026） | 在时间调制智能反射面(TM-IRS)物理层安全设计中，用 MAML 式元学习让 GFN 快速适应移动用户方向变化：内层轨迹平衡更新 + 外层元更新学到跨方向的通用先验，无需标注数据（用 GFN 学到的奖励与真实和速率的伪监督一致性目标）。比重训 GFN 适应更快、保密性能更高。 | **P2** <sub>来源 补录A53</sub> |
| **N097** | [Improving LLM-Based Recommenders with Conservative Generative Flow Networks](https://icml.cc/virtual/2026/poster/65235) · ICML 2026 主会 | 研究离线 LLM 推荐：学习被限制在固定日志数据集上，数据集诱导的 token 前缀 DAG 只有部分转移支持，此时朴素 SubTB 不可辨识、会把概率质量任意分配到无支持区域；论文形式化三类失败来源——流高估、前向质量泄漏、后向补偿——并提出 CFlower：显式惩罚数据支持外前向流质量的保守 SubTB 目标，配合限制在数据 DAG 上的 on-policy 采样，在三个 Amazon 数据集上改善分布匹配与准确率–曝光权衡。它把 A24（COFlowNet）面对的 offline 支持失配问题推进到 SubTB 可辨识性层面，其保守目标对所有离线 GFN 都有诊断价值。 | **P1** <sub>来源 补录A37</sub> |
| **N098** | [WINFlowNets: Warm-up Integrated Training of GFlowNets for Robotics and Machine Fault Adaptation](https://arxiv.org/abs/2603.17301) · 预印本 2026 | 针对连续场景 CFlowNets 依赖预训练 retrieval 网络、难以适应动态机器人环境的问题，提出 flow 网络与 retrieval 网络"预热+共享 replay 协同训练"框架。仿真机器人任务上平均奖励与训练稳定性超过 CFlowNets 与主流 RL，并在故障环境下展现少样本快速自适应能力。 | **P2** <sub>来源 补录A42</sub> |
| **N099** | [torchgfn: A PyTorch GFlowNet Library](https://arxiv.org/abs/2305.14594) · 预印本 | torchgfn 库的配套论文：核心贡献是把环境、神经网络模块与训练目标解耦为可互换组件的模块化架构，提供简洁 API 与复现、统一多个已发表结果的示例，是在标准基准实现上测试新 loss/新策略的参考协议。§7 已以 P0 收录其代码仓库，本卡片补论文本体。注意：截至 2026-08-14，JMLR 官网（v26/v27 卷目录及 MLOSS 轨道）均无该文记录，引用时应按"预印本 + Workshop"口径，待 JMLR 正式卷期页面出现后再升级状态。 | **P2** <sub>来源 补录T63</sub> |

**编号对照（补录 → 本节）**

| 补录编号 | 本节编号 | 论文 |
|---|---|---|
| T61 | **N082** | Path-dependent Discrete Amortized Inference |
| T62 | **N083** | Particle GFlowNets: Rethinking Generative Marginalization Models |
| A41 | **N084** | Interpreting GFlowNets for Drug Discovery: What Probes Can and Cannot  |
| A43 | **N085** | Designing the Haystack: Programmable Chemical Space for Generative Mol |
| A45 | **N086** | Curriculum-Augmented GFlowNets for mRNA Sequence Generation (CAGFN) |
| A49 | **N087** | FlowPipe: LLM-Enhanced Conditional GFlowNets for Data Preparation Pipe |
| A38 | **N088** | Generative Learning for Quantum Measurement Design (FlowMeas) |
| A46 | **N089** | GFlowNets for Model Adaptation in Digital Twins of Natural Systems |
| A47 | **N090** | Exploration through Generation: Applying GFlowNets to Structured Searc |
| A52 | **N091** | Structurally Valid Log Generation using FSM-GFlowNets |
| A54 | **N092** | Transform-Invariant Generative Ray Path Sampling for Efficient Radio P |
| A40 | **N093** | CounterFlowNet: From Minimal Changes to Meaningful Counterfactual Expl |
| A50 | **N094** | AlphaG-OPD: Reliability-Gated Sibling Counterfactuals for On-Policy Di |
| A51 | **N095** | TILDE: TILt-based Distributional Erasure for Concept Unlearning |
| A53 | **N096** | Meta-Learning-Driven GFlowNets for 3D Directional Modulation in Mobile |
| A37 | **N097** | Improving LLM-Based Recommenders with Conservative Generative Flow Net |
| A42 | **N098** | WINFlowNets: Warm-up Integrated Training of GFlowNets for Robotics and |
| T63 | **N099** | torchgfn: A PyTorch GFlowNet Library |

**补录中与本节重复、已合并的 7 篇**

| 补录编号 | 本节编号 | 论文 |
|---|---|---|
| T58 | **N010** | Improved Off-Policy Training of Diffusion Samplers |
| T59 | **N002** | From Discrete-Time Policies to Continuous-Time Diffusion Samplers: Asy |
| T60 | **N024** | Adaptive Teachers for Amortized Samplers |
| A36 | **N063** | Learning Diverse Attacks on Large Language Models for Robust Red-Teami |
| A39 | **N077** | Generative Flow Network for Listwise Recommendation (GFN4Rec) |
| A44 | **N070** | Catalyst GFlowNet for Electrocatalyst Design |
| A48 | **N068** | Latent Thought Flow: Efficient Latent Reasoning in LLMs (LTF) |

### 11.22 2026-08 追补（检索截止 2026-08-25）

对 2026-08-01 起的 arXiv 做了一轮增量检索（共命中 7 篇），去掉已收录与非核心者后追补 1 篇。

| 编号 | 论文与状态 | 简介 | 建议 |
|---|---|---|---|
| **N100** | [IFlowNets: Extending Generative Samplers to Learn Strategies in Incomplete Information Games](https://arxiv.org/abs/2608.05422) · NeurIPS 2025 Workshop (Dynamics at the Frontiers of Optimization, Sampling, and Games) | 把 Adversarial Flow Networks (AFlowNets) 推广到**不完全信息博弈**：证明完全信息博弈下已确立的 flow 约束在此设定下无法给出合法密度（即合法策略）与合法训练目标，提出 Information Flow Networks 修复该问题并严格推广 AFlowNets；三个标准博弈环境上与 Outcome Sampling CFR 相当或更好。 | **P2**：与 T18（随机环境与二人零和博弈的 expected flow）配读，是把 GFN 推进到博弈论的最新一步 |

**已检索但不纳入**（按 §5 审计口径：GFlowNet 须为核心模型/训练目标/主要分析对象）

| arXiv | 标题 | 不纳入理由 |
|---|---|---|
| 2608.01789 | Towards Autonomous Formulaic Alpha Discovery: An Evolutionary Computation Perspective | 演化计算视角的**综述**，GFlowNet 仅作为并列技术之一被提及（cs.NE） |
| 2608.05314 | Machine learning for sample-based quantum diagonalization | 量子化学综述，摘要未涉及 GFlowNet |

其余 4 篇（2608.11396、2608.10171、2608.03967、2608.01303）已分别收录为 N088、N066 一带、N005、N094。

### 11.23 顶会查全补录（6 篇主会）

顶会审计复核（C07）独立检索出 11 篇 P0 + 5 篇 P1 候选的漏收主会论文，其中多数已在前面各车道收入；本表补齐剩余 6 篇。**漏收根因**：原审计只按标题含 “GFlowNet” 检索，标题不含该词但以 GFlowNet 为核心的论文会被整批跳过。

| 编号 | 论文与状态 | 简介 | 建议 |
|---|---|---|---|
| **N101** | [Scalable and Cost-Efficient de Novo Template-Based Molecular Generation](https://openreview.net/forum?id=zssWxiiJZ1) · NeurIPS 2025 主会 · arXiv:2506.19865 | 直面 template-based GFlowNet 的三个核心难题：最小化合成成本、扩展到大规模 building block 库、有效利用小片段集；提出 Recursive Cost Guidance 后向策略等机制。它是 RxnFlow（N056）之后可合成分子生成这条线的规模化续作。 | **P1**：做可合成分子生成必读，与 A09/A11/N056 连读 |
| **N102** | [Adversarial Generative Flow Network for Solving Vehicle Routing Problems](https://openreview.net/forum?id=tBom4xOW1H) · ICLR 2025 主会 · arXiv:2503.01931 | 针对 VRP 构造式神经求解器普遍用 Transformer、扩展性受限且解多样性不足的问题，提出对抗式 GFlowNet（AGFN）架构。它与 T41（Hybrid-Balance，同为 VRP）构成组合优化线的一对，且是 T41 所嵌入的宿主求解器之一。 | **P1**：与 T41、N078（GFACS）合读，理解 GFN 在组合优化中的三种用法 |
| **N103** | [Discrete Compositional Generation via General Soft Operators and Robust Reinforcement Learning](https://openreview.net/forum?id=MGWk2tEgLW) · ICLR 2026 主会 · arXiv:2506.17007 | 指出各类熵正则方法（含 GFlowNet）在代理奖励下的过度保守问题，提出用**通用 soft operator** 统一并推广离散组合生成，并以鲁棒 RL 视角给出理论刻画。它把 GFlowNet 放进一个更大的算子族里，是 T14/T15 等价性研究的最新推进。 | **P0 精读**：理解 GFN 与熵正则 RL 关系的当前最一般框架 |
| **N104** | [On Scalable and Efficient Training of Diffusion Samplers](https://openreview.net/forum?id=Xzabk07lao) · NeurIPS 2025 主会 · arXiv:2505.19552 | 针对能量评估昂贵、采样空间高维时 diffusion sampler 难以扩展的问题，提出可扩展且样本高效的训练框架。与 N010（NeurIPS 2024 统一基准）同属"用 GFlowNet 目标训练扩散采样器"这一谱系，是该线在扩展性上的续作。 | **P1**：与 T20、N010 构成连续采样器线的三步递进 |
| **N105** | [Reinforced Sequential Monte Carlo for Amortised Sampling](https://openreview.net/forum?id=DWaToCuNwa) · ICML 2026 主会（Spotlight） · arXiv:2510.11711 | 建立 SMC 与最大熵 RL 训练的神经序列采样器之间的联系——学到的策略与值函数正好给出 SMC 的 proposal kernel 与 twist function，从而把摊销方法与粒子方法结合。它给 GFlowNet 与经典蒙特卡洛的接口提供了明确的形式化。 | **P1**：做 VI/MCMC/SMC 统一视角必读，与 N022、T17 连读 |
| **N106** | [Improved Off-policy Reinforcement Learning in Biological Sequence Design](https://openreview.net/forum?id=0TY5lhhdZm) · ICML 2025 主会 · arXiv:2410.04461 | 提出 \(\delta\)-Conservative Search：把 off-policy 搜索限制在可信区域内，缓解代理模型在分布外输入上的 misspecification，覆盖 DNA/RNA/蛋白/肽。它补齐了 A02（2022 原型）之后"序列 GFN 的 proxy 鲁棒与 off-policy 训练"这条线的主会空档。 | **P1**：生物序列设计方向的必读续作，与 A02、A10 连读 |

### 11.24 趋势核实补录（2026-09-01，6 篇）

三份趋势报告（`insights/`）在 web 检索与 GitHub/PyPI 实测中遇到 17 篇论文，逐条对照本目录后确认 11 篇已收录、6 篇为目录外，此表补齐这 6 篇。**漏收根因**：前 23 节的检索都以 GFlowNet 为关键词，而竞品方法（力引导采样、Jacobian 估计）与邻域综述（GRPO）不含该词；bioRxiv 与库论文也不在原检索源里。

| 编号 | 论文与状态 | 简介 | 建议 |
|---|---|---|---|
| **N107** | [Drifting to Boltzmann: Million-Fold Acceleration in Boltzmann Sampling with Force-Guided Drifting](https://arxiv.org/abs/2603.05527) · 预印本 2026-03 | 把力场信息引入采样位移（Force-Interpolated Drifting）与邻居重加权（Force-Aligned Kernel）。MD17 Ethanol 上单步生成即达 h(r) TVD 0.139、W2 0.031、键稳定性 97.5%，比迭代方法快约 2000 倍；并指出标准 h(r) 指标会被非键 H-H 对主导而误导，提出 per-type TVD 与 Bond MAE。 | 选读：竞品定位。读它是为了知道 GFlowNet 系在「有力场可用」的物理场景已不占优 <sub>来源 insights/trends_neighbors.md</sub> |
| **N108** | [Flow Perturbation++: Multi-Step Unbiased Jacobian Estimation for High-Dimensional Boltzmann Sampling](https://arxiv.org/abs/2601.21177) · 预印本 2026-01 | 为 SMC 流水线提供无偏多步 Jacobian 估计。1000 维 GMM 上模态权重估计 0.256±0.027（真值 0.25），成本与单步 FP 相当，而各 Hutchinson 变体或有严重偏差或模态坍缩。 | 选读：不含 GFlowNet 成分，仅作高维采样赛道的性能参照 <sub>来源 insights/trends_neighbors.md</sub> |
| **N109** | [Generating Structurally Diverse Therapeutic Peptides with GFlowNet](https://www.biorxiv.org/content/10.64898/2026.01.05.697258v3) · bioRxiv 2026-01 | 与带显式多样性惩罚的 GRPO-D（λ=0.15）正面对比：GFlowNet 二肽采样均匀度高 5.4 倍、奖励方差低 1.9 倍、重复序列少 3.9 倍。压力测试是关键——去掉奖励中的熵门控后 GRPO-D 的 1000 个样本 100% 含同一三肽模式（RMMRMMRMM），而 GFlowNet 多样性保持 0.937。 | **优先读**：目前「奖励成比例采样对奖励设计缺陷更鲁棒」最硬的实验证据之一；注意是 bioRxiv 预印本，未经同行评审 <sub>来源 insights/trends_neighbors.md</sub> |
| **N110** | [Advances in GRPO for Generation Models: A Survey](https://arxiv.org/abs/2603.06623) · 预印本 2026-03 | GRPO 及其后续发展的综述，把 diversity preservation 单列为一个研究方向（DiverseGRPO 用 Vendi Score 提升 13%–18%、OSCAR 做训练无关的隐空间多样性增强）。 | 选读：从反面确认「模式坍缩是 GRPO 的结构性弱点而非实现问题」，写 related work 时有用 <sub>来源 insights/trends_neighbors.md</sub> |
| **N111** | [AbFlowNet: Optimizing Antibody-Antigen Binding Energy via Diffusion-GFlowNet Fusion](https://arxiv.org/abs/2505.12358) · 预印本 2025-05 | 把扩散去噪步当作 GFlowNet 的状态转移来做抗体设计，用结合自由能作奖励。 | 选读：「扩散骨架 + GFlowNet 目标」在生物大分子上的实例，原分子应用全景文档缺此条 <sub>来源 insights/trends_applications.md</sub> |
| **N112** | [gfnx: Fast and Scalable Library for Generative Flow Networks in JAX](https://arxiv.org/abs/2511.16592) · 预印本 2025-11 | JAX 实现的 GFlowNet 库：环境、reward、指标全部 JIT-able，每个环境配 CleanRL 风格单文件基线；声明 CPU 序列生成最高 55 倍、GPU 贝叶斯结构学习最高 80 倍加速，覆盖 8 个环境，明确以 standardize empirical evaluation 为目标。 | 选读：做大批量性能实验时值得试，但**工程成熟度需保留意见**——PyPI 停在 0.0.1（2025-11-16 后未发新版）、仅 2 个 fork，目前是作者自用加论文配套 <sub>来源 insights/trends_applications.md §5.3</sub> |

### 11.25 失效边界专题补录（2026-09-01，1 篇）

写 `insights/trends_failure_modes.md`（GFlowNet 失效边界专题）时，用「失效/崩溃/欠拟合/反例」等词扫描本目录 212 篇简介只命中 1 篇（T07），说明这类证据在原目录里被系统性埋没在各篇的实验章节与 limitations 里。该专题把散落证据按失效机制归类，并补录下列漏收论文。**漏收根因**：IEEE 会议（BIBM）不在原检索源内。

| 编号 | 论文与状态 | 简介 | 建议 |
|---|---|---|---|
| **N113** | [Fixing Truncation-Induced Mode Collapse in GFlowNets via Pruning Loss](https://doi.org/10.1109/bibm66473.2025.11356156) · IEEE BIBM 2025 | 把 mode collapse 的根因定位到「人为轨迹截断产生的强制终止态」：它们违反流守恒的边界约束，造成流泄漏并把生成偏向最大长度轨迹。提出 Pruning Loss 要求强制终止处的 sink flow 与总出流都等于奖励，理论上恢复截断空间的流守恒且保证梯度不消失。实验对照极干净：稀疏奖励（kinase 靶点）上大幅超过标准目标，稠密奖励（drug-likeness）上各法相当——精确说明流泄漏只在稀疏奖励下限制性能。 | **优先读**：论断「修正强制终止处的边界约束比改进平衡方程更根本」若成立，则 TB→SubTB→f-TB 这条主线在 mode collapse 上一直在治标；与 T07、T51 连读 <sub>来源 insights/trends_failure_modes.md §2</sub> |

另有一处元数据补全：**T32** 的预印本题名与目录记录的会议题名不同（arXiv:2411.05899 题为 *Analyzing GFlowNets: Limitations, Countermeasures, and Assessment*，ICLR 2025 版题为 *When Do GFlowNets Learn the Right Distribution?*）。按题名检索容易误判为两篇，已在 status 字段标注。该文的 FCS 指标与指标否证结论见 `insights/trends_failure_modes.md` §6。

### 11.26 应用方向覆盖审计补录（2026-09-01，4 篇）

按 12 个应用方向对目录做覆盖统计（脚本化关键词扫描），发现两个方向为 **0 篇**：芯片/EDA 与 NAS/超参搜索。前者尤其可疑——`torchgfn` 内置了 `chip_design` 环境并带 `plc_client.py` 接口，说明社区在做，但目录检索不到论文。顺着这两个缺口补录下列 4 篇。**漏收根因**：原检索以 GFlowNet 为关键词，而 (a) 芯片布局的 SOTA 用的是 flow matching 而非 GFlowNet，标题不含关键词；(b) NAS 与主动学习方向的工作多发在 workshop 与领域会议，不在原检索源内。

| 编号 | 论文与状态 | 简介 | 建议 |
|---|---|---|---|
| **N114** | [FlowPlace: Flow Matching for Chip Placement](https://arxiv.org/abs/2604.23658) · 预印本 2026-04 | 芯片宏单元布局的 flow matching 生成式布局器：mask 引导的合成数据（注入模块化与边界感知先验）+ 确定性流轨迹 + 硬约束引导采样（把消除重叠的投影算子嵌进生成轨迹）。在 ICCAD 2015 Contest C 与 OpenROAD 基准上全面超过解析式 DREAMPlace 4.1、RL 式 MaskPlace/EfficientPlace 与扩散式 ChipDiffusion，零样本推理数秒出图。 | 选读：**警示性对照**。芯片布局是「组合空间 + 多样解有价值」的天然 GFlowNet 场景，但 SOTA 由 flow matching 占据。想在 EDA 方向做 GFlowNet 的人应先解释「为什么不用确定性流轨迹」<sub>来源 应用覆盖审计</sub> |
| **N115** | [Generative flow induced neural architecture search (FWNO)](https://arxiv.org/abs/2405.06910) · 预印本 2024-05 | 把 GFlowNet 的按奖励比例生成用于神经算子架构搜索：串联一组网络逐个采样超参，终端网络是 wavelet neural operator 本体，奖励取负验证损失的指数，用流一致性损失训练。 | 选读：GFlowNet 在 NAS/HPO 方向少见的完整实例，本目录该方向原为空白 <sub>来源 应用覆盖审计</sub> |
| **N116** | [BatchGFN: Generative Flow Networks for Batch Active Learning](https://arxiv.org/abs/2306.15058) · ICML 2023 SPIGM Workshop | 用 GFlowNet 按批次奖励（批次与模型参数的联合互信息 JMI）比例采样数据点集合。核心卖点是摊销掉批次感知算法的组合复杂度——推理时每点一次前向即可采到近最优效用批次，不需贪心近似。SubTB + forward-looking 参数化 + 集合置换不变架构。 | 选读：思路干净但只在玩具回归上验证，作者自陈需摊销跨轮训练才能上真实任务 <sub>来源 应用覆盖审计</sub> |
| **N117** | [Why Pool When You Can Flow? Active Learning with GFlowNets](https://ai4d3.github.io/2025/papers/31_Why_Pool_When_You_Can_Flow_.pdf) · NeurIPS 2025 AI4D3 Workshop | 把主动学习从「在固定池里挑」改成「直接生成」：BALD-GFlowNet 用互信息作奖励、RTB 损失训练，初始池 174 万的分子任务上做 30 轮采集（每轮 100 个），代理为 MoLFormer + MC Dropout。 | 选读：与 N116 是同一思路的两代（前者受池约束、后者放弃池化）；附录给全了超参，便于复现 <sub>来源 应用覆盖审计</sub> |

**方法论备注**：本次补录暴露了一个查重陷阱——A19《Let the Flows Tell: Solving Graph Combinatorial **Problems**...》与原论文题名《...Combinatorial **Optimization** Problems...》差一个词，按精确标题查重会误判为漏收。已固化 `scripts/check_duplicate.py`（arXiv/DOI 号命中 + 标题模糊相似度，阈值 0.75），补录前应先跑它。

### 11.26 应用覆盖二次审计补录（2026-09-02，1 篇）

用 `scripts/audit_coverage.py` 复扫 12 个应用方向，金融/量化仍只有 2 篇（N079 投稿中、N094 预印本），与该方向的工业动机不符。深挖后发现漏掉了一篇**主会**论文。

| 编号 | 论文与状态 | 简介 | 建议 |
|---|---|---|---|
| **N118** | [AlphaSAGE: Structure-Aware Alpha Mining via GFlowNets for Robust Exploration](https://arxiv.org/abs/2509.25055) · ICLR 2026 主会 | 把公式化 alpha 挖掘从 RL 换成 GFlowNet，直指 RL 路线的三个结构性问题：奖励稀疏（只有完整公式才有反馈）、把数学表达式当序列处理丢掉结构、以及最大化期望回报天然收敛到单一模态——而量化实践要的恰是一组互不相关的 alpha。三项设计：RGCN 结构感知编码器（公式当 AST 图而非 token 序列）、TB 目标训练的生成器、$R_{IC}$（预测能力）$+ R_{SA}$（结构对齐）$+ R_{NOV}$（与参考集低相关）三段稠密奖励加熵正则。评测用 IC/ICIR/RIC/RICIR 与年化收益/最大回撤/夏普，数据覆盖 CSI300 等三个子集；组合阶段沿用 AlphaForge 的动态重选线性组合。 | **优先读**：目前金融方向唯一的主会论文，且是「奖励成比例采样打败奖励最大化」这个论点在非分子领域最完整的实例；有官方代码（github.com/BerkinChen/AlphaSAGE） <sub>来源 insights/trends_applications.md 二次审计</sub> |

**漏收根因**（与 §11.24、§11.25 的根因不同，值得单列）：这篇标题含 GFlowNets、发表在 ICLR 2026 主会、有官方代码——按任何一条常规检索都该命中。它被漏掉是因为原检索的关键词组合偏向「GFlowNet + 理论/分子/LLM」，没有单独扫过「GFlowNet + 金融/量化」这个组合。**教训是：按方向做覆盖审计（而非只按关键词检索）才能发现这类缺口**，这也是 `audit_coverage.py` 存在的理由。
