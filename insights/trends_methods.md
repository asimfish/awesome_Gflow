# GFlowNet 方法学前沿动态（2025-2026）

> 调研日期：2026-09-01 · 方法：web 检索（arXiv / OpenReview / 会议 proceedings）+ 本仓库 206 篇目录交叉核对 · 作者：awesome_Gflow 维护流水线

本报告只记录可核实的增量动态：每条给来源链接；检索不到的写「未检索到」。与本仓库目录（papers/current_papers.tsv，检索截止 2026-08-25）重叠的论文只在需要建立趋势链条时引用编号。

## 1. 训练目标演化：从「平衡族」走向「证书族」与「博弈族」

TB（T03）→ SubTB（T05）→ f-TB（T49）→ 散度族（T17）这条主线在 2026 年出现三个新分支。

分支一：带概率证书的稳定训练。[Stable GFlowNets with Probabilistic Guarantees](https://arxiv.org/abs/2605.01729)（arXiv 2605.01729，2026）首次把 TB loss 与全局 Total Variation 误差双向打通：一方面推出 loss 到 TV 的上界（其 Theorem 3.5），另一方面构造反例证明小 TV 误差不蕴含有界 loss——训练后期的 loss 尖峰可能是良性的。方法上引入「参考流」（reference flow）注入流守恒约束来封顶 loss 比率，并量化稳定性-保真度折中（Theorem 3.10/3.11）。这是把 T32（何时学对分布）的诊断视角推进成「可认证训练」的代表作。

分支二：对抗式探索目标。[Avoid What You Know: Divergent Trajectory Balance for GFlowNets](https://arxiv.org/abs/2602.17827)（arXiv 2602.17827，2026-02）提出 DTB（divergent TB）损失：训练一个探索策略去采样「当前 GFlowNet 低估的区域」，与主 GFlowNet 构成双人博弈（其 Propositions 3.4/3.9 刻画均衡）。实验显示在多样高奖励区域发现率上一致超过 curiosity 驱动的探索基线。

分支三：LLM 场景的前缀信用分配。[Rooted Absorbed Prefix Trajectory Balance (RapTB)](https://arxiv.org/abs/2603.00454)（arXiv 2603.00454，2026-03）针对 LLM-GFlowNet 微调中的两类可复现失败模式——前缀坍缩与长度偏置——把子轨迹监督锚定在根节点，用吸收后缀回传把终端奖励密集地传给中间前缀；配套的 SubM 子模回放刷新策略同时兼顾奖励与多样性。实验做到 32B 参数规模、两个架构家族。这是 TB 系目标第一次针对自回归 LLM 的失败模式做系统改造。

统一视角：[Controlling Exploration-Exploitation in GFlowNets via Markov Chain Perspectives](https://arxiv.org/abs/2602.01749)（arXiv 2602.01749，2026-02）证明现有 GFlowNet 目标隐含前向/后向策略的等权混合，据此提出 alpha-GFN：用单个超参 alpha 在 (0,1) 内混合 $P_F$ 与 $P_B$ 得到一族目标，标准目标是特例。这延续了 T11（Markov chain 视角）的概念路线，但给出了可调的探索-利用旋钮。

## 2. 收敛性理论：loss 到分布误差的链条基本闭合

2025 年之前的悬置问题是「TB loss 下降是否意味着分布靠近目标」。现在链条各环都有了结果：

- loss 到 KL：[Convergence guarantees of GFlowNets](https://openreview.net/forum?id=JmsgmkdIkk)（NeurIPS 2025 FPI Workshop，本目录 N001）证明最小化 TB loss 时学到分布与目标的 KL 被所最小化的量上界控制。
- loss 到 TV（含有限样本证书）：Stable GFlowNets（见上节）给出 loss 到 TV 上界与经轨迹采样的有限样本 TV 证书。
- 按目标分层的收敛速率：[Secrets of GFlowNets' Learning Behavior: A Theoretical Study](https://arxiv.org/abs/2505.02035)（arXiv 2505.02035，2025-05）给出 FM 目标 $\mathcal{O}(1/\sqrt{T})$、DB 目标 $\mathcal{O}(1/T^{1/3})$ 的收敛速率分层，样本复杂度 $\mathcal{O}(|S|L\log(|S|/\delta)/\epsilon^2)$（$|S|$ 状态空间大小、$L$ 最大轨迹长度），并证明不同目标诱导不同隐式正则：FM 偏最大熵、DB 偏前后向 KL 最小化、TB 偏路径长度效率；奖励噪声鲁棒性按 $R_{\min}^{-4}$ 缩放。
- 泛化界：[Generalization and Distributed Learning of GFlowNets](https://proceedings.iclr.cc/paper_files/paper/2025/file/000eba875068854d5ff003b1fa534cd6-Paper-Conference.pdf)（ICLR 2025）给出轨迹级泛化界：更长轨迹与更尖的目标分布使泛化更难，并提出分布式训练缓解。

我的判断：这条理论链的完成度已经足以支撑「选目标」的工程决策——短轨迹/需要熵覆盖选 FM 系，长轨迹/需要快收敛选 TB 系并配 reference flow 类稳定器；审稿人对「loss 低=学对了」的质疑现在有标准回应。

## 3. 非无环、连续与随机扩展

- 理论底座仍是 T12（连续 GFN 理论，测度化 pointed graph）与 [A Theory of Non-acyclic Generative Flow Networks](https://ojs.aaai.org/index.php/AAAI/article/view/28989)（AAAI 2024）：后者证明既有 loss 会把流推入环并给出修复损失族。
- 2025-2026 的增量主要在离散非无环：T36（Revisiting Non-Acyclic GFlowNets in Discrete Environments，ICML 2025）修正非无环设置下的训练目标与评估协议；T19 的非无环理论被多篇 2026 预印本引用为出发点。
- 连续控制方向自 N015（CFlowNets，ICLR 2023）后未检索到 2026 年的直接后续大改进；活跃度明显低于扩散采样器方向（见下节）。我的判断：连续 GFN 的赛道事实上已被「GFN 式目标 + 扩散采样器」吸收。

## 4. 与扩散采样器的融合：GFN 目标成为 off-policy 采样器的标准选项

关键事实链：

- 方法底座是 [Improved off-policy training of diffusion samplers](https://arxiv.org/abs/2402.05098)（NeurIPS 2024，本目录 N010）：统一库 + 基准，确认 TB/VarGrad + 局部搜索回放（LS）+ Langevin 参数化（LP）的组合在 Manywell、LGCP 等基准上超过或追平 PIS 等 simulation-based 变分方法；logZ 用 K=2000 的 MC 估计评测。该文同时指出领域存在基准不一致问题（不同论文对同一目标密度的定义都有分歧，其 B.1 节）。
- 2026 年该路线的评测协议（GMM40/Manywell/LGCP + logZ 下界/ELBO/EUBO）已成为新采样器论文的事实标准；N029（learned diffusion sampling）与 N002（离散-连续渐近等价，TMLR）把离散 GFN 收敛分析接入连续时间工具箱。
- torchgfn 官方文档已把 off-policy 训练（replay buffer、epsilon-greedy、温度缩放）写成[标准指南](https://torchgfn.readthedocs.io/en/stable/guides/off_policy_training.html)，说明这套训练法已经工程固化。

## 5. 顶会信号（以本仓库目录统计为准，检索截止 2026-08-25）

本目录 206 篇中按 venue 统计：NeurIPS 2024 收 14 篇、ICLR 2025 收 17 篇、ICML 2025 收 13 篇、NeurIPS 2025 收 13 篇、ICML 2026 收 17 篇、ICLR 2026 收 7 篇；2026 年预印本 20 篇、2025 年预印本 11 篇。趋势读法：

- 主会产出量在 2024-2026 保持每届 7-17 篇的稳定水平，没有爆发也没有衰减。
- 主题重心迁移明显：2023-2024 以训练目标与理论为主，2025-2026 新增论文集中在 LLM 后训练（N018 TBA、N019 FlowRL、RapTB）、扩散采样器、以及「可认证训练」（Stable GFlowNets、N001）。
- ICML 2026 出现 Oral 级别的相邻工作（N082 Path-dependent Discrete Amortized Inference），说明「路径依赖摊销推断」这个 GFN 泛化框架开始被主会前排接受。

## 6. 对研究者的 5 条可操作判断

1. 做训练目标改进的窗口正在关闭：TB 族的「平衡恒等式」设计空间已被 f-TB（T49）、alpha-GFN 参数化覆盖；增量空间在「目标 + 证书」（loss 到 TV 的有限样本认证）与「目标 + 博弈」（DTB 式对抗探索），单纯提出新平衡式很难过审。
2. **收敛性理论的空位已被论文作者自己确认**：T40（Secrets of GFlowNets）在 limitations 里明确写道，把收敛保证扩展到「具有现实架构的深度神经网络」这类一般函数逼近设定仍是待做工作，包括分析网络深度、宽度与性能的相互作用。现有速率（FM $\mathcal{O}(1/\sqrt{T})$、DB $\mathcal{O}(1/T^{1/3})$）与样本复杂度都依赖 tabular 或可实现性假设。这个鸿沟有实测证据：N081 在高维稀疏离散图上观察到 **FM 多数情况不收敛**、DB 一致优于 TB 与 FM，与理论速率排序相反（详见 `insights/trends_failure_modes.md` §1）。**结论：不要用理论速率指导高维选型。**
3. LLM 后训练是当前最大应用增长点：TBA（N018）→ FlowRL（N019）→ RapTB 的推进节奏是每 3-6 个月一篇有分量的工作；失败模式（前缀坍缩、长度偏置）已被明确命名，跟进者应直接在这些命名问题上给可测量改进。
4. 评测协议已收敛，别自造基准：采样器方向用 N010 的库与指标（logZ MC 估计 K=2000、ELBO/EUBO），分子方向用 fragment-based + QM9；自造基准的论文会被要求补标准基准。
5. 连续控制方向不建议新入：CFlowNets 后无有力跟进，社区资源已流向扩散采样器；除非有机器人等真实场景背书，否则该方向难有下文。

## 附：本报告引用的论文与其在本目录中的编号

本节纠正一个检索疏漏：起初以为下列论文多数在 206 篇目录之外，逐条核对后发现 7 篇已收录（本目录的覆盖比预期完整）。已收录的给出编号，便于回查目录里的简介与优先级。

| 论文 | 目录编号 | 来源 | 日期 |
|---|---|---|---|
| Stable GFlowNets with Probabilistic Guarantees | **T51** | [arXiv 2605.01729](https://arxiv.org/abs/2605.01729) | 预印本 2026 |
| Avoid What You Know: Divergent Trajectory Balance | **T50** | [arXiv 2602.17827](https://arxiv.org/abs/2602.17827) | ICML 2026 主会 |
| Rooted Absorbed Prefix Trajectory Balance (RapTB) | **T54** | [arXiv 2603.00454](https://arxiv.org/abs/2603.00454) | ICML 2026 |
| Controlling Exploration-Exploitation via Markov Chain (alpha-GFN) | **T46** | [arXiv 2602.01749](https://arxiv.org/abs/2602.01749) | 预印本 2026 |
| Secrets of GFlowNets' Learning Behavior | **T40** | [arXiv 2505.02035](https://arxiv.org/abs/2505.02035) | 预印本 2025 |
| Generalization and Distributed Learning of GFlowNets | **T33** | [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/000eba875068854d5ff003b1fa534cd6-Paper-Conference.pdf) | ICLR 2025 |
| A Theory of Non-acyclic Generative Flow Networks | **T19** | [AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/28989) | AAAI 2024 |

也就是说，本报告的价值不在于「发现了目录没有的论文」，而在于**把这些已收录条目串成趋势链条并给出取舍判断**——目录提供的是逐篇简介，这里回答的是「该往哪走、哪条路已经走满」。
