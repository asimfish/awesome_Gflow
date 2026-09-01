# Awesome GFlowNets [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

精选的 GFlowNet（Generative Flow Networks）论文、代码、课程与深度解读清单。收录论文 **206** 篇（检索截止 2026-08-25），其中 **35** 篇核心论文配有中文深度解读（已完成 35 篇）与保版式中文翻译 PDF（已完成 28 篇，由 [SuperTranslate](https://github.com/asimfish/super_translate) + Qwen2.5-32B 生成）。

A curated list of GFlowNet papers, code, courses and in-depth Chinese notes. Legend: 📝 深度解读 in-depth note · 🇨🇳 中文PDF Chinese translation · 📄 英文PDF original PDF.

## 目录 Contents

- [综述与资源 Surveys & Resources](#综述与资源-surveys--resources)
- **理论 Theory**
  - [奠基、训练目标与信用分配 · Foundations & Training Objectives](#奠基训练目标与信用分配--foundations--training-objectives) (7)
  - [VI、RL 与状态空间扩展 · Connections to VI / RL & Extensions](#virl-与状态空间扩展--connections-to-vi--rl--extensions) (24)
  - [收敛性、泛化与表达能力 · Convergence, Generalization & Expressivity](#收敛性泛化与表达能力--convergence-generalization--expressivity) (6)
  - [训练目标与损失设计 · Training Objectives & Loss Design](#训练目标与损失设计--training-objectives--loss-design) (4)
  - [训练、探索与效率 · Training, Exploration & Efficiency](#训练探索与效率--training-exploration--efficiency) (28)
  - [连续、非无环与随机扩展 · Continuous, Non-acyclic & Stochastic](#连续非无环与随机扩展--continuous-non-acyclic--stochastic) (3)
  - [2026 前沿方法 · Frontier Methods (2026)](#2026-前沿方法--frontier-methods-2026) (21)
- **交叉方向 Cross-cutting**
  - [GFlowNet × 最优传输 · GFlowNet × Optimal Transport](#gflownet--最优传输--gflownet--optimal-transport) (18)
  - [扩散采样器与随机最优控制 · Diffusion Samplers & SOC](#扩散采样器与随机最优控制--diffusion-samplers--soc) (8)
- **应用 Applications**
  - [分子、蛋白与材料 · Molecules, Proteins & Materials](#分子蛋白与材料--molecules-proteins--materials) (23)
  - [结构学习与组合优化 · Structure Learning & Combinatorial Optimization](#结构学习与组合优化--structure-learning--combinatorial-optimization) (8)
  - [LLM、推理与视觉 · LLMs, Reasoning & Vision](#llm推理与视觉--llms-reasoning--vision) (21)
  - [多目标与条件生成 · Multi-objective & Conditional Generation](#多目标与条件生成--multi-objective--conditional-generation) (3)
  - [安全、红队与对齐 · Safety, Red-teaming & Alignment](#安全红队与对齐--safety-red-teaming--alignment) (2)
  - [其他应用 · Other Applications](#其他应用--other-applications) (3)
- **生态 Ecosystem**
  - [评测基准与软件 · Benchmarks & Software](#评测基准与软件--benchmarks--software) (2)
  - [审查流水线补录 · Additional Curated Papers](#审查流水线补录--additional-curated-papers) (25)
- [趋势洞察 Trends & Insights](#趋势洞察-trends--insights)
- [课程与教程 Courses & Tutorials](#课程与教程-courses--tutorials)
- [代码库 Codebases](#代码库-codebases)

## 综述与资源 Surveys & Resources

本仓库自带的系统性中文调研文档：

- [Gflownet Research Resource Catalog](surveys/GFLOWNET_RESEARCH_RESOURCE_CATALOG_CN.md) — 206 篇论文的完整目录：分类、简介、优先级与阅读路线
- [Gflownet Theory Guide](surveys/GFLOWNET_THEORY_GUIDE_CN.md) — GFlowNet 理论指南：从流匹配到各训练目标的推导与对比
- [Gflownet Ot Potential Analysis](surveys/GFLOWNET_OT_POTENTIAL_ANALYSIS_CN.md) — GFlowNet × 最优传输的潜力分析与四个候选研究课题
- [GFLOWNET COMBINATORIAL OPT CN](surveys/GFLOWNET_COMBINATORIAL_OPT_CN.md)
- [GFLOWNET CONDITIONAL MULTIOBJ CN](surveys/GFLOWNET_CONDITIONAL_MULTIOBJ_CN.md)
- [GFLOWNET CONTINUOUS CN](surveys/GFLOWNET_CONTINUOUS_CN.md)
- [GFLOWNET DIFFUSION SAMPLER CN](surveys/GFLOWNET_DIFFUSION_SAMPLER_CN.md)
- [GFLOWNET EVALUATION ECOSYSTEM CN](surveys/GFLOWNET_EVALUATION_ECOSYSTEM_CN.md)
- [GFLOWNET EXPLORATION LINEAGE CN](surveys/GFLOWNET_EXPLORATION_LINEAGE_CN.md)
- [GFLOWNET GLOSSARY TOC CN](surveys/GFLOWNET_GLOSSARY_TOC_CN.md)
- [GFLOWNET GRADIENT VARIANCE CN](surveys/GFLOWNET_GRADIENT_VARIANCE_CN.md)
- [GFLOWNET LEARNING PATH UPGRADE CN](surveys/GFLOWNET_LEARNING_PATH_UPGRADE_CN.md)
- [GFLOWNET LLM CN](surveys/GFLOWNET_LLM_CN.md)
- [GFLOWNET MOLECULAR APPLICATIONS PANORAMA CN](surveys/GFLOWNET_MOLECULAR_APPLICATIONS_PANORAMA_CN.md)
- [GFLOWNET NONACYCLIC CN](surveys/GFLOWNET_NONACYCLIC_CN.md)
- [GFLOWNET RL EQUIVALENCE CN](surveys/GFLOWNET_RL_EQUIVALENCE_CN.md)
- [GFLOWNET STABILITY CN](surveys/GFLOWNET_STABILITY_CN.md)
- [GFLOWNET STRUCTURE LEARNING CN](surveys/GFLOWNET_STRUCTURE_LEARNING_CN.md)
- [GFLOWNET VISION CN](surveys/GFLOWNET_VISION_CN.md)
- [GFLOWNET VI MCMC SMC CN](surveys/GFLOWNET_VI_MCMC_SMC_CN.md)

## 奠基、训练目标与信用分配 · Foundations & Training Objectives

- `T01` [Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation](https://proceedings.neurips.cc/paper/2021/hash/e614f646836aaed9f89ce58e837e2310-Abstract.html) · *NeurIPS 2021* [📝 深度解读](notes/T01_gflownet_original.md) · [🇨🇳 中文PDF](pdfs/zh/T01_zh.pdf) · [📄 英文PDF](pdfs/en/T01_flow_network_based_generative_models_for_non_iterative_diver.pdf)  
  GFlowNet 的原始论文，用 DAG 上的流匹配把终止对象的采样概率训练成与奖励成比例，并以分子设计和主动学习说明“多样高奖励采样”不同于最大化奖励。
- `T02` [GFlowNet Foundations](https://jmlr.org/papers/v24/22-0364.html) · *JMLR 2023* [📝 深度解读](notes/T02_gflownet_foundations.md) · [🇨🇳 中文PDF](pdfs/zh/T02_zh.pdf) · [📄 英文PDF](pdfs/en/T02_gflownet_foundations.pdf)  
  建立 Markovian flow、state/edge/trajectory flow、前向与后向策略、reward matching 等统一数学框架，并系统说明 FM、DB 等约束为何导出正确终止分布。
- `T03` [Trajectory Balance: Improved Credit Assignment in GFlowNets](https://proceedings.neurips.cc/paper_files/paper/2022/hash/27b51baca8377a0cf109f6ecc15a0f70-Abstract.html) · *NeurIPS 2022* [📝 深度解读](notes/T03_trajectory_balance_credit_assignment.md) · [🇨🇳 中文PDF](pdfs/zh/T03_zh.pdf) · [📄 英文PDF](pdfs/en/T03_trajectory_balance_improved_credit_assignment_in_gflownets.pdf)  
  提出轨迹级恒等式，将一条完整构造路径上的前向概率、后向概率、配分函数 \(Z\) 和终点奖励联系起来。
- `T04` [Generative Flow Networks for Discrete Probabilistic Modeling](https://proceedings.mlr.press/v162/zhang22v.html) · *ICML 2022*  
  从离散概率建模角度研究 Markov flow 的参数化和学习，澄清多条生成路径如何共同决定对象概率及其熵结构。
- `T05` [Learning GFlowNets from Partial Episodes for Improved Convergence and Stability](https://proceedings.mlr.press/v202/madan23a.html) · *ICML 2023* [📝 深度解读](notes/T05_subtrajectory_balance_partial_episodes.md) · [🇨🇳 中文PDF](pdfs/zh/T05_zh.pdf) · [📄 英文PDF](pdfs/en/T05_learning_gflownets_from_partial_episodes_for_improved_conver.pdf)  
  提出 Subtrajectory Balance，用任意子轨迹上的 balance 约束连接一步 DB 与完整 TB，并用 \(\lambda\) 控制信用分配尺度。
- `T06` [Better Training of GFlowNets with Local Credit and Incomplete Trajectories](https://proceedings.mlr.press/v202/pan23c.html) · *ICML 2023*  
  利用中间状态的局部能量或奖励信息，并允许从不完整轨迹学习，从而缓解只有终点奖励时的稀疏信用问题。
- `T07` [Towards Understanding and Improving GFlowNet Training](https://proceedings.mlr.press/v202/shen23a.html) · *ICML 2023* [📝 深度解读](notes/T07_understanding_improving_gfn_training.md) · [🇨🇳 中文PDF](pdfs/zh/T07_zh.pdf) · [📄 英文PDF](pdfs/en/T07_towards_understanding_and_improving_gflownet_training.pdf)  
  系统研究有限训练下 loss、终止分布误差和内部流之间的鸿沟，指出常用目标可能在欠拟合、稀疏奖励和长轨迹环境中产生误导。

## VI、RL 与状态空间扩展 · Connections to VI / RL & Extensions

- `T08` [GFlowNets and Variational Inference](https://openreview.net/forum?id=uKiE0VIluA-) · *ICLR 2023* [📝 深度解读](notes/T08_gflownets_vs_hvi.md) · [🇨🇳 中文PDF](pdfs/zh/T08_zh.pdf) · [📄 英文PDF](pdfs/en/T08_gflownets_and_variational_inference.pdf)  
  比较 GFlowNet 与层次变分推断，说明特定条件下两者梯度估计之间的联系，同时强调采样分布、后向策略和 off-policy 数据会造成实质差异。
- `T09` [A Variational Perspective on Generative Flow Networks](https://openreview.net/forum?id=AZ4GobeSLq) · *TMLR 2023* [📝 深度解读](notes/T09_variational_perspective_gfn.md) · [🇨🇳 中文PDF](pdfs/zh/T09_zh.pdf) · [📄 英文PDF](pdfs/en/T09_a_variational_perspective_on_generative_flow_networks.pdf)  
  在轨迹空间中构造前向与反向分布，并用 KL 目标统一解释 TB 及若干 GFlowNet 训练规则。
- `T10` [GFlowNet-EM for Learning Compositional Latent Variable Models](https://proceedings.mlr.press/v202/hu23c.html) · *ICML 2023*  
  把 GFlowNet 用作组合离散潜变量的摊销后验，并嵌入 EM 式参数学习循环。
- `T11` [Generative Flow Networks: A Markov Chain Perspective](https://arxiv.org/abs/2307.01422) · *预印本 2023*  
  用 Markov chain 的语言重新表述 GFlowNet，帮助连接平稳分布、路径测度和循环状态空间。
- `T12` [A Theory of Continuous Generative Flow Networks](https://proceedings.mlr.press/v202/lahlou23a.html) · *ICML 2023* [📝 深度解读](notes/T12_continuous_gflownets_theory.md) · [🇨🇳 中文PDF](pdfs/zh/T12_zh.pdf) · [📄 英文PDF](pdfs/en/T12_a_theory_of_continuous_generative_flow_networks.pdf)  
  将离散 DAG 上的流守恒推广到一般测度空间，处理连续及混合状态、可测流和密度等问题。
- `T13` [Stochastic Generative Flow Networks](https://proceedings.mlr.press/v216/pan23a.html) · *UAI 2023*  
  把环境转移本身具有随机性的情况纳入 GFlowNet，而不是假设动作唯一决定下一状态。
- `T14` [Generative Flow Networks as Entropy-Regularized RL](https://proceedings.mlr.press/v238/tiapkin24a.html) · *AISTATS 2024* [📝 深度解读](notes/T14_gfn_entropy_regularized_rl.md) · [🇨🇳 中文PDF](pdfs/zh/T14_zh.pdf) · [📄 英文PDF](pdfs/en/T14_generative_flow_networks_as_entropy_regularized_rl.pdf)  
  将 GFlowNet 写成特定的最大熵强化学习问题，连接 soft value、policy consistency 和 reward-proportional sampling。
- `T15` [Discrete Probabilistic Inference as Control in Multi-path Environments](https://proceedings.mlr.press/v244/deleu24a.html) · *UAI 2024* [📝 深度解读](notes/T15_inference_as_control_multipath.md) · [🇨🇳 中文PDF](pdfs/zh/T15_zh.pdf) · [📄 英文PDF](pdfs/en/T15_discrete_probabilistic_inference_as_control_in_multi_path_en.pdf)  
  从控制角度推导离散概率推断，明确多路径结构中必须怎样校正奖励才能得到目标对象分布。
- `T16` [GFlowNet Training by Policy Gradients](https://proceedings.mlr.press/v235/niu24c.html) · *ICML 2024*  
  提出用 policy-gradient 形式训练 GFlowNet，并讨论前向与后向策略的联合设计。
- `T17` [On Divergence Measures for Training GFlowNets](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8aaae73117c9266b35eb977e70fcf95f-Abstract-Conference.html) · *NeurIPS 2024* [📝 深度解读](notes/T17_divergence_training_gflownets.md) · [🇨🇳 中文PDF](pdfs/zh/T17_zh.pdf) · [📄 英文PDF](pdfs/en/T17_on_divergence_measures_for_training_gflownets.pdf)  
  系统研究不同 divergence 如何诱导 GFlowNet 训练目标，并指出实践失败往往来自梯度估计方差而非 divergence 本身。
- `T18` [Expected Flow Networks in Stochastic Environments and Two-Player Zero-Sum Games](https://iclr.cc/virtual/2024/poster/17581) · *ICLR 2024*  
  用期望意义下的流约束扩展确定性 GFlowNet，使其能够处理随机环境和二人零和博弈。
- `T19` [A Theory of Non-Acyclic Generative Flow Networks](https://ojs.aaai.org/index.php/AAAI/article/view/28989) · *AAAI 2024* [📝 深度解读](notes/T19_nonacyclic_gfn_theory.md) · [🇨🇳 中文PDF](pdfs/zh/T19_zh.pdf) · [📄 英文PDF](pdfs/en/T19_a_theory_of_non_acyclic_generative_flow_networks.pdf)  
  建立允许环的一般 GFlowNet 理论，用 expected visit flow 和吸收条件取代 DAG 中的简单路径计数，并揭示循环可能造成的 flow explosion。
- `T20` [Diffusion Generative Flow Samplers](https://arxiv.org/abs/2310.02679) · *ICLR 2024*  
  将 GFlowNet 的部分轨迹训练信号引入连续迭代式 diffusion sampler，以学习未归一化目标分布。
- `N018` [Trajectory Balance with Asynchrony (TBA)](https://arxiv.org/abs/2503.18929) · *NeurIPS 2025（已接收，海报 #1115）* [📝 深度解读](notes/N018_trajectory_balance_asynchrony.md) · [🇨🇳 中文PDF](pdfs/zh/N018_zh.pdf) · [📄 英文PDF](pdfs/en/N018_trajectory_balance_with_asynchrony.pdf)  
  把 **off-policy TB 目标**嵌入异步分布式 RL：多个 searcher 并行生成轨迹写入 replay buffer，单个 trainer 按 reward/recency 优先级异步采样更新，解耦"探索"与"学习"。
- `N019` [FlowRL: Matching Reward Distributions for LLM Reasoning](https://arxiv.org/abs/2509.15207) · *ICLR 2026* [📝 深度解读](notes/N019_flowrl_reward_distribution_llm.md) · [🇨🇳 中文PDF](pdfs/zh/N019_zh.pdf) · [📄 英文PDF](pdfs/en/N019_flowrl_matching_reward_distributions_for_llm_reasoning.pdf)  
  主张用"匹配完整奖励分布"取代 PPO/GRPO 的奖励最大化：用**可学习配分函数** \(Z_\phi(x)\) 把标量奖励归一化为目标分布，最小化策略与目标的**反向 KL**，并证明其在期望梯度上**等价于 GFlowNet 的 TB 损失**。
- `N020` [GDPO: Learning to Directly Align Language Models with Diversity Using GFlowNets](https://aclanthology.org/2024.emnlp-main.951/) · *EMNLP 2024 Main，pp. 17120-17139*  
  把**离线**偏好对齐视为贝叶斯推断，用 GFlowNet 直接从离线偏好数据学习"按奖励分布采样"的前向策略，得到多样性寻优的 DPO 变体（GFlowNet-DPO）。
- `N021` [Beyond Normalization: Rethinking the Partition Function as a Difficulty Scheduler for RLVR (PACED-RL)](https://arxiv.org/abs/2602.12642) · *预印本 2026*  
  在 GFlowNet 式 LLM 后训练（如 FlowRL）之上重新诠释配分函数：它在最优处等于某 prompt 所有补全的奖励质量之和，可直接当作**在线准确率/难度估计器**，用于难度感知的自适应 prompt 选择；并利用 GFN 目标的 **off-policy 容忍度**做误差优先 replay，提升 RLVR 样本效率。
- `N022` [Probabilistic Inference in Language Models via Twisted Sequential Monte Carlo](https://proceedings.mlr.press/v235/zhao24c.html) · *ICML 2024 · arXiv:2404.17546*  
  把 RLHF、自动红队、填充等 LLM 任务统一为对未归一化目标分布的后验采样，用学习的 twist 函数估计部分序列的未来价值、聚焦推断期算力，并给出双向 SMC 配分函数界以评估各类推断方法的 KL 误差。
- `N023` [Outsourced Diffusion Sampling: Efficient Posterior Inference in Latent Spaces of Generative Models](https://proceedings.mlr.press/v267/venkatraman25a.html) · *ICML 2025 · arXiv:2502.06999*  
  把任意生成模型写成外生高斯噪声的确定性变换，在噪声空间训练扩散采样器（RL/GFN 式目标）以采样约束后验，使 GAN、(H)VAE、流模型等先验下的条件采样成为可能；噪声空间后验通常更平滑，更适合摊销推断。
- `N024` [Adaptive teachers for amortized samplers](https://openreview.net/forum?id=BdmVgLMvaf) · *ICLR 2025 · arXiv:2410.01432*  
  为摊销采样器训练引入自适应行为策略"教师"：教师专门采样学生模型高损失区域且能泛化到未探索模式，形成高效课程以提升模式覆盖与样本效率；在探索困难的合成环境、两类扩散采样任务与四类生化发现任务上验证。
- `N025` [GFlowOut: Dropout with Generative Flow Networks](https://proceedings.mlr.press/v202/liu23r.html) · *ICML 2023 · arXiv:2210.12928*  
  把 dropout mask 视为潜变量，用 GFlowNet 学习其高度多峰的后验分布，取代独立固定分布采样或标准变分推断，并利用样本相关信息改进后验估计；实证改善分布外泛化与不确定性估计。
- `N026` [DynGFN: Towards Bayesian Inference of Gene Regulatory Networks with GFlowNets](https://papers.nips.cc/paper_files/paper/2023/hash/eb5254c4ee813d05af9c098f2d9c5708-Abstract-Conference.html) · *NeurIPS 2023 · arXiv:2302.04178*  
  利用 RNA velocity 把基因调控网络推断转为动力系统稀疏辨识，再用 GFlowNet 在含环依赖结构的组合空间上摊销贝叶斯后验，同时解决"调控结构天然含环、不能建成 DAG"与"观测噪声导致大等价类、需刻画不确定性"两个难题。
- `N027` [Learning Decision Trees as Amortized Structure Inference](https://arxiv.org/abs/2503.06985) · *预印本 2025 · arXiv:2503.06985*  
  把决策树构造写成顺序规划问题，训练 GFlowNet 策略从贝叶斯后验采样决策树（DT-GFN），采样集成即得随机森林；在表格数据分类、分布偏移鲁棒性与异常检测上超过主流树模型和深度方法，且模型描述长度更短、可解释，集成规模上表现一致扩展。
- `N028` [Stop the Sampler! Classifier-Based Adaptive Stopping for Sampling Kernels](https://arxiv.org/abs/2606.16073) · *ICML 2026 SPIGM Workshop · arXiv:2606.16073*  
  把 MCMC 轨迹的终止时机当作可学习组件：在非无环 GFlowNet 理论框架内训练状态依赖分类器决定链何时停止，经 detailed balance 建立最优分类器与目标密度的联系，并用多层级训练方案辅助复杂几何下的探索；实证缩短平均轨迹长度并改善混合与模式覆盖。

## 收敛性、泛化与表达能力 · Convergence, Generalization & Expressivity

- `N001` [Convergences guarantees of GFlowNets](https://openreview.net/forum?id=JmsgmkdIkk) · *NeurIPS 2025 FPI Workshop Poster*  
  针对"TB loss 下降是否意味着分布靠近目标"这一长期悬置的直觉给出正面定理:证明最小化 Trajectory Balance loss 时,学到分布与目标分布的 KL 散度被所最小化的量上界控制,并在小型采样任务上验证。
- `N002` [From discrete-time policies to continuous-time diffusion samplers: Asymptotic equivalences and faster training](https://arxiv.org/abs/2501.06148) · *TMLR*  
  研究无目标样本时训练神经 SDE/扩散采样器,证明离散步长趋零的极限下,GFlowNet 式熵正则 RL 目标与连续时间对象(PDE、路径空间测度)之间的一族渐近等价,并据此设计更快的训练方案(附官方代码)。
- `N003` [Maximum entropy GFlowNets with soft Q-learning](https://proceedings.mlr.press/v238/mohammadpour24a.html) · *AISTATS 2024 主会*  
  通过构造修正奖励,把 GFlowNet 与最大熵 RL 建立为精确对应而非特例式近似,并导出用 soft Q-learning 训练 GFN 的算法。
- `N004` [Investigating Generalization Behaviours of Generative Flow Networks](https://arxiv.org/abs/2402.05309) · *TMLR 2025*  
  系统实证检验"GFN+深度网络泛化良好"这一流行假设:构造奖励难度可调、\(p(x)\) 可精确计算、含未见测试集的图环境。
- `N005` [Information-Geometric Forward Policy Training in GFlowNets](https://arxiv.org/abs/2608.03967) · *预印本 2026-08*  
  把前向策略视为轨迹采样器,证明其一阶内蕴几何由轨迹族的 Fisher-Rao 度量给出、自然梯度是规范局部更新;给出轨迹 Fisher 到逐步条件二阶矩的精确分解,并划分精确计算/Monte Carlo/利用目标局部结构三种可计算范式。
- `N006` [Analyzing GFlowNets: Stability, Expressiveness, and Assessment](https://openreview.net/forum?id=B8KXmXFiFj) · *ICML 2024 SPIGM Workshop Poster*  
  从稳定性、表达能力、评估三维度分析 GFN:证明 balance 违反的影响在状态图上不均匀、节点影响力与其后代奖励挂钩,据此提出加权 balance loss 加速收敛;并证明合适状态图下 GFN 可精确表示任意树上分布,同时构造出 balance 不可达的失败案例。

## 训练目标与损失设计 · Training Objectives & Loss Design

- `N007` [Distributional GFlowNets with Quantile Flows](https://openreview.net/forum?id=vFSsRYGpjW) · *TMLR 2024 · arXiv:2302.05793* [📝 深度解读](notes/N007_distributional_quantile_flows.md) · [🇨🇳 中文PDF](pdfs/zh/N007_zh.pdf) · [📄 英文PDF](pdfs/en/N007_distributional_gflownets_with_quantile_flows.pdf)  
  把每条边的流从标量改为分布并用分位数函数参数化，提出 quantile matching（QM）这一分布式 TD 风格 balance 目标：既能处理随机奖励，又可经失真风险度量得到风险敏感策略；确定性基准上也因更强训练信号优于 FM/DB/TB。
- `N008` [Order-Preserving GFlowNets](https://openreview.net/forum?id=VXDPXuq4oG) · *ICLR 2024 · arXiv:2310.00386*  
  不再拟合给定标量奖励，而是学习与候选（偏）序一致的奖励并按其采样；理论证明训练过程逐步稀疏化奖励景观，天然实现"先探索后利用"，免去奖励指数 \(\beta\) 调参，并直接适配多目标 Pareto 前沿。
- `N009` [Delta-AI: Local objectives for amortized inference in sparse graphical models](https://proceedings.iclr.cc/paper_files/paper/2024/hash/710445227fa8c1b6a9ceada902dd4741-Abstract-Conference.html) · *ICLR 2024 · arXiv:2310.02423*  
  针对稀疏概率图模型的摊销推断，利用变量 Markov blanket 上的条件分布匹配构造 GFlowNet 式局部损失：每次参数更新无需实例化全部变量或完整轨迹，信用分配完全局部化，训练显著加速且支持 off-policy。
- `N010` [Improved off-policy training of diffusion samplers](https://openreview.net/forum?id=vieIamY2Gi) · *NeurIPS 2024 poster · arXiv:2402.05098* [📝 深度解读](notes/N010_offpolicy_diffusion_samplers.md) · [🇨🇳 中文PDF](pdfs/zh/N010_zh.pdf) · [📄 英文PDF](pdfs/en/N010_improved_off_policy_training_of_diffusion_samplers.pdf)  
  建立统一代码库与基准，系统比较连续 GFlowNet 目标（TB、VarGrad 型 Z-free 估计、FL-SubTB、Langevin 参数化）与 PIS 等模拟式变分目标训练 diffusion sampler 的表现，并提出目标空间局部搜索加回放缓冲的探索策略；修正了此前关于鲁棒性与样本效率的部分结论。

## 训练、探索与效率 · Training, Exploration & Efficiency

- `T21` [Learning Energy Decompositions for Partial Inference in GFlowNets](https://iclr.cc/virtual/2024/poster/18721) · *ICLR 2024*  
  学习把终点能量分解到中间状态的势函数，为部分轨迹提供局部信用。
- `T22` [Pre-Training and Fine-Tuning Generative Flow Networks](https://iclr.cc/virtual/2024/poster/17406) · *ICLR 2024*  
  先进行 outcome-conditioned、弱依赖具体奖励的预训练，再对下游目标微调。
- `T23` [Local Search GFlowNets](https://iclr.cc/virtual/2024/poster/19387) · *ICLR 2024*  
  在已有样本附近执行回退和重构，将局部搜索轨迹用于 off-policy 训练，以更快进入高奖励区域。
- `T24` [Embarrassingly Parallel GFlowNets](https://proceedings.mlr.press/v235/silva24a.html) · *ICML 2024*  
  并行训练多个具有不同经验或偏好的子 GFN，再组合其覆盖能力。
- `T25` [Pessimistic Backward Policy for GFlowNets](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c1ab28d0fe0bfb53067a1af7e578cd7d-Abstract-Conference.html) · *NeurIPS 2024*  
  设计更“悲观”的后向策略来改变内部轨迹分配，使训练更集中于高奖励区域。
- `T26` [QGFN: Controllable Greediness with Action Values](https://proceedings.neurips.cc/paper_files/paper/2024/hash/948d8ba4e30c8c3a800cf436b31f376e-Abstract-Conference.html) · *NeurIPS 2024*  
  引入动作价值信息，在推断阶段连续调节采样的 greediness，而无需为每种温度重新训练模型。
- `T27` [Streaming Bayes GFlowNets](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2fb57276bfbaf1b832d7bfcba36bb41c-Abstract-Conference.html) · *NeurIPS 2024*  
  研究新数据批次不断到达时如何增量更新离散后验 GFN，避免每次从头训练。
- `T28` [Learning to Scale Logits for Temperature-Conditional GFlowNets](https://proceedings.mlr.press/v235/kim24s.html) · *ICML 2024*  
  用单个条件模型学习跨温度的 logit 缩放，从而覆盖从多样采样到集中搜索的一族目标分布。
- `T29` [On Generalization for GFlowNets](https://arxiv.org/abs/2407.03105) · *预印本 2024*  
  从统计学习角度讨论仅见到部分状态或轨迹时，局部 balance 拟合能否推广到未见区域。
- `T30` [Action Abstractions for Amortized Sampling](https://arxiv.org/abs/2410.15184) · *ICLR 2025*  
  通过宏动作或动作抽象缩短有效生成深度，减少长轨迹中的信用传播负担。
- `T31` [Optimizing Backward Policies in GFlowNets via Trajectory Likelihood Maximization](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f3efbcfe76bed022a37c5aeb1daf2326-Abstract-Conference.html) · *ICLR 2025*  
  通过轨迹似然最大化和 entropy-RL 视角显式优化 \(P_B\)，而不是把它固定为均匀分布。
- `T32` [When Do GFlowNets Learn the Right Distribution?](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a48f8928f78a58399ef0049453c14b02-Abstract-Conference.html) · *ICLR 2025 Spotlight* [📝 深度解读](notes/T32_when_gfn_learn_right_distribution.md) · [📄 英文PDF](pdfs/en/T32_when_do_gflownets_learn_the_right_distribution.pdf)  
  分析局部 balance 误差、函数表示限制与最终对象分布误差之间的关系，并提出比平均 loss 更可靠的 correctness 评价。
- `T33` [Generalization and Distributed Learning of GFlowNets](https://iclr.cc/virtual/2025/poster/29760) · *ICLR 2025*  
  给出数据依赖的泛化分析，并提出异步分布式 SAL 训练方法。
- `T34` [Beyond Squared Error: Exploring Loss Design for Enhanced Training of Generative Flow Networks](https://proceedings.iclr.cc/paper_files/paper/2025/hash/353ec686503cd7020460d2829578ee4e-Abstract-Conference.html) · *ICLR 2025*  
  系统比较平方误差之外的回归损失，研究尾部惩罚和鲁棒性如何改变探索—利用及梯度行为。
- `T35` [Towards Improving Exploration through Sibling Augmented GFlowNets](https://iclr.cc/virtual/2025/poster/30233) · *ICLR 2025*  
  用一个探索 sibling 与主 GFN 协作，将发现新模式和保持目标分布的职责部分解耦。
- `T36` [Revisiting Non-Acyclic GFlowNets in Discrete Environments](https://proceedings.mlr.press/v267/morozov25a.html) · *ICML 2025* [📝 深度解读](notes/T36_revisiting_nonacyclic_gfn.md) · [🇨🇳 中文PDF](pdfs/zh/T36_zh.pdf) · [📄 英文PDF](pdfs/en/T36_revisiting_non_acyclic_gflownets_in_discrete_environments.pdf)  
  在有限离散非无环环境中给出更简洁的存在性、唯一性和稳定性分析，并刻画固定 \(P_B\) 与 minimum flow 的关系。
- `T37` [Random Policy Evaluation Uncovers Policies of GFlowNets](https://proceedings.mlr.press/v267/he25a.html) · *ICML 2025*  
  把 GFlowNet flow 与普通随机策略的 policy evaluation 联系起来，提供分析策略结构的新工具。
- `T38` [Symmetry-Aware GFlowNets](https://proceedings.mlr.press/v267/kim25s.html) · *ICML 2025*  
  处理多个轨迹或表示对应同一对称对象时产生的系统采样偏差，并在训练中显式利用等价类结构。
- `T39` [Ergodic Generative Flows](https://proceedings.mlr.press/v267/brunswic25a.html) · *ICML 2025*  
  以遍历变换和较弱的 flow-matching 条件推广 GFN，覆盖循环、连续变换及模仿学习等情形。
- `T40` [Secrets of GFlowNets' Learning Behavior: A Theoretical Study](https://arxiv.org/abs/2505.02035) · *预印本 2025*  
  分析 GFlowNet 的训练动力学和策略演化，试图解释模型为何会经历特定的模式发现与再分配过程。
- `T41` [Hybrid-Balance GFlowNet for Solving Vehicle Routing Problems](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1898054a9392207250ad9cfde5286b2c-Abstract-Conference.html) · *NeurIPS 2025 Spotlight*  
  面向车辆路径问题（VRP/CVRP/TSP），自适应结合 TB 的全局监督与 DB 的局部监督，并为含仓库节点的 CVRP 设计专门推断策略；以插件形式嵌入 AGFN、GFACS 等 GFlowNet 求解器。
- `T42` [Adaptive Quantization in Generative Flow Networks for Probabilistic Sequential Prediction](https://proceedings.neurips.cc/paper_files/paper/2025/hash/dc63026c74191032dff373ed9d2d038a-Abstract-Conference.html) · *NeurIPS 2025*  
  为概率序列预测学习非均匀离散化，使 GFN 能把建模容量分配到更重要的数值区域。
- `T43` [Flow Factorization for Efficient Generative Flow Networks](https://ojs.aaai.org/index.php/AAAI/article/view/34887) · *AAAI 2025*  
  将边流分解为状态流与动作分配项，以减少直接参数化所有边的成本。
- `T44` [Relative Trajectory Balance Is Equivalent to Trust-PCL](https://arxiv.org/abs/2509.01632) · *NeurIPS 2025 FPI Workshop*  
  给出 relative TB 与 path-consistency RL 方法 Trust-PCL 的对应关系，进一步收紧 GFN 与最大熵 RL 的理论连接。
- `N011` [Generative Augmented Flow Networks](https://arxiv.org/abs/2210.03308) · *ICLR 2023（notable top 25%） · arXiv:2210.03308* [📝 深度解读](notes/N011_generative_augmented_flow_networks.md) · [🇨🇳 中文PDF](pdfs/zh/N011_zh.pdf) · [📄 英文PDF](pdfs/en/N011_generative_augmented_flow_networks.pdf)  
  GAFlowNet 把 intrinsic motivation（Random Network Distillation）作为中间奖励注入 flow：同时用 edge-based 与 state-based 内在奖励增广，把只有终点奖励的稀疏信号变成沿轨迹的密集反馈，并证明增广目标渐近无偏于原 GFlowNet。
- `N012` [Thompson Sampling for Improved Exploration in GFlowNets](https://arxiv.org/abs/2306.17693) · *ICML 2023 SPIGM Workshop · arXiv:2306.17693*  
  TS-GFN 把"训练时选哪条轨迹"视作主动学习问题，用贝叶斯/多臂老虎机思想解决：将前向策略网络最后一层参数化为 ensemble，维护对策略的近似后验，每步采一个 ensemble 成员并按其策略采样轨迹，从而优先探索不确定区域。
- `N013` [An Empirical Study of the Effectiveness of Using a Replay Buffer on Mode Discovery in GFlowNets](https://arxiv.org/abs/2307.07674) · *ICML 2023 SPIGM Workshop · arXiv:2307.07674*  
  系统实证 replay buffer 对 GFlowNet 模式发现的作用：对比"无缓冲 / 随机采样缓冲 / R-PRS（Reward Prioritized Replay Sampling，仿 PER 的按奖励优先）"三种配置，在 Hypergrid 与分子合成环境上显示带缓冲尤其是 R-PRS 显著加速模式发现、提升多样性；关键结论是"提升来自更频繁访问高奖励轨迹，而非缓冲本身的存在"。
- `N014` [Looking Backward: Retrospective Backward Synthesis for Goal-Conditioned GFlowNets](https://arxiv.org/abs/2406.01150) · *ICLR 2025 · arXiv:2406.01150*  
  RBS 换一个方向用 backward policy：对未达标（零奖励）的前向轨迹，从目标状态出发用 P_B 合成一条"必然成功"的后向轨迹，反转后作为高质量正样本注入训练缓冲，从而在目标条件（goal-conditioned）、极稀疏奖励下制造大量可学习信号；配合 age-based 采样、P_B 正则（惩罚与均匀分布的 KL）与强化终点奖励反馈，样本效率大幅超过 HER、OC-GAFN 等强基线。

## 连续、非无环与随机扩展 · Continuous, Non-acyclic & Stochastic

- `N015` [CFlowNets: Continuous Control with Generative Flow Networks](https://openreview.net/forum?id=yAYHho4fATa) · *ICLR 2023（正式）* [📝 深度解读](notes/N015_cflownets_continuous_control.md) · [🇨🇳 中文PDF](pdfs/zh/N015_zh.pdf) · [📄 英文PDF](pdfs/en/N015_cflownets_continuous_control_with_generative_flow_networks.pdf)  
  首个把 GFlowNet 推向**连续控制/连续动作空间**的方法：用重要性采样近似连续状态的入流与出流，改写 flow-matching 损失，并给出流近似误差界（随采样数增大而衰减）。
- `N016` [MetaGFN: Exploring Distant Modes with Adapted Metadynamics for Continuous GFlowNets](https://openreview.net/forum?id=dtyNeemB7A) · *TMLR 2025*  
  针对"连续 GFN 的探索几乎无人研究"的空白，提出 Adapted Metadynamics：借 molecular dynamics 的 metadynamics 思想，对任意黑盒奖励在连续域上施加历史偏置势以逃离已访问模式，作为连续 GFN 的 off-policy 探索器，在多个连续/流形（球面、环面）环境上比既有探索策略更快收敛、发现更远模式。
- `N017` [Torsional-GFN: a conditional conformation generator for small molecules](https://arxiv.org/abs/2507.11759) · *预印本 2025 · arXiv:2507.11759*  
  在**超环面流形** \([0,2\pi]^m\) 上采样分子扭转角的连续 GFN，前/后向策略用 von Mises 混合参数化，并以 GNN 将策略**条件化**于分子图与局部结构，实现单一模型跨多分子的摊销采样（Vargrad 损失、Boltzmann 目标）。

## 2026 前沿方法 · Frontier Methods (2026)

- `T45` [Boosted GFlowNets: Improving Exploration via Sequential Learning](https://arxiv.org/abs/2511.09677) · *AISTATS 2026*  
  用一系列 residual-reward GFN 逐步修补当前模型欠覆盖的区域，再组合成 boosted sampler。
- `T46` [Controlling Exploration-Exploitation in GFlowNets via Markov Chain Perspectives](https://arxiv.org/abs/2602.01749) · *预印本 2026*  
  从 Markov-chain 视角引入可调参数 \(\alpha\)（方法名 \(\alpha\)-GFN），显式控制探索与利用强度。
- `T47` [Evaluating GFlowNet from Partial Episodes](https://iclr.cc/virtual/2026/poster/10007783) · *ICLR 2026*  
  提出 evaluation balance 和 partial-episode evaluator，在不完整轨迹上估计或诊断策略，并连接 value-based 与 policy-based GFN。
- `T48` [Loss-Guided Auxiliary Agents for Overcoming Mode Collapse in GFlowNets](https://ojs.aaai.org/index.php/AAAI/article/view/39613) · *AAAI 2026*  
  训练辅助 agent 专门访问主模型高 loss、低覆盖的区域，再把经验回流给主 GFN。
- `T49` [\(f\)-Trajectory Balance](https://icml.cc/virtual/2026/poster/61247) · *ICML 2026* [📝 深度解读](notes/T49_f_trajectory_balance.md) · [🇨🇳 中文PDF](pdfs/zh/T49_zh.pdf) · [📄 英文PDF](pdfs/en/T49_f_trajectory_balance.pdf)  
  建立 translation-invariant trajectory loss 与 \(f\)-divergence 的系统对应，使 TB loss 设计从经验试错转向 divergence 选择。
- `T50` [Avoid What You Know: Divergent Trajectory Balance for GFlowNets](https://icml.cc/virtual/2026/poster/62783) · *ICML 2026 主会*  
  ACE 同时维护 canonical GFN 和 exploration GFN，让后者针对主模型欠覆盖的高奖励区域收集数据。
- `T51` [Stable GFlowNets with Probabilistic Guarantees](https://arxiv.org/abs/2605.01729) · *预印本 2026*  
  尝试把可观测训练误差转化为全局分布距离或稳定性证书，直接回应 low-loss/high-TV 的风险。
- `T52` [GFlowState: Visualizing the Training of Generative Flow Networks Beyond the Reward](https://arxiv.org/abs/2604.21830) · *预印本 2026*  
  提出可视化和描述 GFlowNet 训练状态的诊断方法，避免只用平均奖励或单一 loss 判断模型。
- `T53` [Beyond the Proxy: Trajectory-Distilled Guidance for Offline GFlowNet Training](https://icml.cc/virtual/2026/poster/62632) · *ICML 2026*  
  TD-GFN 用逆强化学习从离线优质轨迹中蒸馏引导信号，同时让最终 GFN 更新仍依赖真实终点奖励。
- `T54` [Rooted Absorbed Prefix Trajectory Balance with Submodular Replay for GFlowNet Training](https://icml.cc/virtual/2026/poster/65366) · *ICML 2026*  
  RapTB 用 rooted/absorbed 前缀轨迹平衡为长序列提供前缀信用，并用 submodular replay 保持缓存多样性，针对长度偏置和稀疏奖励。
- `T55` [Routing by Reaching: Composition of Pre-trained GFlowNets for Multi-Objective Generation](https://icml.cc/virtual/2026/poster/61403) · *ICML 2026*  
  研究如何无需从头重训就组合多个已训练 GFN，以满足新的多目标奖励。
- `T56` [Spectral Flow Matching: Stabilizing Stochastic GFlowNets via Frequency-Domain Regularization](https://icml.cc/virtual/2026/poster/64150) · *ICML 2026*  
  用频域（谱）正则稳定随机 GFlowNet，提升噪声环境下的稳定性与稀疏奖励探索。
- `T57` [Proximal Policy Optimization for Amortized Discrete Sampling](https://arxiv.org/abs/2606.15793) · *ICML 2026 SPIGM Workshop*  
  从最大熵 RL 连接出发，将 PPO 式 clipped policy update 用于摊销离散采样。
- `O01` [Optimal Transport for Machine Learners](https://arxiv.org/abs/2505.06589) · *课程讲义/预印本 2025* [📝 深度解读](notes/O01_ot_for_machine_learners.md) · [📄 英文PDF](pdfs/en/O01_optimal_transport_for_machine_learners.pdf)  
  面向机器学习读者系统讲解 Monge、Kantorovich、对偶、动态 OT、Wasserstein 几何和梯度流。
- `O02` [A Framework for Wasserstein-1-Type Metrics](https://arxiv.org/abs/1701.01945) · *J. Convex Anal. 2019*  
  给出 Wasserstein-1 型距离的统一框架，将其推广到不同质量的非负测度之间（非平衡 OT），保持凸性与可计算性并涵盖多种已有度量。
- `O03` [GeONet: A Neural Operator for Learning the Wasserstein Geodesic](https://proceedings.mlr.press/v244/gracyk24a.html) · *UAI 2024*  
  用 neural operator 学习分布对之间的 Wasserstein geodesic，目标是跨任务摊销求解动态 OT。
- `O04` [Schrödinger Bridge Flow for Unpaired Data Translation](https://arxiv.org/abs/2409.09347) · *NeurIPS 2024 Spotlight*  
  研究带熵正则的动态 transport 和随机路径桥接，把起点分布逐步推向终点分布。
- `O05` [Universal Neural Optimal Transport](https://proceedings.mlr.press/v267/geuter25a.html) · *ICML 2025*  
  学习可条件化、可摊销的 neural OT map 或 plan，希望一个模型处理一族源—目标分布。
- `O06` [Computing High-Dimensional Optimal Transport by Flow Neural Networks](https://proceedings.mlr.press/v258/xu25f.html) · *AISTATS 2025*  
  用神经流处理连续高维 OT，重点解决传统离散 coupling 在维度和样本数上的扩展困难。
- `O07` [Learning Shortest Paths with Generative Flow Networks](https://arxiv.org/abs/2603.01786) · *ICML 2026 SPIGM Workshop* [📝 深度解读](notes/O07_shortest_paths_gflownets.md) · [🇨🇳 中文PDF](pdfs/zh/O07_zh.pdf) · [📄 英文PDF](pdfs/en/O07_learning_shortest_paths_with_generative_flow_networks.pdf)  
  研究在满足目标终止分布的多个可行内部流中，minimum-flow 准则为何偏向最短生成路径。
- `O08` [Your GFlowNet Secretly Learns an Optimal Transport Plan](https://arxiv.org/abs/2606.06272) · ***ICML 2026 SPIGM Workshop；非主会*** [📝 深度解读](notes/O08_gfn_secretly_ot_plan.md) · [🇨🇳 中文PDF](pdfs/zh/O08_zh.pdf) · [📄 英文PDF](pdfs/en/O08_your_gflownet_secretly_learns_an_optimal_transport_plan.pdf)  
  论文考虑非无环图、给定源分布与奖励诱导目标分布，并在固定初始流边缘下最小化总流量；在图最短路诱导的 transport cost 下，最优 GFlowNet 边流编码一个 Kantorovich 最优 coupling。

## GFlowNet × 最优传输 · GFlowNet × Optimal Transport

- `N037` [Discrete Diffusion Schrödinger Bridge Matching for Graph Transformation (DDSBM)](https://arxiv.org/abs/2410.01500) · *ICLR 2025*  
  用连续时间马尔可夫链把 Iterative Markovian Fitting 推广到高维离散/图空间求解 SB 并证明收敛；关键是把"独立修改节点与边"的参考动力学对应到 **以图编辑距离(GED)为 cost 的熵正则 OT(EOT)**，落地于分子优化（最小结构改动达成目标性质）。
- `N038` [Generalized Schrödinger Bridge on Graphs (GSBoG)](https://arxiv.org/abs/2602.04675) · *预印本 2026-02（v2）* [📝 深度解读](notes/N038_generalized_sb_on_graphs.md) · [🇨🇳 中文PDF](pdfs/zh/N038_zh.pdf) · [📄 英文PDF](pdfs/en/N038_generalized_schrodinger_bridge_on_graphs.pdf)  
  首个**数据驱动**的"图上广义 Schrödinger bridge"：以受控 CTMC + 迭代比例拟合 + TD 目标，学习满足源—目标边缘、并在 state-dependent running cost 下优化中间轨迹的**可执行图传输策略**，避免稠密全局 solver、可扩展到大稀疏图。
- `N039` [Entering the Era of Discrete Diffusion Models: A Benchmark for Schrödinger Bridges and Entropic Optimal Transport](https://arxiv.org/abs/2509.23348) · *预印本 2025-09（v2）*  
  首个**离散空间 SB/EOT 基准**：用 CP 参数化构造"解析已知 SB 解"的分布对，可严格评测 solver 是否真解 EOT/SB（而非只看 FID/MSE）；副产品给出 DLightSB、DLightSB-M、α-CSBM 三个求解器并在高维离散设置比较。
- `N040` [Minimal-Action Discrete Schrödinger Bridge Matching for Peptide Sequence Design (MadSBM)](https://arxiv.org/abs/2601.22408) · *预印本 2026-01（v1）*  
  把肽序列生成建模为**氨基酸编辑图上的受控 CTMC**，以预训练蛋白语言模型 logits 作参考过程，学习时间相关控制场以走"最小作用量/低成本"传输路径；并首次给离散 SB 加**分类器引导**。
- `N041` [Unsupervised Learning for Optimal Transport plan prediction between unbalanced graphs (ULOT)](https://papers.nips.cc/paper_files/paper/2025/hash/873fd89b3e4db1f6242c2333673e104d-Abstract-Conference.html) · *NeurIPS 2025 主会 · arXiv:2506.12025* [📝 深度解读](notes/N041_unsupervised_ot_unbalanced_graphs.md) · [🇨🇳 中文PDF](pdfs/zh/N041_zh.pdf) · [📄 英文PDF](pdfs/en/N041_unsupervised_ot_plan_unbalanced_graphs.pdf)  
  用 GNN + cross-attention、并**以 FUGW 权衡超参为条件**，无监督地预测两图之间的（不平衡）OT plan，比经典 solver 快约两个数量级，且预测 plan 可为经典 solver 提供 **warm-start**、对输入与超参可微。
- `N042` [Modeling Stochastic Conditional Dynamics from Sparse Observations via Kernel-Stabilized Flow Matching (CVFM)](https://arxiv.org/abs/2411.08314) · *TMLR 2026*  
  提出 **Conditional Variable Flow Matching**：在**连续条件密度空间**上摊销学习条件分布之间的流；用条件 Wasserstein 距离 + "条件失配核"抑制稀疏/不配对数据下 mini-batch 条件耦合的方差爆炸，并可扩展逼近**条件 Schrödinger bridge**。
- `N043` [Diffusion Schrödinger Bridge Matching (DSBM)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/c428adf74782c2092d254329b6b02482-Abstract.html) · *NeurIPS 2023 · arXiv:2303.16852* [📝 深度解读](notes/N043_diffusion_sb_matching.md) · [📄 英文PDF](pdfs/en/N043_diffusion_schrodinger_bridge_matching.pdf)  
  提出 **Iterative Markovian Fitting (IMF)** 与 DSBM：用一系列"简单回归"迭代逼近 SB，而 **SB 恰恰恢复熵正则版 OT**，避免旧 DSB 的时间离散化与"遗忘"误差累积。
- `N044` [Generalized Schrödinger Bridge Matching (GSBM)](https://arxiv.org/abs/2310.02233) · *ICLR 2024*  
  广义 SB matching：在固定首末边缘下支持**依赖状态与分布的一般 running cost**，用变分/条件模拟分解逐步求解。
- `N045` [Light Schrödinger Bridge (LightSB)](https://openreview.net/forum?id=WhZoCLRWYJ) · *ICLR 2024 · arXiv:2310.01174*  
  轻量、**免模拟**的 SB/EOT 求解器：用 sum-exp 二次型参数化 Schrödinger potential + 能量视角，中等维度几分钟即可求 SB；并证明其为 **SB 的通用逼近器**、给出**泛化误差分析**。
- `N046` [Categorical Schrödinger Bridge Matching (CSBM)](https://proceedings.mlr.press/v267/ksenofontov25a.html) · *ICML 2025 · arXiv:2502.01416* [📝 深度解读](notes/N046_categorical_schrodinger_bridge.md) · [📄 英文PDF](pdfs/en/N046_categorical_schrodinger_bridge_matching.pdf)  
  为**离散时间/离散空间 SB** 提供理论与算法基座：证明**离散时间 IMF (D-IMF) 在有限空间、一般 Markov 参考过程下收敛到唯一 SB**，据此给出可"少步生成"的 Categorical SB Matching（VQ 图像/合成数据验证）。
- `N047` [Neural Optimal Transport](https://openreview.net/forum?id=d8CBRlWNkqH) · *ICLR 2023 Spotlight*  
  Korotin 等提出基于鞍点对偶的神经 OT 算法，可同时求强/弱 cost 下的确定性 map 与随机 plan，并证明神经网络是 transport plan 的通用逼近器。
- `N048` [Neural Optimal Transport with General Cost Functionals](https://openreview.net/forum?id=gIiz7tBtYZ) · *ICLR 2024 poster*  
  把 NOT 推广到一般 cost functional（不限 \(\ell^1/\ell^2\)），可编码类别保持、成对约束等任务信息，并给出恢复 plan 的误差分析。
- `N049` [The Monge Gap: A Regularizer to Learn All Transport Maps](https://proceedings.mlr.press/v202/uscidda23a.html) · *ICML 2023（PMLR 202:34709-34733）*  
  提出 Monge gap 正则子：度量任意映射 \(T\) 偏离 c-OT 最优性的程度，摆脱 ICNN 架构约束与平方欧氏 cost 限制，用"拟合损失 + 最优性正则"学任意 cost 的 transport map。
- `N050` [Progressive Entropic Optimal Transport Solvers](https://openreview.net/forum?id=7WvwzuYkUq) · *NeurIPS 2024 poster*  
  Apple/NYU 提出 ProgOT：借动态 OT 的时间离散把质量位移拆成多步，每步用调度好的 \(\varepsilon\) 跑 Sinkhorn，大规模下比标准 EOT 更快更稳，甚至优于部分神经求解器，并证明 map 估计的统计一致性。
- `N051` [Semidefinite Relaxations of the Gromov-Wasserstein Distance](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8189d86a5d8dea0694d43bb90e01c14d-Abstract.html) · *NeurIPS 2024*  
  GW 距离本质是非凸二次规划，既有求解器只能到局部最优。
- `N052` [Any2Graph: Deep End-To-End Supervised Graph Prediction With An Optimal Transport Loss](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b81a352c156ca123c30c740f147a4496-Abstract.html) · *NeurIPS 2024*  
  提出部分掩码融合 GW（PMFGW）损失：置换不变、可微、尺寸无关，让深度模型端到端地从任意模态预测整图（卫星图→路网、指纹→分子）。
- `N053` [A Convergent Single-Loop Algorithm for Relaxation of Gromov-Wasserstein in Graph Data](https://openreview.net/forum?id=0jxPyVWmiiF) · *ICLR 2023 poster*  
  提出 BAPG：首个有收敛保证的单循环 GW 近似算法，用放松耦合可行性换取效率，基于 Luo-Tseng 误差界给出不动点集与 GW 临界点集的距离界，在图对齐/图划分上快且好。
- `N054` [Estimating Barycenters of Distributions with Neural Optimal Transport](https://proceedings.mlr.press/v235/kolesov24a.html) · *ICML 2024 poster*  
  把 NOT 的对偶鞍点法从两边缘推广到 Wasserstein barycenter：用双层对抗目标取代既有方法的三层优化，支持一般 cost，并给出理论误差界。

## 扩散采样器与随机最优控制 · Diffusion Samplers & SOC

- `N029` [Improved sampling via learned diffusions](https://proceedings.iclr.cc/paper_files/paper/2024/hash/2c9f31add52c28c2db39329b464bce91-Abstract-Conference.html) · *ICLR 2024 poster · arXiv:2307.01198* [📝 深度解读](notes/N029_learned_diffusion_sampling.md) · [🇨🇳 中文PDF](pdfs/zh/N029_zh.pdf) · [📄 英文PDF](pdfs/en/N029_improved_sampling_via_learned_diffusions.pdf)  
  Richter 与 Berner 把 PIS、DIS、DDS、Schrödinger bridge 统一为"广义 Schrödinger 桥"问题，在时间反转路径测度之间定义变分散度族；提出 log-variance loss，避免对 SDE 求解器求导、显著抑制 mode collapse。
- `N030` [Beyond ELBOs: A Large-Scale Evaluation of Variational Methods for Sampling](https://proceedings.mlr.press/v235/blessing24a.html) · *ICML 2024 · arXiv:2406.07423*  
  Blessing、Vargas、Neumann 等建立神经采样器统一基准：标准化任务套件覆盖 SMC、AFT、CRAFT、FAB、GMMVI 及各类 diffusion sampler，系统研究 ELBO/EUBO 等指标何时掩盖 mode collapse，并提出熵模式覆盖等新度量。
- `N031` [Iterated Denoising Energy Matching for Sampling from Boltzmann Densities](https://proceedings.mlr.press/v235/akhound-sadegh24a.html) · *ICML 2024 poster · arXiv:2402.06121*  
  Mila 团队（含 Bengio、Malkin、Tong）提出 iDEM：只用能量函数及其梯度、不需数据样本的迭代式随机 score matching，内环模拟自由、外环用模型自采样探索，凭 diffusion 的快速模式混合平滑能量面，首次在 LJ-55（165 维）粒子系统上用纯能量训练成功，训练快 2–5 倍。
- `N032` [Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models with Memoryless Stochastic Optimal Control](https://openreview.net/forum?id=xQBRrtQM8u) · *ICLR 2025 **Spotlight** · arXiv:2409.08861* [📝 深度解读](notes/N032_adjoint_matching_soc_finetuning.md) · [📄 英文PDF](pdfs/en/N032_adjoint_matching_fine_tuning_flow_and_diffusion_generative_m.pdf)  
  Domingo-Enrich、Chen（Meta）把 reward 微调严格表述为 SOC 问题：证明必须使用 memoryless 噪声调度才能无偏收敛到 reward-tilted 分布，并把 SOC 化为回归式 Adjoint Matching 目标。
- `N033` [Adjoint Sampling: Highly Scalable Diffusion Samplers via Adjoint Matching](https://proceedings.mlr.press/v267/havens25a.html) · *ICML 2025 · arXiv:2504.11713*  
  Meta FAIR 将 Adjoint Matching 特化为从未归一化密度采样：Reciprocal Adjoint Matching 加 replay buffer，使梯度更新次数远超能量评估次数，成为首个如此可扩展的 on-policy 方法；支持 SE(3) 对称与周期边界，扩展到神经能量函数上的摊销构象生成并开源基准。
- `N034` [Feynman-Kac Correctors in Diffusion: Annealing, Guidance, and Product of Experts](https://proceedings.mlr.press/v267/skreta25a.html) · *ICML 2025 spotlight poster · arXiv:2503.02819*  
  Skreta、Akhound-Sadegh、Doucet、Brekelmans、Tong、Neklyudov 等基于 Feynman–Kac 公式推导加权模拟方案：不重训即可从退火、几何平均或专家乘积分布精确采样，用 SMC 重采样做推断时扩展，应用于温度退火摊销采样、多目标分子生成与 CFG 改进。
- `N035` [Sequential Controlled Langevin Diffusions](https://openreview.net/forum?id=dImD2sgy86) · *ICLR 2025 poster · arXiv:2412.07081*  
  Chen、Richter、Berner、Blessing 等在路径测度上统一 SMC 与学习式 diffusion sampler：SMC 的重采样/MCMC 提供稳健性与渐近保证，学习的控制漂移提供适应性，SCLD 常以先前 diffusion sampler 约 10% 的训练预算达到更好性能。
- `N036` [Proximal Diffusion Neural Sampler](https://openreview.net/forum?id=XTHQqS7ObC) · *ICLR 2026*  
  Guo、Choi、Tao、Y. Chen（Georgia Tech）用路径测度上的 SOC 统一连续（SDE）与离散（CTMC）神经采样器，指出一次性全局优化会加剧 mode collapse，改用近端点法把学习拆成一串 KL 约束子问题，在分子动力学与 Ising/Potts 等连续和离散基准上达 SOTA。

## 分子、蛋白与材料 · Molecules, Proteins & Materials

- `A01` [GFlowNets for AI-Driven Scientific Discovery](https://pubs.rsc.org/en/content/articlelanding/2023/dd/d3dd00002h) · *Digital Discovery 2023* [📝 深度解读](notes/A01_gfn_scientific_discovery.md) · [📄 英文PDF](pdfs/en/A01_gflownets_for_ai_driven_scientific_discovery.pdf)  
  系统回顾 GFlowNet 在分子、蛋白、材料、因果发现和主动学习中的作用，并解释为什么科学发现往往需要一组多样候选而非单个最优解。
- `A02` [Biological Sequence Design with GFlowNets](https://proceedings.mlr.press/v162/jain22a.html) · *ICML 2022*  
  将序列逐步构造为 GFN 环境，并与主动学习结合，在昂贵 oracle 下寻找多样高适应度生物序列。
- `A03` [Multi-Objective GFlowNets](https://proceedings.mlr.press/v202/jain23a.html) · *ICML 2023*  
  用偏好向量条件化 GFN，在一次训练后为不同 Pareto 权衡采样多样解。
- `A04` [Multi-Fidelity Active Learning with GFlowNets](https://arxiv.org/abs/2306.11715) · *TMLR 2024*  
  联合选择候选对象与评估保真度，在低成本近似和高成本真实实验之间分配预算。
- `A05` [Towards Equilibrium Molecular Conformation Generation with GFlowNets](https://pubs.rsc.org/en/content/articlelanding/2024/dd/d4dd00023d) · *Digital Discovery 2024, 3(5):1038–1047*  
  以分子构象能量定义近似 Boltzmann 目标，探索 GFN 对连续/几何构象分布的建模。
- `A06` [Crystal-GFN](https://arxiv.org/abs/2310.04925) · *预印本 2023*  
  将晶体结构按组成、空间群和几何参数逐步生成，并纳入有效性及材料性质约束。
- `A07` [PhyloGFN](https://iclr.cc/virtual/2024/poster/18107) · *ICLR 2024*  
  把系统发育树的逐步合并过程建模为 DAG，在给定序列数据下摊销采样树后验。
- `A08` [Genetic-guided GFlowNets](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4b25c000967af9036fb9b207b198a626-Abstract-Conference.html) · *NeurIPS 2024*  
  将遗传算法产生的结构化搜索经验蒸馏到 off-policy GFN 中，以加速高奖励模式发现。
- `A09` [RGFN: Synthesizable Molecular Generation Using GFlowNets](https://proceedings.neurips.cc/paper_files/paper/2024/hash/53704142f230054140418ecd8857f391-Abstract-Conference.html) · *NeurIPS 2024*  
  直接在化学反应和可用构件空间中生成分子，使样本天然附带合成路径。
- `A10` [GFlowNet Assisted Biological Sequence Editing](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c14760740573001c0d18d58879a6a305-Abstract-Conference.html) · *NeurIPS 2024*  
  从给定 seed sequence 出发进行少量但多样的编辑，而不是完全从头生成。
- `A11` [SynFlowNet: Design of Diverse and Novel Molecules with Synthesis Constraints](https://iclr.cc/virtual/2025/poster/27946) · *ICLR 2025*  
  以化学反应和可购买 reactant 为动作，保证生成分子可由显式路径合成。
- `A12` [Pretraining Generative Flow Networks with Inexpensive Rewards for Molecular Graph Generation](https://proceedings.mlr.press/v267/pandey25b.html) · *ICML 2025*  
  先用廉价 proxy reward 预训练原子级分子 GFN，再适配昂贵性质目标。
- `A13` [Synergy of GFlowNet and Protein Language Model Makes a Diverse Antibody Designer](https://ojs.aaai.org/index.php/AAAI/article/view/34370) · *AAAI 2025*  
  将蛋白语言模型先验与效力、可开发性等多项 reward 组合，用 GFN 生成多样抗体序列。
- `A14` [LeakGFN](https://icml.cc/virtual/2026/poster/63060) · *ICML 2026*  
  区分真正有效分子流和因截断、无效动作产生的“泄漏”终止流，减少化学任务中的目标分布污染。
- `A15` [Synthesizable Molecular Generation via Soft-constrained GFlowNets with Rich Chemical Priors (S3-GFN)](https://icml.cc/virtual/2026/poster/64424) · *ICML 2026*  
  结合软约束和大规模 SMILES 先验，提高生成分子的有效性与可合成率，同时保留 GFN 的多样性目标。
- `A16` [A Distributional Framework for Generative Modeling of Molecular Crystals](https://arxiv.org/abs/2607.05266) · *预印本 2026*  
  面向分子晶体建立分布式而非单点优化的生成框架，以覆盖多个稳定候选结构。
- `N055` [TacoGFN: Target-conditioned GFlowNet for Structure-based Drug Design](https://openreview.net/forum?id=N8cPv95zOU) · *TMLR 2024*  
  首个把 SBDD 表述为"以蛋白口袋为条件、按 affinity×药性×可合成度 reward 成比例采样"的 GFlowNet：不拟合有限 protein-ligand 复合物数据分布，而对全体口袋诱导的 reward 分布建模，并用基于 pharmacophore 的快速对接预测器压低 reward 成本、支持训练中评估数百万分子。
- `N056` [Generative Flows on Synthetic Pathway for Drug Design（RxnFlow）](https://openreview.net/forum?id=pB1XSj2y4X) · *ICLR 2025*  
  在"反应模板 + 可购买 building block"的合成路径空间上训练 GFlowNet，天然保证可合成性；核心创新 action space subsampling 让其能在 120 万 building block × 71 反应模板的超大动作空间上训练而无显著开销，且不重训即可更换/扩充 building block 或新增目标（如溶解度）。
- `N057` [Compositional Flows for 3D Molecule and Synthesis Pathway Co-design（CGFlow）](https://proceedings.mlr.press/v267/shen25b.html) · *ICML 2025*  
  提出 Compositional Generative Flows：把 flow matching 扩展到"逐步构造组合对象 + 建模连续状态"，并接入 GFlowNet 理论实现 reward-guided 采样；据此的 3DSynthFlow 同时共设计分子的合成路径与 3D 结合构象。
- `N058` [Sample-efficient Multi-objective Molecular Optimization with GFlowNets（HN-GFN）](https://proceedings.neurips.cc/paper_files/paper/2023/hash/fbc9981dd6316378aee7fd5975250f21-Abstract.html) · *NeurIPS 2023*  
  将多目标分子优化纳入多目标贝叶斯优化（MOBO），以单个 preference-conditioned 的 hypernetwork-GFlowNet 作为 acquisition 优化器，从近似 Pareto front 采样一批多样候选；并提出 hindsight 式 off-policy 策略在不同偏好间共享高分子以加速学习。
- `N069` [Efficient Symmetry-Aware Materials Generation via Hierarchical Generative Flow Networks（SHAFT / CHGlowNet）](https://pubs.rsc.org/en/content/articlelanding/2026/dd/d4dd00392f) · *Digital Discovery（RSC） · arXiv:2411.04323*  
  在 Crystal-GFN 仅生成空间群/组成/晶格参数、不含原子坐标的基础上，首次用 GFlowNet 生成**含原子坐标的完整晶体**：先建 flat GFlowNet 基线，再提出分层且对称性感知的 SHAFT，按空间群→晶格→原子逐层分解为子目标，缩短长程轨迹并利用对称性；在有效性、稳定性、多样性上超过 flat GFN、CDVAE 与 DiffCSP。
- `N070` [Catalyst GFlowNet for electrocatalyst design: A hydrogen evolution reaction case study](https://arxiv.org/abs/2510.02142) · *预印本 arXiv:2510.02142（2025-10）*  
  首个把 GFlowNet 用于**电催化剂设计**：基于 Crystal-GFN 构造周期性晶体并切出催化表面，用 FAENet(GNN) 预测吸附能、并做 ML 结构弛豫，以形成能与吸附能构造奖励，从而采样一组多样催化剂候选而非单一最优。
- `N071` [Collective Variable Free Transition Path Sampling with Generative Flow Network（TPS-GFN）](https://arxiv.org/abs/2405.19961) · *预印本 arXiv:2405.19961*  
  把分子亚稳态间的**过渡路径采样(TPS)**重构为对分子轨迹的摊销式能量采样，用 GFlowNet(one-step TB) 训练偏置势，无需依赖昂贵的集体变量(CV)，并借 replay buffer 做 off-policy 以避免模式坍缩。

## 结构学习与组合优化 · Structure Learning & Combinatorial Optimization

- `A17` [Bayesian Structure Learning with GFlowNets](https://proceedings.mlr.press/v180/deleu22a.html) · *UAI 2022*  
  将有向无环图逐边构造，并按 Bayesian score 近似采样因果结构后验。
- `A18` [Joint Bayesian Inference of Graphical Structure and Parameters with a Single GFlowNet](https://neurips.cc/virtual/2023/poster/70228) · *NeurIPS 2023*  
  在单一 GFN 中联合表示图结构及其连续或离散参数后验，避免结构学习后再单独拟合参数。
- `A19` [Let the Flows Tell: Solving Graph Combinatorial Problems with GFlowNets](https://proceedings.neurips.cc/paper_files/paper/2023/hash/27571b74d6cd650b8eb6cf1837953ae8-Abstract.html) · *NeurIPS 2023*  
  把若干图组合优化问题写成逐步构造过程，并用 GFN 生成多样近优解。
- `A20` [Robust Scheduling with GFlowNets](https://arxiv.org/abs/2302.05446) · *ICLR 2023*  
  用 GFN 生成一组对不确定扰动具有不同权衡的调度方案，从而提高决策鲁棒性。
- `N059` [Expert-Aided Causal Discovery of Ancestral Graphs](https://doi.org/10.1016/j.ins.2026.123816) · *Information Sciences, Vol. 756, Art. 123816 · arXiv:2309.12032*  
  首个在潜在混杂（不假设因果充分性）下做分布式推断的 GFlowNet 因果发现方法：按 BIC 类得分成比例采样祖先图（AG），用最优实验设计主动向专家或 LLM 提问，再以重要性加权把带噪反馈并入后验而无需重训，并证明反馈足够准确时收敛到真实祖先图。
- `N060` [Learning Equivalence Classes of Bayesian Network Structures with GFlowNet](https://openreview.net/forum?id=FAcc7oAdaa) · *TMLR 2025*  
  提出 CPDAG-GFN：不在 DAG 空间、而直接在 Markov 等价类（CPDAG）空间上用 GFlowNet 学习后验并抽取高分候选，配合偏稀疏过滤器改进与真图的对齐。
- `N061` [Generative Flow Networks: Theory and Applications to Structure Learning](https://arxiv.org/abs/2501.05498) · *博士论文 · arXiv:2501.05498* [📝 深度解读](notes/N061_deleu_phd_thesis_gfn.md) · [📄 英文PDF](pdfs/en/N061_generative_flow_networks_theory_and_applications_to_structur.pdf)  
  DAG-GFlowNet（A17）与 JSP-GFN（A18）一作 Deleu 的博士论文：上篇系统建立 GFlowNet 数学基础及其与变分推断、强化学习的联系和连续空间扩展；下篇完整展开贝叶斯结构学习——在观测与干预数据下对 DAG 结构及机制参数做联合后验近似。
- `N062` [Symmetric Replay Training: Enhancing Sample Efficiency in Deep RL for Combinatorial Optimization (SRT)](https://arxiv.org/abs/2306.01276) · ***ICML 2024 主会***  
  提出对称回放训练：利用 CO 解空间的对称性（多条 (partial) 轨迹通向同一状态，类似 GFN 后向策略 \(P_B\)），周期性用最大似然对**对称轨迹**做直接信用分配，显著提升样本效率。

## LLM、推理与视觉 · LLMs, Reasoning & Vision

- `A21` [Amortizing Intractable Inference in Large Language Models](https://proceedings.iclr.cc/paper_files/paper/2024/hash/bc667ac84ef58f2b5022da97a465cbab-Abstract-Conference.html) · *ICLR 2024*  
  用 GFlowNet 微调语言模型，使其按后验权重采样潜在推理或文本变量，而不是依赖逐查询 MCMC。
- `A22` [Amortizing Intractable Inference in Diffusion Models for Vision, Language, and Control](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8b21a7ea42cbcd1c29a7a88c444cce45-Abstract-Conference.html) · *NeurIPS 2024*  
  用 relative TB 学习 diffusion 模型中的难后验，覆盖视觉、语言和控制任务。
- `A23` [Efficient Diversity-Preserving Diffusion Alignment via Gradient-Informed GFlowNets (Nabla-GFlowNet)](https://iclr.cc/virtual/2025/poster/30600) · *ICLR 2025*  
  利用可微 reward 的梯度引导 diffusion/GFN alignment，在提升目标性质时尽量保持预训练生成先验和输出多样性。
- `A24` [COFlowNet](https://iclr.cc/virtual/2025/poster/28047) · *ICLR 2025*  
  在没有在线 oracle、只能依赖离线数据时约束流进入数据支持之外的区域，以减少虚假的高奖励外推。
- `A25` [Flow of Reasoning: Training LLMs for Divergent Reasoning with Minimal Examples](https://proceedings.mlr.press/v267/yu25k.html) · *ICML 2025*  
  用少量可验证样例训练语言模型生成多条高质量推理路径，强调 divergent reasoning 而非单一路径模仿。
- `A26` [EraseFlow](https://proceedings.neurips.cc/paper_files/paper/2025/hash/66c9de41210338c9581d5313125b7486-Abstract-Conference.html) · *NeurIPS 2025*  
  用 GFN 探索 diffusion 模型中多样的去概念/擦除轨迹，以避免单一路径造成能力损伤。
- `A27` [Discovering Latent Graphs with GFlowNets](https://proceedings.neurips.cc/paper_files/paper/2025/hash/96c6f409a374b5c81d2efa4bc5526f27-Abstract-Conference.html) · *NeurIPS 2025*  
  学习可解释的潜在图结构，再条件化图像生成，以覆盖多种关系配置。
- `A28` [GFlowVLM: Enhancing Multi-step Reasoning in Vision-Language Models with Generative Flow Networks](https://openaccess.thecvf.com/content/CVPR2025/html/Kang_GFlowVLM_Enhancing_Multi-step_Reasoning_in_Vision-Language_Models_with_Generative_Flow_CVPR_2025_paper.html) · *CVPR 2025*  
  将视觉语言模型的多步推理轨迹作为 GFN 构造过程，训练模型覆盖多条高回报 reasoning/planning 路径。
- `A29` [Learning to Sample Effective and Diverse Prompts for Text-to-Image Generation](https://openaccess.thecvf.com/content/CVPR2025/html/Yun_Learning_to_Sample_Effective_and_Diverse_Prompts_for_Text-to-Image_Generation_CVPR_2025_paper.html) · *CVPR 2025*  
  用 GFlowNet 学习一族有效且多样的文本提示，并通过 reward decomposition 缓解 prompt optimizer 的 plasticity loss。
- `A30` [Flow of Spans](https://iclr.cc/virtual/2026/poster/10007998) · *ICLR 2026*  
  使用动态 span vocabulary，使同一文本可通过不同粒度的 token/span 路径构造，从树式自回归生成提升为 DAG。
- `A31` [AlphaSAGE](https://iclr.cc/virtual/2026/poster/10006456) · *ICLR 2026*  
  将 GFN 用于结构化量化 alpha 因子挖掘，目标是得到高质量且彼此低相关的一组策略。
- `A32` [Stable-GFlowNet for LLM Red-Teaming](https://icml.cc/virtual/2026/poster/64302) · *ICML 2026 Spotlight*  
  用 pairwise contrastive TB 和噪声 reward masking 训练稳定、多样的红队提示生成器，并避免显式估计 \(Z\)。
- `A33` [GFlowRL](https://arxiv.org/abs/2607.13394) · *预印本 2026*  
  将 distribution-matching RL 扩展到 dense 与 MoE 语言模型，希望让策略按奖励诱导分布覆盖多种高质量响应。
- `A34` [PowerFlow: Unlocking the Dual Nature of LLMs via Principled Distribution Matching](https://arxiv.org/abs/2603.18363) · *ICML 2026*  
  研究 LLM 分布匹配训练中的能力—多样性权衡，尝试用 GFlowNet 式目标控制响应覆盖。
- `A35` [GFlowPO](https://arxiv.org/abs/2602.03358) · *预印本 2026*  
  用 GFlowNet 对离散 prompt 或策略候选进行分布式优化，强调保留多个有效解而非收敛到单一提示。
- `N063` [Learning Diverse Attacks on Large Language Models for Robust Red-Teaming and Safety Tuning](https://arxiv.org/abs/2405.18540) · *ICLR 2025 Poster*  
  Lee、Kim、Malkin、Jain、Bengio 等（KAIST/Mila）提出两阶段红队方法：先以毒性×似然为奖励做 GFlowNet 微调采样攻击提示，再收集高奖励提示做 MLE 平滑。
- `N064` [GFlowNet Fine-tuning for Diverse Correct Solutions in Mathematical Reasoning Tasks](https://arxiv.org/abs/2410.20147) · *预印本 2024-10 · arXiv:2410.20147*  
  首个在 GSM8K/MATH 上系统对比 GFlowNet 微调与 PPO/DPO/RFT 的实证研究：GFlowNet 在 Pass@8 与各基线相当的前提下，采 8 个解时"不同正确解"数量最高（GSM8K 上 2.34 对 PPO 的 1.73），说明分布匹配微调能从多样中间推理步收敛到同一正确答案。
- `N065` [Accurate and Diverse LLM Mathematical Reasoning via Automated PRM-Guided GFlowNets](https://arxiv.org/abs/2504.19981) · *预印本 2025-04 · arXiv:2504.19981*  
  用 MCTS + 相似度数据增强自动训练过程奖励模型（PRM），再把 GFlowNet（SubTB 变体）从 token 级提升到**推理步级**：状态为部分解、动作为完整推理步。
- `N066` [Proof Flow: Preliminary Study on Generative Flow Network Language Model Tuning for Formal Reasoning](https://arxiv.org/abs/2410.13224) · *NeurIPS 2024 Workshop*  
  把 GFlowNet 微调引入 Lean 神经定理证明（NTP）：以 tactic 为动作、可验证证明为奖励，基于 ReProver（350M）初始化。
- `N067` [Latent Logic Tree Extraction for Event Sequence Explanation from LLMs](https://proceedings.mlr.press/v235/song24j.html) · *ICML 2024 Poster，PMLR 235:46238-46258*  
  LaTee 用摊销 EM 从 LLM 中抽取解释事件序列的潜逻辑树：E 步以 LLM 先验 × 时序点过程似然定义后验，用 GFlowNet 微调 LLM 采样多样逻辑树；M 步更新点过程参数并精调 LLM 先验。
- `N068` [Latent Thought Flow: Efficient Latent Reasoning in Large Language Models](https://arxiv.org/abs/2606.16222) · *预印本 2026-06 · arXiv:2606.16222*  
  LTF 把推理建模为**变长连续潜思维轨迹**，用 continuous GFlowNet（随机潜转移）训练采样器匹配"答案质量 × 计算成本"诱导的后验；提出熵加权 SubTB 处理稀疏监督，并用参考先验正则稳定探索。

## 多目标与条件生成 · Multi-objective & Conditional Generation

- `N072` [Goal-conditioned GFlowNets for Controllable Multi-Objective Molecular Design](https://arxiv.org/abs/2306.04620) · ***ICML 2023 Workshop***  
  针对偏好标量化在**凹 Pareto 前沿**上把解推向极端点的缺陷，改用"目标区域"硬约束条件化 GFN：仅当样本目标向量与目标方向的余弦相似度超过阈值（focus region 圆锥）才给奖励，并用可学习的表格式目标采样器 Tab-GS + hindsight replay 缓解奖励稀疏与不可行目标。
- `N073` [Global-Order GFlowNets](https://arxiv.org/abs/2504.02968) · ***预印本 2025***  
  指出 OP-GFN 按批内 Pareto 支配施加的"局部序"会产生**相互冲突的训练目标**、导致优化不一致；提出用 global-rank 或最近邻等方式把局部偏序提升为与 Pareto 支配相容的"全局（弱）全序"，据此定义学习型奖励再训练 GFN。
- `N074` [Amortized Active Generation of Pareto Sets (A-GPS)](https://neurips.cc/virtual/2025/poster/116473) · ***NeurIPS 2025 主会 Poster***  
  面向在线离散黑盒多目标优化，学习一个支持"事后偏好条件"的 Pareto 集生成模型（建立在 VSD 之上）：用类别概率估计器 CPE 判别**非支配**关系来引导生成，并证明该非支配 CPE 隐式估计 hypervolume 改进概率（PHVI）；再以偏好方向向量做摊销变分推断，实现免重训的偏好可控采样，绕开显式 hypervolume 计算与标量化。

## 安全、红队与对齐 · Safety, Red-teaming & Alignment

- `N075` [GFlowNets with Human Feedback (GFlowHF)](https://arxiv.org/abs/2305.07036) · ***ICLR 2023 Tiny Papers Track**（两页、**非归档短文、非主会**）*  
  已知最早的"GFlowNet + 人类反馈"框架：用人类评分拟合奖励，学习严格**正比于评分**的策略，而非像 RLHF 只追逐最高分，从而获得更强探索与更高多样性；并论证对约 10% 噪声标签比 RLHF 更鲁棒。
- `N076` [Generating Attacks for LLMs with GFlowNets](https://arxiv.org/abs/2608.10171) · ***预印本 2026-08** · arXiv:2608.10171*  
  直接沿用 NEW-E19-1 的两阶段 GFlowNet+MLE 红队框架，把攻击提示生成扩展到**英语与土耳其语的多语言场景**，用 attacker / victim / evaluator 三模型闭环训练并给出定量鲁棒性分数。

## 其他应用 · Other Applications

- `N077` [Generative Flow Network for Listwise Recommendation (GFN4Rec)](https://arxiv.org/abs/2306.02239) · *KDD 2023*  
  面向**列表级推荐**：把"为用户生成一个物品列表"建模为 GFN 的逐步构造过程，用 log-scale reward matching 损失让列表生成概率与其整体效用对齐，并用自回归选择模型刻画列表内物品互相影响。
- `N078` [Ant Colony Sampling with GFlowNets for Combinatorial Optimization (GFACS)](https://proceedings.mlr.press/v258/kim25a.html) · *AISTATS 2025*  
  把 GFlowNet 与蚁群优化（ACO）分层结合：先用 GFN 摊销一个覆盖高奖励且多样解的**多峰先验**，再以 ACO 式并行随机搜索迭代更新为逼近近优解的后验。
- `N079` [Discovery of Diverse and Realistic Financial Tail-Risk Using GFlowNets (GRID)](https://openreview.net/forum?id=YHiS8knV3s) · ***投稿 ICLR 2026（under review）*  
  把**金融尾部风险情景生成**建模为在**连续状态空间**上逐步构造宏观经济轨迹：每步由 GFN 预测动作分布参数（可用高斯/Beta 混合等灵活分布族）并采样转移，终点由预测 oracle 给标量 reward，用 flow-matching 训练。

## 评测基准与软件 · Benchmarks & Software

- `N080` [Evaluating Generalization in GFlowNets for Molecule Design](https://openreview.net/forum?id=JFSaHKNZ35b) · *ICLR 2022 MLDD Workshop（**奠基作，2022**）*  
  针对分子设计"生成一批多样高分候选"的目标，系统比较多种候选评价指标，提出 **TopKDiverse**（Tanimoto 多样性约束下取 top-K 平均分）刻画下游搜索性能，并发现 **GFNEvalS**（对齐采样概率与目标奖励分布）比 flow error / top-k 更能预测泛化。
- `N081` [Benchmarking GFlowNets against MCMC: The Role of Peak Sharpness and Dimensionality](https://jac.ut.ac.ir/article_106220_565b5b56aeb6a2e1813f39d7ffebcd62.pdf) · *J. of Algorithms and Computation 57（2）*  
  在 HyperGrid 上系统对比 TB/DB/FM 与 Metropolis–Hastings 对"奖励地形几何（峰宽/尖锐）"与维度的敏感性，挑战"学习式采样器普遍更优"的默认叙事：尖峰/低维/近 Dirac 目标上 MCMC 更稳且快约 500×，宽峰多模态时 GFlowNet-DB 可多发现约 4.75× 模式。

## 审查流水线补录 · Additional Curated Papers

- `N082` [Path-dependent Discrete Amortized Inference](https://arxiv.org/abs/2608.08644) · *ICML 2026 **Oral*** [📝 深度解读](notes/N082_path_dependent_amortized_inference.md) · [🇨🇳 中文PDF](pdfs/zh/N082_zh.pdf) · [📄 英文PDF](pdfs/en/N082_path_dependent_discrete_amortized_inference.pdf)  
  证明 GFlowNet 类离散摊销采样器的 Markov 假设会阻碍训练中的信号传播，且因 state aliasing 灾难性地限制可表达的终止分布；提出用可学习潜动力系统提升 MDP，使策略依赖整条历史轨迹（路径依赖），并把现有摊销采样器训练算法可证地扩展到该非 Markov 设定，实验显示更快收敛与更好探索。
- `N083` [Particle GFlowNets: Rethinking Generative Marginalization Models](https://proceedings.mlr.press/v337/silva26a.html) · *UAI 2026 主会（PMLR 337:6366–6383）*  
  证明生成边缘化模型（MaM，为任意阶自回归离散建模同时学习边缘与条件概率）与 GFlowNet 等价——此前两者被视为不同范式；进而把 MaM 的采样策略推广到非自回归生成过程，用由 Gelman–Rubin 统计量导出的自动准则对持久 Gibbs 采样器做全状态重启（rejuvenation），显著加速大组合空间中的训练收敛。
- `N084` [Interpreting GFlowNets for Drug Discovery: What Probes Can and Cannot Show](https://arxiv.org/abs/2511.19264) · *NeurIPS 2025 WiML Workshop / MoML 2025*  
  对 SynFlowNet（合成感知分子 GFN）做控制严谨的可解释性研究：结合梯度显著性、反事实编辑、因子分析与稀疏自编码器，并用打乱标签、未训练同构网络等对照。
- `N085` [Designing the Haystack: Programmable Chemical Space for Generative Molecular Discovery (SpaceGFN)](https://arxiv.org/abs/2603.00614) · *预印本 2026*  
  把"化学空间本身"提升为可编程对象：用户用构件与反应规则显式构造化学/合成自洽的分子宇宙，GFN 在其中做性质偏置采样。
- `N086` [Curriculum-Augmented GFlowNets for mRNA Sequence Generation (CAGFN)](https://arxiv.org/abs/2510.03811) · *预印本 2025*  
  针对 mRNA 设计的稀疏长时程奖励与多目标权衡，把课程学习与多目标 GFN 结合：按序列长度渐进的课程从易到难引导探索，并提供新的 mRNA 设计环境。
- `N087` [FlowPipe: LLM-Enhanced Conditional GFlowNets for Data Preparation Pipeline Construction](https://arxiv.org/abs/2606.24679) · *SIGMOD 2027（作者标注已接收）*  
  把数据准备流水线合成表述为 DAG 上的条件概率流生成，用**条件 GFlowNet + TB** 把终端验证奖励连回早期决策；再用 FiLM 深度语义调制注入 LLM 逻辑先验，并把失败感知写入流目标以避开无效状态。
- `N088` [Generative Learning for Quantum Measurement Design (FlowMeas)](https://arxiv.org/abs/2608.11396) · *预印本 2026-08*  
  把有限 shot 预算下的量子测量协议设计重述为生成学习问题，用 GFN 直接采样浅层 Clifford 测量电路集合并满足硬件约束。
- `N089` [GFlowNets for Model Adaptation in Digital Twins of Natural Systems](https://arxiv.org/abs/2604.20707) · *预印本 2026（Under Review）*  
  把自然系统数字孪生的模型适应视为模拟推断问题：稀疏间接观测常不能唯一标定参数，GFN 在完整模拟器配置上按"模拟-观测一致度"奖励比例采样多个可信参数化。
- `N090` [Exploration through Generation: Applying GFlowNets to Structured Search](https://arxiv.org/abs/2510.21886) · *预印本 2025*  
  教学式地把 GFN 用 TB loss 应用到三个经典图优化问题——旅行商(TSP)、最小生成树(MST)、最短路——顺序选边/选点/选城市构造解。
- `N091` [Structurally Valid Log Generation using FSM-GFlowNets](https://arxiv.org/abs/2510.26197) · *预印本 2025*  
  把有限状态机(FSM)与 GFN 结合生成结构合法且行为多样的合成事件日志：FSM 由专家轨迹推导、编码领域规则，GFN 用 flow matching + FSM 合规/统计保真的混合奖励、经动态动作掩码与引导采样保证句法有效。
- `N092` [Transform-Invariant Generative Ray Path Sampling for Efficient Radio Propagation Modeling](https://arxiv.org/abs/2603.01655) · *预印本 2026*  
  用 GFN 替代射线追踪的穷举路径搜索做智能采样，缓解高阶交互下有效路径稀少导致的稀疏奖励：经验回放缓冲留存稀有有效路径、均匀探索策略防过拟合、基于物理的动作掩码先滤除不可能路径。
- `N093` [CounterFlowNet: From Minimal Changes to Meaningful Counterfactual Explanations](https://arxiv.org/abs/2602.17244) · *预印本 2026*  
  把反事实解释(CF)生成建模为条件 GFlowNet 上的顺序特征修改，按用户自定义奖励（有效性、稀疏性、邻近性、可信度）成比例采样多样 CF。
- `N094` [AlphaG-OPD: Reliability-Gated Sibling Counterfactuals for On-Policy Distillation in Symbolic Alpha Factor Discovery](https://arxiv.org/abs/2608.01303) · *预印本 2026-08*  
  在符号 alpha 因子挖掘中，GFN 保持对完整表达式的多样奖励比例分布，但其轨迹级目标不比较中间状态未选的兄弟动作。
- `N095` [TILDE: TILt-based Distributional Erasure for Concept Unlearning](https://arxiv.org/abs/2607.06432) · *预印本 2026-07*  
  文生图 diffusion 的概念遗忘：把遗忘表述为"在遗忘约束下与预训练模型偏差最小的条件分布"这一分布对齐目标，用**残差 ∇-GFlowNet** 学习由遗忘能量相对预训练模型诱导的 score 修正。
- `N096` [Meta-Learning-Driven GFlowNets for 3D Directional Modulation in Mobile Wireless Systems (Meta-GFlowNet)](https://arxiv.org/abs/2511.06188) · *预印本 2025（投 IEEE ICC 2026）*  
  在时间调制智能反射面(TM-IRS)物理层安全设计中，用 MAML 式元学习让 GFN 快速适应移动用户方向变化：内层轨迹平衡更新 + 外层元更新学到跨方向的通用先验，无需标注数据（用 GFN 学到的奖励与真实和速率的伪监督一致性目标）。
- `N097` [Improving LLM-Based Recommenders with Conservative Generative Flow Networks](https://icml.cc/virtual/2026/poster/65235) · *ICML 2026 主会*  
  研究离线 LLM 推荐：学习被限制在固定日志数据集上，数据集诱导的 token 前缀 DAG 只有部分转移支持，此时朴素 SubTB 不可辨识、会把概率质量任意分配到无支持区域；论文形式化三类失败来源——流高估、前向质量泄漏、后向补偿——并提出 CFlower：显式惩罚数据支持外前向流质量的保守 SubTB 目标，配合限制在数据 DAG 上的 on-policy 采样，在三个 Amazon 数据集上改善分布匹配与准确率–曝光权衡。
- `N098` [WINFlowNets: Warm-up Integrated Training of GFlowNets for Robotics and Machine Fault Adaptation](https://arxiv.org/abs/2603.17301) · *预印本 2026*  
  针对连续场景 CFlowNets 依赖预训练 retrieval 网络、难以适应动态机器人环境的问题，提出 flow 网络与 retrieval 网络"预热+共享 replay 协同训练"框架。
- `N099` [torchgfn: A PyTorch GFlowNet Library](https://arxiv.org/abs/2305.14594) · *预印本* [📝 深度解读](notes/N099_torchgfn_pytorch_library.md) · [🇨🇳 中文PDF](pdfs/zh/N099_zh.pdf) · [📄 英文PDF](pdfs/en/N099_torchgfn_a_pytorch_gflownet_library.pdf)  
  torchgfn 库的配套论文：核心贡献是把环境、神经网络模块与训练目标解耦为可互换组件的模块化架构，提供简洁 API 与复现、统一多个已发表结果的示例，是在标准基准实现上测试新 loss/新策略的参考协议。
- `N100` [IFlowNets: Extending Generative Samplers to Learn Strategies in Incomplete Information Games](https://arxiv.org/abs/2608.05422) · *NeurIPS 2025 Workshop (Dynamics at the Frontiers of Optimization, Sampling, and Games)*  
  把 Adversarial Flow Networks (AFlowNets) 推广到**不完全信息博弈**：证明完全信息博弈下已确立的 flow 约束在此设定下无法给出合法密度（即合法策略）与合法训练目标，提出 Information Flow Networks 修复该问题并严格推广 AFlowNets；三个标准博弈环境上与 Outcome Sampling CFR 相当或更好。
- `N101` [Scalable and Cost-Efficient de Novo Template-Based Molecular Generation](https://openreview.net/forum?id=zssWxiiJZ1) · *NeurIPS 2025 主会 · arXiv:2506.19865*  
  直面 template-based GFlowNet 的三个核心难题：最小化合成成本、扩展到大规模 building block 库、有效利用小片段集；提出 Recursive Cost Guidance 后向策略等机制。
- `N102` [Adversarial Generative Flow Network for Solving Vehicle Routing Problems](https://openreview.net/forum?id=tBom4xOW1H) · *ICLR 2025 主会 · arXiv:2503.01931*  
  针对 VRP 构造式神经求解器普遍用 Transformer、扩展性受限且解多样性不足的问题，提出对抗式 GFlowNet（AGFN）架构。
- `N103` [Discrete Compositional Generation via General Soft Operators and Robust Reinforcement Learning](https://openreview.net/forum?id=MGWk2tEgLW) · *ICLR 2026 主会 · arXiv:2506.17007* [📝 深度解读](notes/N103_general_soft_operators_robust_rl.md) · [🇨🇳 中文PDF](pdfs/zh/N103_zh.pdf) · [📄 英文PDF](pdfs/en/N103_discrete_compositional_generation_via_general_soft_operators.pdf)  
  指出各类熵正则方法（含 GFlowNet）在代理奖励下的过度保守问题，提出用**通用 soft operator** 统一并推广离散组合生成，并以鲁棒 RL 视角给出理论刻画。
- `N104` [On Scalable and Efficient Training of Diffusion Samplers](https://openreview.net/forum?id=Xzabk07lao) · *NeurIPS 2025 主会 · arXiv:2505.19552*  
  针对能量评估昂贵、采样空间高维时 diffusion sampler 难以扩展的问题，提出可扩展且样本高效的训练框架。
- `N105` [Reinforced Sequential Monte Carlo for Amortised Sampling](https://openreview.net/forum?id=DWaToCuNwa) · *ICML 2026 主会（Spotlight） · arXiv:2510.11711*  
  建立 SMC 与最大熵 RL 训练的神经序列采样器之间的联系——学到的策略与值函数正好给出 SMC 的 proposal kernel 与 twist function，从而把摊销方法与粒子方法结合。
- `N106` [Improved Off-policy Reinforcement Learning in Biological Sequence Design](https://openreview.net/forum?id=0TY5lhhdZm) · *ICML 2025 主会 · arXiv:2410.04461*  
  提出 \(\delta\)-Conservative Search：把 off-policy 搜索限制在可信区域内，缓解代理模型在分布外输入上的 misspecification，覆盖 DNA/RNA/蛋白/肽。

## 趋势洞察 Trends & Insights

- [trends applications](insights/trends_applications.md)
- [trends methods](insights/trends_methods.md)
- [trends neighbors](insights/trends_neighbors.md)

## 课程与教程 Courses & Tutorials

见 [资源目录 §6 课程、教程与博客](surveys/GFLOWNET_RESEARCH_RESOURCE_CATALOG_CN.md#6-课程教程与博客)，含 Mila IFT6167、Edward Hu 教程、torchgfn tutorial 等。

## 代码库 Codebases

| 库 | 说明 |
|---|---|
| [GFNOrg/torchgfn](https://github.com/GFNOrg/torchgfn) | PyTorch GFlowNet 库，官方维护，含 tutorials |
| [recursionpharma/gflownet](https://github.com/recursionpharma/gflownet) | Recursion 的分子生成 GFlowNet 实现 |
| [GFNOrg/gfn-lm-tuning](https://github.com/GFNOrg/gfn-lm-tuning) | GFlowNet 微调 LLM 参考实现 |
| [alexhernandezgarcia/gflownet](https://github.com/alexhernandezgarcia/gflownet) | 面向科学发现的 GFlowNet 框架（Crystal-GFN 等） |

## 报告 Reports

- [HTML 汇报（PPT 风格）](slides/index.html)
- [Beamer PDF 报告](slides/awesome_gflownets_report.pdf)

## 贡献 Contributing

欢迎 PR：新论文按 `编号 | 标题链接 | venue | 一句话简介` 追加到对应分区；深度解读放入 `notes/`，命名 `<ID>_<slug>.md`。

## License

[CC0](LICENSE)
