# GFlowNet 的邻域竞品与相对位置（2025-2026）

> 调研日期：2026-09-01 · 方法：web 检索（arXiv / OpenReview / NeurIPS proceedings / bioRxiv）+ 本仓库 206 篇目录交叉核对 · 作者：awesome_Gflow 维护流水线

核心问题：在「从非归一化奖励或能量中采样多样高质量对象」这条赛道上，GFlowNet 的替代方案发展到什么程度？GFlowNet 的相对位置是上升还是下降？

先给结论，再给证据：**GFlowNet 的位置发生了性质变化，不是简单的升或降。**它作为「独立方法品牌」的地位在下降——熵正则 RL 等价性（T14）与连续时间渐近等价（N002）证明它是更大方法族的一个特例，纯采样器竞品在部分物理基准上已经明显超过它。但它作为「off-policy 训练目标的默认选项」的地位在上升——扩散采样器方向已经把 TB 系目标当作标准配置，而在「奖励设计不完美时仍要保多样性」的场景中，它有竞品拿不出的鲁棒性证据。

## 1. 神经采样器家族：物理基准上 GFlowNet 不再是 SOTA

高维 Boltzmann 采样这条线在 2024-2026 明显走向「用力场/梯度信息 + 免仿真训练」，而不是走 GFlowNet 的轨迹级平衡：

- [iDEM (Iterated Denoising Energy Matching)](https://arxiv.org/abs/2402.06121) 用免仿真的能量匹配训练，在 DW-4、LJ-13、LJ-55 上比 PIS/DDS/FAB 更能 scale。该文明确把自己定位为「某种结构的 GFlowNet 的免仿真训练算法」，也就是绕开了 GFN 的 SDE 积分训练环节。
- [Drifting to Boltzmann: Force-Guided Drifting](https://arxiv.org/abs/2603.05527)（2026-03）在 MD17 Ethanol 上单步生成即达到 h(r) TVD = 0.139、W2 = 0.031、键稳定性 97.5%，推理比迭代方法快约 2000 倍。它同时指出标准 h(r) 指标会被非键 H-H 对主导而产生误导，提出 per-type TVD 与 Bond MAE。
- [Flow Perturbation++](https://arxiv.org/abs/2601.21177)（2026-01）在 1000 维 GMM 上给出无偏多步 Jacobian 估计，模态权重估计 0.256±0.027（真值 0.25），成本与单步 FP 相当，而 Hutchinson 各变体或有严重偏差或模态坍缩。

我的判断：在「有力场、有梯度、要物理有效性」的场景，GFlowNet 系已经不占优。这些方法用的是能量/力信息的结构，而 GFN 的强项（离散组合空间、只有终端奖励）在这类任务里不构成优势。

反向证据同样存在：[On scalable and efficient training of diffusion samplers](https://proceedings.neurips.cc/paper_files/paper/2025/file/255a98afe2f5fe28c518eef9f7905da3-Paper-Conference.pdf)（NeurIPS 2025）明确写出选 GFlowNet 目标的理由——相比 PIS/DDS 的 KL 类 on-policy 目标，GFlowNet 目标可以用任意全支撑提议分布的 off-policy 轨迹优化，从而能用噪声 roll-out、回放缓冲、MCMC 局部搜索这些对多模态采样关键的探索手段。该文的对比表里 TB + LS、GAFN（本目录 N011）、AT + LP 都是 GFN 系基线。

## 2. 熵正则 RL 等价性之后：社区没有弃用 GFN 框架

T14（GFN = 熵正则 RL，AISTATS 2024）与 N003（最大熵 GFN 与 soft Q-learning 的精确对应，AISTATS 2024）确立等价性后，一个合理担忧是社区会直接改用成熟的 RL 工具箱。检索证据不支持这个走向：

- 2026 年新论文（RapTB、DTB/ACE、alpha-GFN、Stable GFlowNets）仍在 GFlowNet 的语言里做改进，而不是改写成 soft Q-learning 的形式。
- 等价性的实际用途是「搬运工具」而非「替换框架」：N002（TMLR）把离散 GFN 与连续时间 PDE/路径测度对接，用来设计更快的训练方案；[alpha-GFN](https://arxiv.org/abs/2602.01749) 用 Markov chain 可逆性给出统一视角后，仍然产出 GFN 形式的目标。

我的判断：等价性降低了 GFlowNet 的「理论新颖性溢价」，但它的 off-policy 训练接口与 $Z$（配分函数）显式估计这两点仍是 RL 工具箱里没有现成对应物的，所以框架没被替换。

## 3. Schrödinger 桥与流匹配：在离散空间是竞争者，也是被 GFN 吸收的对象

竞争的一面：[DDSBM (Discrete Diffusion Schrödinger Bridge Matching)](https://arxiv.org/abs/2410.01500)（ICLR 2025，本目录 N043 的姊妹工作）把 Iterative Markovian Fitting 扩展到离散空间，用连续时间 Markov 链解 SB 问题并证明收敛，直接做分子优化（图变换），且证明其动力学设计等价于以图编辑距离为代价的熵正则 OT。这与 GFlowNet 的分子优化场景正面重叠，且它有「最小图变换」这个 GFN 不天然具备的性质。

被吸收的一面：[Your GFlowNet Secretly Learns an Optimal Transport Plan](https://arxiv.org/abs/2606.06272)（本目录 O08）证明最小流非无环 GFlowNet 学习问题可等价写成线性规划，固定初始边流分布后就变成以图上距离为代价的 Kantorovich OT 问题（也等价于图上 Beckmann 问题的离散形式），最优前向策略采样的正是最优路径、诱导最优耦合。也就是说 GFlowNet 反过来成了求解图上 OT 的一种可扩展神经方法。

我的判断：这两条线在 2026 年不是替代关系而是收敛关系——SB/OT 提供目标与代价结构，GFlowNet 提供离散组合空间上的可扩展参数化与 off-policy 训练。GFN×OT 是本仓库标注的高潜力方向（见 surveys/GFLOWNET_OT_POTENTIAL_ANALYSIS_CN.md），O08 与 N041 是当前最实的两个抓手。

## 4. 自回归 + RL（GRPO 系）：最强的定量对比证据在 GFlowNet 这边

这是本次调研中证据最硬的一节，两组独立实验给出一致结论：

**分子（scaffold-conditioned SMILES 优化，来自 [RapTB](https://arxiv.org/abs/2603.00454) Table 1）：**

| 方法 | 有效性 Acc | 奖励 Score | 熵 Entropy | 指纹多样性 FPDiv | 平均长度 |
|---|---|---|---|---|---|
| PPO | 1.000 | 0.604 | ≈0 | — | — |
| GRPO | 0.997 | 0.661 | 0.98 | — | 10.0 |
| TB | 0.998 | 0.717 | 2.503 | 0.807 | 3.065 |
| SubTB | 0.328 | 0.755 | 2.127 | 0.836 | 8.354 |
| RapTB | 0.996 | 0.740 | 2.448 | 0.860 | 6.142 |
| RapTB + SubM | 0.988 | 0.844 | 2.726 | 0.898 | 7.435 |

PPO 熵接近 0（坍缩到单一模态）、GRPO 熵不超过 0.98，而 TB 系熵在 2.1-2.7 之间，且 RapTB+SubM 在奖励分数上（0.844）也超过 GRPO（0.661）。读法：奖励最大化与奖励成比例采样的差别不只是多样性，在这个任务上连奖励质量都被拉开。

**肽（治疗性肽生成，[bioRxiv 2026-01](https://www.biorxiv.org/content/10.64898/2026.01.05.697258v3)）：** 与带显式多样性惩罚的 GRPO-D（λ=0.15）对比，GFlowNet 的二肽采样均匀度高 5.4 倍、奖励方差低 1.9 倍、重复序列少 3.9 倍。压力测试更关键：把奖励里的熵门控去掉后，GRPO-D 完全坍缩（1000 个样本 100% 含同一三肽模式 RMMRMMRMM，三种二肽占全部二肽的 92.4%），而 GFlowNet 多样性保持 0.937。

我的判断：这组证据定义了 GFlowNet 在 2026 年真正的护城河——**不是峰值奖励，而是对奖励函数设计缺陷的鲁棒性**。GRPO 系要靠精心设计的多样性惩罚和熵门控才不坍缩，而这些超参在真实药物发现流水线里通常调不准。[Advances in GRPO for Generation Models: A Survey](https://arxiv.org/pdf/2603.06623)（2026-03）里 diversity preservation 被列为 GRPO 的一个专门研究方向（DiverseGRPO 用 Vendi Score 提升 13%-18%、OSCAR 做训练无关的隐空间多样性增强），从反面印证这是 GRPO 的结构性弱点而非实现问题。

## 5. 社区热度与团队扩散

- 本目录 206 篇按 venue 分布：NeurIPS 2024 收 14 篇、ICLR 2025 收 17 篇、ICML 2025 收 13 篇、NeurIPS 2025 收 13 篇、ICML 2026 收 17 篇、ICLR 2026 收 7 篇。主会产出稳定在每届 7-17 篇，没有爆发也没有断崖。
- 团队已明显扩散出 Mila/Bengio 系：DDSBM 来自韩国团队（KAIST 系，代码 github.com/junhkim1226/DDSBM）、alpha-GFN 与 Stable GFlowNets 的作者群与 Mila 无直接关系、非无环 GFN 这条线由 Brunswic 等人（华为诺亚方舟系）与 Morozov 等人持续推进、O08 出自 Morozov 团队。
- 精确的 arXiv 年度计数未检索到（检索接口未返回可核实的计数），这里不给数字。

## 6. 相对位置的最终判断

1. **纯采样性能**：下降。物理场景的 SOTA 已由 iDEM、力引导漂移、FP++ 这些用能量/力/Jacobian 结构的方法占据。
2. **训练目标的采用度**：上升。扩散采样器方向把 TB/VarGrad + 回放 + 局部搜索当作标准配置，torchgfn 也把 off-policy 训练写进官方指南，工程上已固化。
3. **理论独立性**：下降。熵正则 RL、Markov chain、OT、连续时间路径测度四个视角都能把 GFlowNet 写成特例，纯框架级新颖性不再是卖点。
4. **应用护城河**：上升且清晰。「奖励设计不完美时仍保多样性」这条在分子与肽两个独立实验里都有量化证据，是 GRPO/PPO 系拿不出的。
5. **对新入场者的建议**：不要做「GFlowNet vs X 谁更强」的对比论文（结论已明确且依场景而定），要做的是把 GFN 的鲁棒性优势落到具体流水线（药物发现、LLM 后训练的覆盖率问题），或者做 GFN×OT 这类结构性连接（O08、N041 是抓手）。

## 附：本报告引用的论文与其在本目录中的编号

逐条核对 206 篇目录后的结果：3 篇已收录（给出编号），5 篇确为目录外（已按 CONTRIBUTING 流程补录，见新编号）。

| 论文 | 目录编号 | 来源 | 日期 |
|---|---|---|---|
| Iterated Denoising Energy Matching (iDEM) | **N031** | [arXiv 2402.06121](https://arxiv.org/abs/2402.06121) | ICML 2024 poster |
| On scalable and efficient training of diffusion samplers | **N104** | [NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/255a98afe2f5fe28c518eef9f7905da3-Paper-Conference.pdf) | NeurIPS 2025 主会 |
| Rooted Absorbed Prefix Trajectory Balance (RapTB) | **T54** | [arXiv 2603.00454](https://arxiv.org/abs/2603.00454) | ICML 2026 |
| Discrete Diffusion Schrödinger Bridge Matching (DDSBM) | 已收录（见目录 SB 分区） | [arXiv 2410.01500](https://arxiv.org/abs/2410.01500) | ICLR 2025 |
| Drifting to Boltzmann: Force-Guided Drifting | **N107**（本次补录） | [arXiv 2603.05527](https://arxiv.org/abs/2603.05527) | 预印本 2026-03 |
| Flow Perturbation++ | **N108**（本次补录） | [arXiv 2601.21177](https://arxiv.org/abs/2601.21177) | 预印本 2026-01 |
| Generating Structurally Diverse Therapeutic Peptides with GFlowNet | **N109**（本次补录） | [bioRxiv 2026.01.05.697258](https://www.biorxiv.org/content/10.64898/2026.01.05.697258v3) | bioRxiv 2026-01 |
| Advances in GRPO for Generation Models: A Survey | **N110**（本次补录） | [arXiv 2603.06623](https://arxiv.org/abs/2603.06623) | 预印本 2026-03 |
| AbFlowNet | **N111**（本次补录） | [arXiv 2505.12358](https://arxiv.org/abs/2505.12358) | 预印本 2025-05 |
| gfnx（JAX 库与基准） | **N112**（本次补录） | [arXiv 2511.16592](https://arxiv.org/abs/2511.16592) | 预印本 2025-11 |
