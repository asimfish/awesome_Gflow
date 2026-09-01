# GFlowNet × Optimal Transport：方向潜力深度分析

> 本文是对 [Your GFlowNet Secretly Learns an Optimal Transport Plan（arXiv:2606.06272）](https://arxiv.org/abs/2606.06272) 的专题深读，
> 回答"GFlowNet 与 OT 结合的研究方向是否很有潜力"。
> 分析基于论文全文（本地文本 `research/text/2606.06272.txt`）与联网核验，完成于 2026-07-29。
>
> 相关材料：
> - [资料清单 · OT 专题论文卡片（O01–O08）与精读问题](GFLOWNET_RESEARCH_RESOURCE_CATALOG_CN.md)（第 2 节、第 4 节为本文的精简版）
> - [理论指南 · 5.5 节：该论文定理的准确解读](GFLOWNET_THEORY_GUIDE_CN.md)
> - [论文 PDF](literature/core/2606.06272.pdf)

结论：这个方向有明显潜力，但潜力不在“用 GFlowNet 替代所有 Sinkhorn/OT 求解器”，而在于：

> 在巨大、隐式、组合化的状态图上，学习一个既满足源—目标边缘分布，又能通过合法局部动作执行传输的随机策略。

当前论文更像一座很有价值的理论桥梁，距离成熟通用算法还有不小距离。

## 1. 这篇论文究竟证明了什么

普通 GFlowNet 只固定终止分布：

\[
P_T(x)=\frac{R(x)}{Z},\qquad Z=\sum_{x'}R(x'),
\]

即 \(P_T(x)\propto R(x)\)——GFlowNet 的核心卖点正是只需未归一化奖励 \(R\)、无需知道 \(Z\)。本文下面的 OT 设定额外要求 \(\sum_x R(x)=1\)（此时 \(Z=1\)），才可以把它简写成 \(P_T(x)=R(x)\)。

论文额外固定第一步的源分布：

\[
F(s_0\to u)=L(u),\qquad
F(x\to s_f)=R(x),
\]

其中

\[
\sum_uL(u)=\sum_xR(x)=1.
\]

然后在满足流守恒的所有非无环 GFlowNet 中，最小化内部总边流：

\[
\min_{F\ge 0}\sum_{e\in E^\circ}F(e),
\]

\[
\operatorname{div}F(s)=L(s)-R(s).
\]

因为非无环 GFlowNet 的边流可解释为期望访问次数，所以这个目标**等价于**（相差一个与优化无关的常数）最小化期望轨迹长度：

\[
\sum_{e\in E^\circ}F(e)=\mathbb E[|\tau|]-1,
\]

其中减去的 1 对应固定的首步 \(s_0\to u\)（它不属于内部边集 \(E^\circ\)）。原论文明确说明省略该常数不改变问题的最小点，故作为**优化等价**成立；但写成恒等式 \(\sum_{e\in E^\circ}F(e)=\mathbb E[|\tau|]\) 并不严谨。

令 \(d_G(u,x)\) 为图上从 \(u\) 到 \(x\) 的最短路长度，则论文的定理 3.2 证明：

\[
\min_F\sum_eF(e)
=
\min_{\Pi\in\Gamma(L,R)}
\sum_{u,x}d_G(u,x)\Pi(u,x).
\]

右边正是以图最短路为 cost 的 Kantorovich OT。[论文原文](https://arxiv.org/abs/2606.06272)

证明由两个方向组成：

- 给定 coupling \(\Pi(u,x)\)，将每份质量沿 \(u\to x\) 的最短路径运输，得到一个可行 edge flow。
- 给定 GFlowNet flow，按其策略采样轨迹，由起点和终点构造 coupling；任何实际路径都不会短于最短路。

因此最优 GFlowNet 不仅给出“多少质量从 \(u\) 到 \(x\)”，还给出“具体通过哪些合法局部动作过去”。

## 2. 真正新颖的地方

“图上的 shortest-path OT 等价于 minimum-cost network flow”本身是经典结果；图 OT 文献早已使用这种表述，例如 [Quadratically Regularized Optimal Transport on Graphs](https://epubs.siam.org/doi/10.1137/17M1132665)。

这篇论文真正的新意是把它翻译到了 GFlowNet 语言中：

\[
\text{OT coupling}
\quad+\quad
\text{local routing policy}
\quad=\quad
\text{minimum-flow GFlowNet}.
\]

因此它输出的不是一个静态 coupling matrix，而是一个可以实际执行的 \(P_F(s'|s)\)。这对隐式组合空间很重要，因为你可能根本无法枚举所有 \((u,x)\) 组合或显式存储 coupling。

它依赖的理论链条也很清楚：

1. 非无环 GFlowNet 理论把 flow 定义为 expected visit counts——该定义与吸收条件由 [A Theory of Non-Acyclic GFlowNets（AAAI 2024）](https://ojs.aaai.org/index.php/AAAI/article/view/28989)首先给出，[Revisiting Non-Acyclic GFlowNets in Discrete Environments（ICML 2025）](https://proceedings.mlr.press/v267/morozov25a.html)在有限离散情形做了更简洁的存在性/唯一性刻画（本文用到的是后者的可计算形式）。
2. [Learning Shortest Paths with Generative Flow Networks](https://arxiv.org/abs/2603.01786)证明最小总流会选择最短路径。
3. 当前论文加入固定源分布 \(L\)，从单源最短路提升为多源—多目标 OT。

## 3. 为什么我认为它有潜力

### 3.1 为“内部 flow 应该选哪一个”提供几何原则

标准 reward matching 通常只决定终止分布，内部 flow 严重欠定。[Shen et al.](https://proceedings.mlr.press/v202/shen23a.html)已经说明，内部 flow 会影响信用分配和未见状态泛化。

OT/minimum-flow 给出一个明确选择原则：

\[
\text{在所有正确终止分布中，选择运输代价最小的 flow。}
\]

它可能带来：

- 更短 rollout；
- 更少循环；
- 更直接的终端信用；
- 更低推断成本；
- 更符合状态空间几何的路径。

但“最短 flow”是否也最有利于泛化尚未证明。过度集中到少数最短路径，反而可能损害探索或共享子结构学习。

### 3.2 隐式组合图上的 OT

传统 OT 通常需要 cost matrix \(C_{ux}\)。如果 \(|U||X|\) 巨大，这个矩阵无法构造。

GFlowNet 可以只访问局部邻居并学习：

\[
P_F(s'|s),\quad P_B(s|s').
\]

潜在应用包括：

- 分子或化学反应的合法编辑路径；
- permutation、调度和路由问题；
- 程序、证明或表达式变换；
- 因果图编辑；
- 机器人配置空间；
- 离散科学状态之间的转化路径。

前提是局部动作和边代价确实具有领域意义。

### 3.3 摊销求解一族 OT 问题

当前论文基本是“每对 \(L,R\) 训练一个模型”。更有价值的下一步是训练：

\[
P_F(a|s,L,R,c),
\]

让同一个模型对不同源分布、目标分布和 cost 泛化。

这会把 GFlowNet 从单次 OT solver 变成 amortized transport operator。这个方向已有强劲的神经 OT 竞争者，例如 ICML 2025 的 [Universal Neural Optimal Transport](https://proceedings.mlr.press/v267/geuter25a.html)，因此 GFlowNet 必须在“隐式图、局部合法路径和组合泛化”上体现独特优势。

### 3.4 与 Schrödinger bridge 的联系非常自然

当前论文解决的是无正则的最短路径 OT。更自然的路径空间扩展是：

\[
\min_{P(\tau)}
\mathbb E_P[c(\tau)]
+
\varepsilon\operatorname{KL}(P(\tau)\|P_0(\tau)),
\]

同时固定起点和终点边缘。

这就是离散路径空间上的 entropic OT / Schrödinger bridge 风格问题。GFlowNet 天生拥有前向、反向轨迹分布，因此这个结合可能比静态 coupling 更自然：

- \(P_0\)：参考动力学或先验策略；
- \(P_F\)：学习到的传输策略；
- \(P_B\)：时间反演/后向条件；
- TB：路径概率比率约束。

但标准 TB 并不会自动等价于 Schrödinger bridge，需要重新推导参考过程、熵项和边缘约束。

## 4. 当前论文的主要不足

### 4.1 定理条件很强

必须具有：

- 有限有向图；
- 固定且已归一化的 \(L,R\)；
- 任意相关 \(u\to x\) 可达；
- 非负、单位边长的最短路 cost；
- 吸收策略和有限期望轨迹长度；
- 精确流守恒；
- minimum-flow 全局最优。

由于图是有向的，\(d_G(u,x)\) 甚至未必对称；它是合法 OT cost，但不一定构成通常意义下的 Wasserstein metric。

### 4.2 没有使用 GFlowNet 最吸引人的 unknown-\(Z\) 能力

标准 GFlowNet 可以只知道未归一化 \(R(x)\)。但 OT 定理要求

\[
\sum_uL(u)=\sum_xR(x)=1.
\]

也就是说，精确定理中两个边缘都是已知概率分布。如何扩展到未知归一化常数、质量不相等的 unbalanced OT，是一个非常关键的开放问题。

### 4.3 精确定理和神经训练之间仍有距离

定理针对约束 LP；实际训练使用 regularized TB 和系数 \(\lambda\)。

论文自己的 permutation 实验已经显示：

- 较大 \(\lambda\)：路径更短，但终止边缘偏差更大；
- 较小 \(\lambda\)：采样更准确，但路径更长。

例如某些设置中路径长度甚至“优于”精确 OT cost，但这是因为模型没有精确满足目标边缘，并不是真的击败了 OT 最优值。

更理想的后续方法应该使用 primal-dual、augmented Lagrangian 或显式约束训练，而不是仅靠 penalty 权衡。

### 4.4 实验证据仍很早期

论文实验主要是：

- \(H=10,15,20\) 的 HyperGrid；
- permutation \(n=4,8,20\)；
- 三个随机种子；
- 两层、128 hidden units 的 MLP；
- 小规模与 POT/LP 精确解比较；
- \(n=20\) 时已经没有精确 OT ground truth。

目前没有与大型 graph min-cost-flow、Sinkhorn、专用稀疏 OT 或 neural OT 方法进行完整的时间、内存和误差比较。作者页面也将其列为 ICML 2026 workshop 工作，代码仍是 TBA，说明成熟度确实较早。[作者项目状态](https://greatdrake.github.io/)

## 5. 我最看好的后续课题

下表的"潜力"评的是**方向本身的价值**；2026-08 的撞车复核补入了"占位风险"一列——两者要一起看，因为若干高潜力方向已被非 GFlowNet 方法占据生态位。

| 方向 | 潜力 | 占位风险（2026-08 复核） | 核心问题 |
|---|---:|---|---|
| Conditional / amortized graph OT | 很高 | **高**：[ULOT（NeurIPS 2025）](https://papers.nips.cc/paper_files/paper/2025/file/873fd89b3e4db1f6242c2333673e104d-Paper-Conference.pdf) 已做图上/条件化/含 unbalanced 的摊销 OT plan 预测，比经典 solver 快约两个数量级且可 warm-start；另有 CONDOT、CVFM | 能否跨未见 \(L,R\) 泛化，并在多次求解时超过传统 solver |
| Entropic path OT / Schrödinger bridge GFN | 很高 | **高**：离散/图 SB-EOT 生态已拥挤——[DDSBM](https://arxiv.org/abs/2410.01500)、[GSBoG](https://arxiv.org/abs/2602.04675)、[MadSBM](https://arxiv.org/abs/2601.22408)、[离散 SB/EOT benchmark](https://arxiv.org/abs/2509.23348)。均非 GFN（多用 CTMC + iterative Markovian fitting），但占同一生态位 | 如何加入参考动力学和路径 KL，同时保留稳定 balance 训练 |
| Unbalanced OT 与未知 \(Z\) | 很高 | 中：unbalanced 侧有 ULOT，但"联合估计 \(Z\)"这一 GFlowNet 独有能力尚无人做 | 如何允许质量产生/消失，并联合估计归一化常数 |
| Primal-dual GFlowNet OT | 很高 | **低** | 用 Kantorovich potential 作 critic，直接监控 primal-dual gap |
| Balance 残差 → OT 误差的界 | 很高 | **低（最干净）** | 局部 flow 误差能否控制 coupling 与 cost 误差；把 T51 的 TB→TV 界思路迁到 OT |
| 加权或学习型边 cost | 高 | 低 | 非负边权扩展容易；联合学习 cost 时如何避免退化为零 |
| 多边缘、barycenter、multi-marginal OT | 中高 | 中 | 如何表示多个时间截面或多个目标分布 |
| 连续高维 OT | 中等 | 中高（成熟 neural OT 已占） | 需要同时解决 continuous、non-acyclic 和稳定性 |
| 替代显式有限图上的普通 OT solver | 较低 | — | 经典凸优化通常更快、更准、还有证书 |

综合潜力与占位风险，**最值得做的是 primal-dual 版本与"Balance 残差 → OT 误差界"这两条**（都属低撞车）。资料清单 §4 的"四个可执行课题"是从**可执行性**口径挑的（要求有最小可行实验与明确报告指标），与本表的**方向潜力**口径不同；两者的对应关系见清单 §4 的说明。

primal-dual 之所以值得做：论文的 OT dual 对应图上的 Kantorovich potential；互补松弛意味着最优流只走“tight edges”。这很像：

\[
\text{policy/flow = actor},\qquad
\text{Kantorovich potential = critic}.
\]

它既可能改善训练，也能提供 transport cost 和边缘误差证书。

## 6. 最能验证这个方向的实验

我建议不要只继续做更大的单次 HyperGrid，而是：

1. 随机生成许多不同的 \(L,R\) 对。
2. 训练一个条件 GFlowNet。
3. 在未见过的 \(L,R\) 上测试。
4. 对比 network simplex、POT/Sinkhorn 和 neural OT。
5. 同时报告：
   - 源边缘误差；
   - 目标边缘 TV；
   - transport cost gap；
   - primal-dual gap；
   - 平均及 P95 路径长度；
   - 循环/不吸收率；
   - wall-clock、显存和 amortization break-even point。

然后再把环境换成 permutation、分子编辑或其他无法显式构造 cost matrix 的组合图。这才真正检验 GFlowNet 的独特价值。

最终判断：

- **作为理论研究方向：很有潜力。**
- **作为隐式组合空间上的可执行 transport policy：尤其有潜力。**
- **作为成熟通用 OT solver：目前证据不足。**
- **这篇论文是一个很好的起点，但更像“发现了正确接口”，还没有完成算法和应用闭环。**
