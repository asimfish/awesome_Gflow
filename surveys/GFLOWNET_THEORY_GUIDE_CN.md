# GFlowNet 理论调研与学习指南

> 调研截止：2026-07-31  
> 重点：理论主线、训练目标、适用条件、经典论文、2024–2026 前沿、博客与实践资源  
> 适合读者：了解基本概率论、深度学习和强化学习，希望系统进入 GFlowNet 的学习者

## 先给结论

GFlowNet（Generative Flow Network）的目标不是找出奖励最大的一个对象，而是训练一个逐步构造对象的随机策略，使终止对象 \(x\) 的概率满足

\[
P_T(x)=\frac{R(x)}{Z},
\qquad
Z=\sum_{x\in\mathcal X}R(x),
\]

其中 \(R(x)\ge 0\) 是只需知道到归一化常数的奖励或未归一化密度。（严格地说，标准正确性定理通常要求 \(R(x)>0\)，或至少在支撑集上为正——原始论文的推导需要边流为正以保证 \(\log F\) 良定义、终止状态可达；\(R(x)=0\) 对应零流不可达对象，结论式在数值上仍成立。）

它最值得掌握的思想不是某个具体 loss，而是下面这条逻辑链：

1. 把同一个对象的所有生成路径看成流入该对象的“概率质量”。
2. 在每个中间状态维持流守恒。
3. 把终止状态的流固定为 \(R(x)\)。
4. 用归一化的出边流定义前向策略。
5. 因而终止对象的总概率正比于 \(R(x)\)，而不会因为“有多少条路径能生成它”而额外偏置。

我认为学习 GFlowNet 时最容易忽略、也最重要的四点是：

- **正确性定理是零损失、全支持、足够表达能力和全局优化下的结论**，不是有限训练自动成功的保证。
- **终止分布通常唯一，内部状态流和轨迹流通常不唯一**。反向策略 \(P_B\)、状态表示和训练目标会选择不同的信用分配方案，进而影响泛化和优化。
- **GFlowNet 是分布匹配，不是期望奖励最大化**。它与 RL、VI、MCMC 都有精确联系，但只有在相应条件下才等价。
- 2026 年的 OT 结果并不是“任意 GFlowNet 都在做最优传输”，而是说：**固定源分布的、非无环的、最小总流 GFlowNet**，在图最短路代价下等价于 Kantorovich OT。

## 1. 什么时候应该使用 GFlowNet

### 1.1 合适的问题

GFlowNet 特别适合同时满足以下多项的问题：

- 对象是离散或组合结构，如集合、序列、图、分子、因果 DAG、程序或证明。
- 对象可通过一串局部动作逐步构造。
- 只有未归一化的 \(R(x)\)，归一化常数 \(Z\) 很难计算。
- 目标是获得**多个不同的高价值候选**，而非只找一个最大值。
- 奖励地形多峰，且不同好解之间存在可学习的共享结构。
- 训练后需要反复、低成本地产生样本，值得承担一次性的摊销训练成本。

典型例子是候选分子设计：代理模型并不完全可靠，所以只返回单一“最优分子”很危险；更合理的是按照代理奖励产生一批多样候选，再交给昂贵实验验证。这个动机在[最早的 GFlowNet 论文](https://proceedings.neurips.cc/paper/2021/hash/e614f646836aaed9f89ce58e837e2310-Abstract.html)和[作者的早期博客](https://folinoid.com/w/gflownet/)中都很清楚。

### 1.2 不应默认使用的情形

- 只需要一个最优解，且成熟的搜索/优化器已足够。
- 能直接从目标分布精确采样。
- 只需很少样本，摊销训练成本得不偿失。
- 奖励近乎无结构，已访问模式不能帮助泛化到未访问模式。
- 奖励极昂贵，但又没有代理模型、离线数据或主动学习机制。
- 生成过程的状态合并设计不合理，导致 Markov 性或奖励分解假设失真。

“GFlowNet 擅长发现模式”应理解为一种**有条件的潜力**：神经网络必须能从已发现结构泛化。若好解完全随机散布，GFlowNet 不会凭空知道未访问模式在哪里。

## 2. 统一数学框架

下面采用 [GFlowNet Foundations](https://jmlr.org/papers/v24/22-0364.html) 的有限有向无环图（DAG）表述。

### 2.1 状态、轨迹和终止对象

- \(s_0\)：唯一源状态。
- \(s_f\)：形式上的唯一汇状态。
- \(\mathcal X=\operatorname{Par}(s_f)\)：终止对象对应的状态集合。
- 一条完整轨迹

\[
\tau=(s_0\to s_1\to\cdots\to x\to s_f),\qquad x\in\mathcal X.
\]

关键在于：图通常不是树。同一个 \(x\) 可以由许多不同轨迹生成。例如集合 \(\{a,b\}\) 可以先加 \(a\) 再加 \(b\)，也可以反过来。

### 2.2 轨迹流、状态流和边流

先给每条完整轨迹一个非负、未归一化质量 \(F(\tau)\)。由它诱导

\[
F(s)=\sum_{\tau:s\in\tau}F(\tau),
\qquad
F(s\to s')=\sum_{\tau:(s\to s')\in\tau}F(\tau).
\]

总流为

\[
Z=\sum_\tau F(\tau)=F(s_0)=F(s_f).
\]

这里的 flow 是图上的未归一化质量，不是 normalizing flow 中的可逆映射，也不是现代连续生成模型里的 ODE flow matching。

### 2.3 流守恒与奖励匹配

对每个内部状态 \(s\)，要求

\[
\sum_{u\in\operatorname{Par}(s)}F(u\to s)
=F(s)
=\sum_{v\in\operatorname{Child}(s)}F(s\to v).
\]

对终止状态，把流向形式汇点的边流固定为奖励：

\[
F(x\to s_f)=R(x).
\]

于是

\[
Z=F(s_f)
=\sum_{x\in\mathcal X}F(x\to s_f)
=\sum_{x\in\mathcal X}R(x).
\]

### 2.4 前向策略、反向策略和核心定理

从流定义两个局部策略：

\[
P_F(s'|s)=\frac{F(s\to s')}{F(s)},
\qquad
P_B(s|s')=\frac{F(s\to s')}{F(s')}.
\]

若流是 Markovian 的，则完整轨迹概率既可前向分解，也可反向分解：

\[
P(\tau)
=\prod_t P_F(s_t|s_{t-1})
=\prod_t P_B(s_{t-1}|s_t).
\]

严格说，第二个乘积包含从 \(s_f\) 回溯的边；实践中常把终止边吸收到边界条件里。

从 \(s_0\) 按 \(P_F\) 逐步采样，终止于 \(x\) 的概率为

\[
P_T(x)
=\frac{F(x\to s_f)}{Z}
=\frac{R(x)}{\sum_{x'}R(x')}.
\]

这就是 GFlowNet 的核心正确性结果。证明并不神秘：状态守恒保证源端流完整地路由到终点，终点流被规定为奖励，再除以总流即可。

### 2.5 一个最小例子：正确终点不等于唯一内部流

设只有一个终止对象 \(x\)，但存在两条路径：

\[
s_0\to a\to x,\qquad s_0\to b\to x,
\]

且 \(R(x)=1\)。对任何 \(q\in[0,1]\)，都可令

\[
F(s_0\to a)=F(a\to x)=q,
\]

\[
F(s_0\to b)=F(b\to x)=1-q.
\]

所有这些流都满足守恒，也都给出 \(P_T(x)=1\)，但它们对路径和中间状态分配的信用完全不同。若还要在所有已列状态上定义非退化的局部策略，可取 \(q\in(0,1)\)；端点 \(q=0,1\) 会产生不可达的零流状态。这一自由度在大规模神经网络训练中会影响：

- 哪些子结构得到更强监督；
- 未见对象如何泛化；
- 采样与梯度方差；
- 轨迹长度；
- 探索是否会自我强化到早期路径。

### 2.6 反向策略不只是“倒着采样”

在有限 DAG 上，给定终止边流 \(R(x)\) 以及一个合法的 \(P_B\)，会唯一确定相应的 Markovian flow，并进而唯一确定精确解处的 \(P_F\)。因此：

- 固定 \(P_B\) 相当于选择一种把终止奖励向祖先分摊的规则。
- 学习 \(P_B\) 增加了自由度，可能改善训练，也可能造成不良的信用分配。
- 即使所有精确解有相同 \(P_T\)，它们的内部流、轨迹熵和泛化偏置仍可不同。

这正是 [Towards Understanding and Improving GFlowNet Training](https://proceedings.mlr.press/v202/shen23a.html) 的关键出发点之一。

## 3. 主要训练目标：它们到底约束了什么

这些目标在所有约束精确满足时指向同一个 reward-matching 解，但它们给梯度的空间尺度不同，因而信用传播、方差和计算代价不同。

### 3.1 Flow Matching（FM）

FM 在每个状态比较总入流与总出流。采用显式终止边时，

\[
\mathcal L_{\mathrm{FM}}(s)
=\left[
\log
\frac{
\sum_{u\in\operatorname{Par}(s)}F_\theta(u\to s)
}{
\sum_{v\in\operatorname{Child}(s)}F_\theta(s\to v)
}
\right]^2,
\]

终止边满足 \(F_\theta(x\to s_f)\approx R(x)\)。

原论文也常写成

\[
\sum_{u}F_\theta(u\to s)
\approx
R(s)+\sum_vF_\theta(s\to v),
\]

其中只有终止状态的 \(R(s)\) 非零。

特点：

- 约束局部，类似 TD/自举。
- 可从中间状态训练。
- 长轨迹上的终端信用传播可能较慢。
- 需要枚举或求和父边和子边；父节点难枚举时不方便。

### 3.2 Detailed Balance（DB）

DB 对每条边约束

\[
F_\theta(s)P_{F,\theta}(s'|s)
=
F_\theta(s')P_{B,\theta}(s|s').
\]

常用 loss 为

\[
\mathcal L_{\mathrm{DB}}(s,s')
=
\left[
\log
\frac{
F_\theta(s)P_{F,\theta}(s'|s)
}{
F_\theta(s')P_{B,\theta}(s|s')
}
\right]^2,
\]

并使用边界 \(F_\theta(x)=R(x)\) 或等价的终止边约束。

特点：

- 也是局部约束，不必显式求所有父边流之和。
- 引入 \(P_B\) 后，边级信用分配更明确。
- 仍有逐步自举，深层终端信号传到早期状态可能慢。
- 名称借鉴 MCMC 的 detailed balance，但这里并不是要求前向生成链本身可逆。

### 3.3 Trajectory Balance（TB）

对完整轨迹

\[
\tau=(s_0\to s_1\to\cdots\to s_n=x),
\]

TB 约束

\[
Z_\theta\prod_{t=1}^{n}P_{F,\theta}(s_t|s_{t-1})
=
R(x)\prod_{t=1}^{n}P_{B,\theta}(s_{t-1}|s_t).
\]

对应

\[
\mathcal L_{\mathrm{TB}}(\tau)
=
\left[
\log
\frac{
Z_\theta\prod_tP_{F,\theta}(s_t|s_{t-1})
}{
R(x)\prod_tP_{B,\theta}(s_{t-1}|s_t)
}
\right]^2.
\]

如果所有完整轨迹的 TB 残差都为零，则求和所有终止于 \(x\) 的轨迹，利用 \(P_B(\tau|x)\) 的归一化可得

\[
Z_\theta P_T(x)=R(x).
\]

[Trajectory Balance 论文](https://proceedings.neurips.cc/paper_files/paper/2022/hash/27b51baca8377a0cf109f6ecc15a0f70-Abstract.html)证明了这一全局正确性，并报告相较局部目标更快的长程信用传播。代价是：

- 单条完整轨迹直接影响早期动作，信用更直接；
- 同时完整轨迹 log-ratio 的随机梯度可能有更高方差；
- 必须完成轨迹并获得终端奖励；
- \(Z_\theta\) 只有在拟合良好时才可解释为可靠的配分函数估计。

一个常见误读是“TB 有定理，所以训练稳定”。定理说的是所有轨迹零残差或全支持期望 loss 的全局最小值，不是有限样本 SGD 的动态。

### 3.4 为什么 DB 可以推出 TB

沿轨迹逐边相乘 DB 等式：

\[
\prod_{t=1}^n
\frac{
F(s_{t-1})P_F(s_t|s_{t-1})
}{
F(s_t)P_B(s_{t-1}|s_t)
}
=1.
\]

中间的 \(F(s_t)\) 全部消去，只留下

\[
\frac{
F(s_0)\prod_tP_F
}{
F(x)\prod_tP_B
}=1.
\]

代入 \(F(s_0)=Z\)、\(F(x)=R(x)\)，就是 TB。这个 telescoping 推导是理解各目标统一性的最佳练习。

> 记号提醒：telescoping 相消后末端得到的是**状态流** \(F(x)\)，而 TB 约束右端的 \(R(x)\) 是**终止边流** \(F(x\to s_f)\)。二者相等需要"终止状态的唯一出边是 exit"这一标准约定（Malkin et al. 2022 即采此约定）；若按 Foundations 允许终止状态另有子节点，则 \(F(x)\neq F(x\to s_f)=R(x)\)，此处应写 \(F(x\to s_f)\)。

### 3.5 Subtrajectory Balance（SubTB）

对任意部分轨迹

\[
\tau_{m:n}=(s_m\to s_{m+1}\to\cdots\to s_n),
\]

要求

\[
F_\theta(s_m)\prod_{t=m+1}^nP_{F,\theta}(s_t|s_{t-1})
=
F_\theta(s_n)\prod_{t=m+1}^nP_{B,\theta}(s_{t-1}|s_t).
\]

边界使用 \(F(s_0)=Z\)、\(F(x)=R(x)\)。归属上要区分两层：**子轨迹平衡这个约束本身**最早出现在 [Trajectory Balance 论文](https://arxiv.org/abs/2201.13259)附录 A.2（并指出 DB 是单边特例、TB 是全轨迹特例）；[SubTB(\(\lambda\))](https://proceedings.mlr.press/v202/madan23a.html)的贡献是提出以与长度有关的 \(\lambda^{n-m}\) 对所有子轨迹 loss 加权，并系统分析其收敛性与稳定性：

- 长度 1 时接近 DB；
- 完整轨迹时接近 TB；
- 中间选择试图兼顾局部低方差与长程信用。

这更准确地说是**随机梯度和信用传播的经验性折中**，不是在精确目标分布上接受偏差。若所有所需约束都达到零，目标仍是同一 reward-matching 解。

朴素枚举一条长度 \(T\) 轨迹的所有子轨迹有 \(O(T^2)\) 项，但神经网络前向结果可以复用，不能简单把它理解为 \(O(T^2)\) 次网络调用。

### 3.6 Guided Trajectory Balance（GTB）

设 \(q(\tau|x)\) 是人为指定的、在所有到达 \(x\) 的轨迹上归一化的 guide。GTB 希望满足

\[
ZP_F(\tau)=R(x)q(\tau|x).
\]

对终止于 \(x\) 的所有轨迹求和，

\[
ZP_T(x)
=R(x)\sum_{\tau\to x}q(\tau|x)
=R(x).
\]

所以 guide 可以改变路径信用而不改变理想终止边缘分布。

但任意 \(q(\tau|x)\) 未必能由 Markovian \(P_B\) 精确表示。Shen 等人的做法是先让 \(P_B\) 拟合 guide 的 Markov 近似，再固定或交替训练 \(P_F\)。因此“guide 合理”与“guide 可由当前状态表示实现”是两个不同问题。

### 3.7 \(f\)-Trajectory Balance

2026 年的 [\(f\)-Trajectory Balance](https://arxiv.org/abs/2605.15417)把平方 log-ratio 代理 loss 扩展到 \(f\)-divergence 家族：

- on-policy 时，期望梯度对应所选 \(f\)-divergence 的梯度；
- off-policy 时，在支持覆盖等条件下仍保留相同的正确全局最小点；
- 不同 \(f\) 可改变训练阶段的 mode-covering / mode-seeking 行为和梯度性质。

需要区分两个层次：

- 精确全局最小点处，目标分布可以相同；
- 到达最小点的优化路径、有限训练的覆盖倾向和方差可以很不同。

论文已列入 ICML 2026；它是理解“为什么换 loss 形状会改变 GFlowNet 学习行为”的最新统一视角之一。

### 3.8 目标函数对照表

| 目标 | 监督单位 | 主要参数 | 信用范围 | 主要优点 | 主要风险/代价 |
|---|---|---|---|---|---|
| FM | 状态 | 边流 | 局部 | 直接表达守恒，可用中间状态 | 父/子求和；长程信用慢 |
| DB | 边 | \(F(s),P_F,P_B\) | 局部 | 不必枚举全部父流；结构清晰 | 自举；对 \(P_B\) 和状态流敏感 |
| TB | 完整轨迹 | \(Z,P_F,P_B\) | 全局 | 终端奖励直达整条轨迹 | 梯度方差、完整轨迹、极端 log-ratio |
| SubTB | 多尺度子轨迹 | \(F(s),P_F,P_B\) | 局部到全局 | 可调信用尺度 | 子轨迹项多，\(\lambda\) 要调 |
| GTB | 完整轨迹+guide | \(Z,P_F,P_B\) | guide 指定 | 控制内部流与子结构信用 | guide 可能非 Markov、近似误差 |
| \(f\)-TB | 完整轨迹 | \(Z,P_F,P_B\) | 全局 | 可选 divergence 几何与覆盖倾向 | 新方法；仍依赖覆盖和优化 |

## 4. 正确性定理与有限训练之间的鸿沟

### 4.1 “零 loss 推出正确分布”真正需要什么

一个典型的正确性陈述隐含以下条件：

1. **奖励正确且非负**：终止边界确实是希望采样的 \(R(x)\)。
2. **支持条件**：训练分布对需要约束的状态、边或轨迹有全支持，或至少最终覆盖所有有正目标质量的部分。
3. **可实现性**：模型族能同时表示所需 \(P_F,P_B,F,Z\)。
4. **精确或全局优化**：相关约束在所有必要位置达到零。
5. **环境与状态表示正确**：状态包含使策略 Markov 所需的信息。
6. **有限性/可积性**：\(Z=\sum_xR(x)\) 或相应积分有限。

因此，常见说法“GFlowNet 可以任意 off-policy 训练”应改成：

> GFlowNet 的若干 balance loss 在理论上可在任意**全支持**训练分布下拥有同一个正确全局最小点，不必像普通 importance-sampling VI 那样为每个 off-policy 样本乘重要性权重。

这并不意味着：

- 一个只覆盖少数模式的固定数据集足够；
- replay buffer 的抽样偏置没有优化影响；
- 未访问的约束会自动满足；
- 有限容量网络不会因不同区域的权重而产生折中；
- off-policy 分布漂移对方差和收敛速度没有影响。

### 4.2 Shen et al. 2023：训练为什么会长期欠拟合

本调研重点分析的 [Towards Understanding and Improving GFlowNet Training](https://proceedings.mlr.press/v202/shen23a/shen23a.pdf) 是理解实践鸿沟的核心论文。它指出：

- 在巨大的组合空间里，“遍历所有状态或轨迹一次”本身就不现实。
- 训练很久甚至表面收敛后，模型仍可能对低奖励对象分配过多概率。
- 终止奖励固定，并不唯一决定内部流；内部流的选择影响共享子结构能否获得信用和模型对未见对象的泛化。
- 在线按当前 flow 抽样会产生自我强化：高流轨迹更常被抽到，又得到更多更新。在论文的简化表格设定中，这可解释为 Pólya urn 式的 “rich get richer”。

论文提出三类干预：

- **PRT（Prioritized Replay Training）**：重放时偏向高奖励样本，直接纠正低奖励过采样。
- **SSR（relative edge-flow parameterization）**：使用相对边流参数化，改变共享结构的归纳偏置。
- **GTB / substructure guide**：显式引导奖励沿被认为重要的共享子结构回传。

在一个合成 Bag 任务和四个生化设计任务、每个设置三个重复实验中，最佳组合在部分任务明显更快；sEH 上报告了相对 TB 约 9 倍更快达到目标均值。但不能把“9 倍”当作普适算法复杂度结论。

### 4.3 这篇论文的评估边界

论文使用 Anderson–Darling 统计量比较**采样奖励分布**和目标奖励分布，并比较

\[
\mathbb E_{x\sim P_\theta}[R(x)]
\quad\text{与}\quad
\mathbb E_{x\sim P^\star}[R(x)]
=\frac{\sum_xR(x)^2}{Z}.
\]

这是比只看 top-\(k\) 奖励更有信息的评估，但仍有逻辑边界：

\[
\operatorname{Law}_{P_\theta}(R(X))
=
\operatorname{Law}_{P^\star}(R(X))
\]

并不推出

\[
P_\theta(X)=P^\star(X).
\]

只要多个对象具有相同奖励，就可能在对象之间严重分配错误，却拥有完全相同的奖励直方图和平均奖励。论文作者也明确说明其方案只比较目标奖励分布。因此，复现实验时至少还应在小环境报告：

- 终止对象分布的 TV / \(L_1\) / JS；
- 各模式覆盖率与有效样本数；
- reward histogram 与均值；
- 训练 loss 与最大单轨迹残差；
- 若可算，\(\log Z_\theta\) 误差；
- 路径长度、轨迹熵和重复率。

### 4.4 2026：训练 loss 能否认证终止分布

[Stable GFlowNets with Probabilistic Guarantees](https://arxiv.org/abs/2605.01729)进一步研究 loss 与终止分布误差之间的关系。其两个方向很值得区分：

1. \(P_T\) 与目标分布的 TV 距离很小，**不排除**少数轨迹上的 TB loss 无界。也就是说，偶发 loss spike 不必然代表终止分布已经很差。
2. 反方向，如果对**每条轨迹**都有

\[
\mathcal L_{\mathrm{TB}}(\tau)\le c^2,
\]

则论文给出

\[
\operatorname{TV}(P_T,\pi_{\text{target}})
\le 1-\exp(-2c).
\]

对每个局部 DB 边残差或 FM 状态残差的相应上界，串成长为至多 \(L\) 的轨迹后会退化为

\[
\operatorname{TV}(P_T,\pi_{\text{target}})
\le 1-\exp(-2Lc).
\]

这解释了全轨迹约束为什么更容易给出不随深度线性累积的全局界。

但这一漂亮结果有很强前提：“所有轨迹”上的一致界通常无法直接检查。论文进一步用从目标诱导轨迹分布和 \(P_F\) 双向抽样构造概率证书，并提出 reference flow 稳定极端比率。实践限制包括：

- 从真实目标分布抽样本身可能就是原问题；
- 大空间常只能在已发现的高奖励子图上做证书；
- 最坏情形界可能保守；
- 添加 reference flow 会显式引入稳定性与目标保真度的折中；
- 截至本报告日期，该工作是 2026 年预印本，结论尚需更多独立复现。

### 4.5 三种“误差”不要混为一谈

| 层次 | 典型量 | 回答的问题 |
|---|---|---|
| 局部一致性误差 | FM/DB 残差 | 单个状态或边是否守恒 |
| 轨迹一致性误差 | TB/SubTB 残差 | 前向与反向的整段质量是否相配 |
| 终止分布误差 | TV、KL、JS、对象频率误差 | 最终是否按 \(R/Z\) 采样 |

它们在理想条件下有关联，但有限样本中不能把任意一个当成另外两个的同义词。

## 5. GFlowNet 与其他方法的精确关系

### 5.1 与强化学习：相同的壳，不同的默认目标

二者都有状态、动作、策略和长期信用分配，但：

- 标准 RL 通常最大化 \(\mathbb E[R(X)]\)，趋向把质量集中到最高奖励模式。
- GFlowNet 要匹配 \(P_T(x)\propto R(x)\)，保留各模式与其奖励相称的质量。

最大熵 RL 与 GFlowNet 有更紧密的关系。[Generative Flow Networks as Entropy-Regularized RL](https://proceedings.mlr.press/v238/tiapkin24a.html)给出带特定奖励和正则结构的通用重写；[Discrete Probabilistic Inference as Control in Multi-path Environments](https://proceedings.mlr.press/v244/deleu24a.html)进一步澄清：

- 在树中，每个对象只有一条路径，某些 soft RL 形式很直接。
- 在多路径 DAG 中，朴素 MaxEnt RL 的终止概率会被路径数或路径熵偏置。
- 使用结构相关的奖励校正后，部分 GFlowNet 目标可等价于已知 MaxEnt RL 算法。

所以“GFlowNet 就是 MaxEnt RL”太宽泛；正确表述应带上**奖励修正、路径结构和算法形式**。

2024 年的 [GFlowNet Training by Policy Gradients](https://proceedings.mlr.press/v235/niu24c.html)则从 policy-dependent reward 出发建立策略梯度式训练，并联合优化反向策略。2026 年的 [Proximal Policy Optimization for Amortized Discrete Sampling](https://arxiv.org/abs/2606.15793)进一步推导 GFlowNet 语境下的 policy-gradient/PPO 版本。截至截止日，后者仍应视为最新预印本证据，而不是已确立的默认训练方案。

### 5.2 与变分推断：轨迹空间上的近似推断

在目标

\[
\pi^\star(x)=\frac{R(x)}{Z}
\]

中，\(x\) 是可见的终止对象，而生成路径 \(\tau\) 可看成辅助潜变量。前向轨迹分布是 proposal，\(R(x)P_B(\tau|x)/Z\) 是反向扩展出的目标轨迹分布。

[GFlowNets and Variational Inference](https://arxiv.org/abs/2210.00580)与 [A Variational Perspective on Generative Flow Networks](https://arxiv.org/abs/2210.07992)表明，在特定参数化和 on-policy 期望梯度下，TB 等训练与层次 VI / KL 优化有精确联系。

必须保留三个限定词：

- 常见结果是**期望梯度等价**，不是两个逐样本 loss 完全相同。
- 通常要求 on-policy 或相应控制变量条件。
- GFlowNet 的 log-ratio 回归 loss 在全支持 off-policy 数据上仍可保留正确零点；普通 VI 若直接换采样分布，通常需要 importance weighting。

2024 年 NeurIPS 论文 [On Divergence Measures for Training GFlowNets](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8aaae73117c9266b35eb977e70fcf95f-Abstract-Conference.html)把这一联系扩展到一般可测拓扑空间，并指出传统 divergence 目标表现不佳的重要原因是随机梯度方差；论文用 control variates 缓解它。2026 年 \(f\)-TB 则进一步统一了 divergence 与 off-policy 代理 loss。

### 5.3 与 MCMC：同一未归一化目标，不同的成本位置

共同点：

- 都可以只使用未归一化的 \(R(x)\) 或能量 \(E(x)=-\log R(x)\)。
- 都希望从多峰目标分布采样。

差异：

- MCMC 每个新样本通常要继续运行转移链，并依赖混合性质。
- GFlowNet 先花成本学习摊销策略；训练后每个样本只需执行一条构造轨迹。
- MCMC 的局部转移在固定目标下有渐近保证；神经 GFlowNet 通常是近似 sampler。
- GFlowNet 需要主动探索和函数逼近；未发现模式不会自动得到训练信号。

“GFlowNet 不受 mode mixing 影响”也不能绝对化。它不需要在推断时沿同一马尔可夫链从一个远端模式混到另一个，但训练期仍可能因为探索失败、mode collapse 或共享前缀偏置而漏掉模式。

### 5.4 与 normalizing flows / flow matching

| 名称 | 核心对象 | 是否需要可逆映射/Jacobian | “flow”的含义 |
|---|---|---|---|
| GFlowNet | 状态图与轨迹 | 否 | 未归一化概率质量 |
| Normalizing flow | 连续变量的可逆变换 | 通常显式需要 | 密度经变换搬运 |
| 连续生成模型的 flow matching | 时间连续向量场/ODE | 训练通常不需显式 Jacobian；ODE 解在正则条件下可逆 | 概率路径的速度场 |

三者可被更大的生成/传输视角联系起来，但不能仅凭名字互换定理或实现。

### 5.5 与最优传输：指定 2026 论文的准确解读

[Your GFlowNet Secretly Learns an Optimal Transport Plan](https://arxiv.org/abs/2606.06272)考虑一个有限、允许有环的有向图，并作以下关键设定：

- \(U\)：源点之后的首步状态集合，固定初始流分布 \(L(u)\)；
- \(X\)：终止状态集合，固定终止奖励分布 \(R(x)\)；
- \(\sum_uL(u)=\sum_xR(x)=1\)；
- 每个 \(u\) 到每个相关 \(x\) 可达；
- 在满足流守恒、源端和终端边界的流中，最小化内部总边流。

把内部边集合记为 \(E^\circ\)，其约化线性规划为

\[
\min_{F(e)\ge 0}\sum_{e\in E^\circ}F(e)
\]

subject to

\[
\sum_{e\in\operatorname{Out}(s)}F(e)
-
\sum_{e\in\operatorname{In}(s)}F(e)
=L(s)-R(s).
\]

在非无环 flow 的 expected-visit-count 定义下，

\[
\sum_{e\in E^\circ}F(e)=\mathbb E[|\tau|]-1,
\]

减去的 1 对应不属于内部边集 \(E^\circ\) 的固定首步 \(s_0\to u\)。因为该常数不影响 \(\arg\min\)，所以目标**等价于**在满足边缘质量的同时最小化期望轨迹长度（原论文亦声明省略此常数）。

令

\[
d(u,x)=\text{图中从 \(u\) 到 \(x\) 的最短路长度}.
\]

论文定理 3.2 证明上述最优值等于 Kantorovich OT：

\[
\min_{\Pi\in\Gamma(L,R)}
\sum_{u,x}d(u,x)\Pi(u,x).
\]

证明的两个方向非常直观：

1. 给定 OT coupling \(\Pi(u,x)\)，把每对 \((u,x)\) 的质量沿任意最短路发送，得到可行流，代价等于 OT。
2. 给定可行 GFlowNet flow，按其策略采样轨迹，令

\[
\Pi(u,x)=\Pr(\tau\text{ 从 }u\text{ 到 }x),
\]

得到边缘为 \(L,R\) 的 coupling；每条实际路径不短于 \(d(u,x)\)，故流代价不低于该 coupling 的 OT 代价。

两侧夹逼即得最优值相等，而且最优流诱导最优 coupling。

#### 不能省略的限定

标题中的 “Your GFlowNet” 是吸引人的概括，严格结论其实是：

> 固定源边缘 \(L\)、固定目标边缘 \(R\)、使用图最短路作 ground cost，并达到最小总流全局最优的非无环 GFlowNet，编码一个 OT plan。

因此，下列说法都不成立：

- 普通单源 DAG reward-matching GFlowNet 自动在做 OT；
- 任意 TB 训练结果自动是 OT 最优解；
- 任意图代价都能由该定理覆盖；
- 神经网络正则化训练达到的近似解与线性规划最优解完全相同。

这项工作的思想价值在于：一般 reward matching 只规定“最终到哪里”，没有唯一规定“沿哪条路径、从哪部分源质量过去”；minimum-flow 原则在特定非无环设定中选择最短路径，并同时学习 coupling 和局部路由策略。论文的神经实验使用正则化 TB，在 hypergrid 与 permutation 环境中验证与精确 OT solver 的接近程度，也观察到更强 flow regularization 会缩短路径，却可能增加终止分布偏差。

## 6. 理论扩展

### 6.1 条件 GFlowNet 与摊销边缘化

若奖励依赖条件 \(y\)，目标变成

\[
P_T(x|y)=\frac{R(x|y)}{Z(y)}.
\]

用同一个网络接收 \(y\)，可以跨许多相关目标摊销采样和配分函数估计。进一步地，中间状态流有时可以解释为对所有完成方式的未归一化求和：

\[
F(s)
\approx
\sum_{x\succeq s}R(x)\times
\text{与后向路径分配有关的权重}.
\]

这使 GFlowNet 不只是 sampler，也可能成为摊销 marginalization estimator。适合的直观材料是 Mila 的 [GFlowNet and Amortized Marginalization Tutorial](https://milayb.notion.site/The-GFlowNets-and-Amortized-Marginalization-Tutorial-01755ca312834e15ab0ae9ef46bcb1bb)。

但要注意：

- 不同 \(y\) 下 \(Z(y)\) 的尺度可跨越很多数量级；
- 条件外推仍依赖表示与训练分布；
- 状态流的概率解释依赖具体 flow 和 \(P_B\)，不能脱离定义任意解释。

### 6.2 中间奖励与不完整轨迹

[Better Training of GFlowNets with Local Credit and Incomplete Trajectories](https://proceedings.mlr.press/v202/pan23c.html)研究当能量/奖励可在中间状态计算或分解时，如何重参数化状态流：

- 不必等到完整轨迹才获得所有信号；
- 可利用局部增量能量；
- 对长轨迹能改善信用分配；
- 可从不完整 episode 学习。

这一方法不是无条件地“给中间状态随便加 reward”；它要求中间量与最终能量之间具有可利用的分解或一致关系。

### 6.3 连续状态空间

[A Theory of Continuous Generative Flow Networks](https://proceedings.mlr.press/v202/lahlou23a.html)把离散求和改写为测度、kernel 与积分：

- 状态和轨迹流成为测度；
- \(P_F,P_B\) 成为 Markov kernels；
- 局部 balance 要以密度或 Radon–Nikodym 导数表达；
- 需要明确绝对连续性、边界和可积性条件。

它说明 GFlowNet 思想不限于离散图，但连续实现不能只把离散概率替换成“某个网络输出的 density”而忽略基准测度和 Jacobian/几何因素。

### 6.4 非无环环境

在 DAG 中，轨迹必然有限；允许环后必须额外处理：

- 策略是否以概率 1 最终吸收；
- 一条轨迹可重复访问状态和边；
- 状态/边流应解释为 expected visit counts；
- 环上的无效循环可能让总流或期望长度发散；
- 原有 FM/DB/TB 形式可能鼓励流困在环中。

[A Theory of Non-acyclic Generative Flow Networks](https://ojs.aaai.org/index.php/AAAI/article/view/28989)首先在一般可测空间中系统放宽无环假设并分析 loss 的 cycle instability。[Revisiting Non-Acyclic GFlowNets in Discrete Environments](https://proceedings.mlr.press/v267/morozov25a.html)在有限离散情形给出更简洁的框架，讨论固定 \(P_B\)、流的唯一性、稳定性和 entropy-regularized RL 联系。

学习顺序上，建议先完全掌握 DAG Foundations，再读非无环理论，最后读 2026 OT 论文；反过来很容易把 expected visit flow 与 DAG 中“一条轨迹至多访问一次”的流混淆。

## 7. 经典文献脉络

### 7.1 奠基与训练目标

| 年份 | 文献 | 状态 | 最该读懂的内容 |
|---|---|---|---|
| 2021 | [Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation](https://proceedings.neurips.cc/paper/2021/hash/e614f646836aaed9f89ce58e837e2310-Abstract.html) | NeurIPS 2021 | 原始 flow matching、多路径对象、分子设计与主动学习动机 |
| 2021/2023 | [GFlowNet Foundations](https://jmlr.org/papers/v24/22-0364.html)（[arXiv 2111.09266](https://arxiv.org/abs/2111.09266)） | JMLR 2023；arXiv 版本 2026 更新 | 轨迹流、Markovian flow、\(P_F/P_B\)、reward matching、DB、条件/边缘扩展 |
| 2022 | [Trajectory Balance](https://proceedings.neurips.cc/paper_files/paper/2022/hash/27b51baca8377a0cf109f6ecc15a0f70-Abstract.html) | NeurIPS 2022 | 完整轨迹约束、\(Z\) 学习、长程信用 |
| 2022/2023 | [Learning GFlowNets from partial episodes for improved convergence and stability](https://proceedings.mlr.press/v202/madan23a.html) | ICML 2023 | SubTB(\(\lambda\))、局部到全局的信用尺度 |
| 2023 | [Better Training with Local Credit and Incomplete Trajectories](https://proceedings.mlr.press/v202/pan23c.html) | ICML 2023 | 中间能量、不完整轨迹 |
| 2023 | [Towards Understanding and Improving GFlowNet Training](https://proceedings.mlr.press/v202/shen23a.html) | ICML 2023 | 欠拟合、内部流与泛化、PRT/SSR/GTB、评估局限 |

“Non-Iterative” 容易误导。它指训练后不必像 MCMC 那样为每个新样本进行长时间迭代混合；GFlowNet 采样本身通常仍是多步自回归/构造过程，不是一次神经网络前向就得到完整对象。

### 7.2 与 VI、RL 和更一般空间的连接

| 年份 | 文献 | 状态 | 主要贡献 |
|---|---|---|---|
| 2022/2023 | [GFlowNets and Variational Inference](https://arxiv.org/abs/2210.00580) | ICLR 2023 | GFlowNet 与层次 VI 的梯度联系、off-policy 对比 |
| 2022/2023 | [A Variational Perspective on Generative Flow Networks](https://openreview.net/forum?id=AZ4GobeSLq) | TMLR 2023 | 前/反向轨迹 KL、TB 与 VI 的统一解释 |
| 2023 | [A Theory of Continuous Generative Flow Networks](https://proceedings.mlr.press/v202/lahlou23a.html) | ICML 2023 | 测度论连续/混合空间框架 |
| 2023 | [Generative Flow Networks: a Markov Chain Perspective](https://arxiv.org/abs/2307.01422) | 立场/预印本 | 用 recurrent Markov chain 统一比较 GFN 与 MCMC |
| 2024 | [Generative Flow Networks as Entropy-Regularized RL](https://proceedings.mlr.press/v238/tiapkin24a.html) | AISTATS 2024 | 一般 MaxEnt RL 重写 |
| 2024 | [Discrete Probabilistic Inference as Control in Multi-path Environments](https://proceedings.mlr.press/v244/deleu24a.html) | UAI 2024 | 多路径偏置、奖励校正、与 MaxEnt RL 算法等价 |
| 2024 | [GFlowNet Training by Policy Gradients](https://proceedings.mlr.press/v235/niu24c.html) | ICML 2024 | 策略梯度框架、联合反向策略设计 |
| 2024 | [On Divergence Measures for Training GFlowNets](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8aaae73117c9266b35eb977e70fcf95f-Abstract-Conference.html) | NeurIPS 2024 | divergence 视角、梯度方差与 control variates |

### 7.3 环、最短路与最优传输

| 年份 | 文献 | 状态 | 主要贡献 |
|---|---|---|---|
| 2024 | [A Theory of Non-acyclic Generative Flow Networks](https://ojs.aaai.org/index.php/AAAI/article/view/28989) | AAAI 2024 | 放宽无环假设、cycle instability、稳定 loss |
| 2025 | [Revisiting Non-Acyclic GFlowNets in Discrete Environments](https://proceedings.mlr.press/v267/morozov25a.html) | ICML 2025 | 离散非无环简化理论、固定 \(P_B\)、最小流 |
| 2026 | [Learning Shortest Paths with GFlowNets](https://arxiv.org/abs/2603.01786) | 预印本 | 最小总流选择最短路径及 Rubik's cube 实验 |
| 2026 | [Your GFlowNet Secretly Learns an Optimal Transport Plan](https://arxiv.org/abs/2606.06272) | ICML 2026 SPIGM Workshop；arXiv | 固定源分布的 minimum-flow GFN 与图 OT 等价 |

## 8. 截至 2026-07-31 的前沿

这一部分按“对理论主线的重要程度”而非单纯按发布时间排序。2026 年论文除明确标注主会或 Workshop 者外，均应按预印本看待；Workshop 不能写成主会录用。

### 8.1 稳定性和可认证性

[Stable GFlowNets with Probabilistic Guarantees](https://arxiv.org/abs/2605.01729)把“loss 小是否意味着终止分布好”变成显式的 TV 界与有限样本证书问题。这是对 Shen 2023 有限训练观察的理论延伸，但其目标抽样和一致最坏界假设使它暂时更像理论诊断工具，而非已成熟的通用训练配方。

### 8.2 统一 loss 与 divergence

[\(f\)-Trajectory Balance](https://arxiv.org/abs/2605.15417)给出从 translation-invariant log-probability loss 到 \(f\)-divergence 的一一对应，并保留 off-policy 正确零点。它把“标准平方 TB 为什么有某种 mode 行为、如何系统换一种行为”从经验超参数问题提升为 divergence 设计问题。

### 8.3 前缀信用与 replay

[Rooted Absorbed Prefix Trajectory Balance with Submodular Replay（RapTB）](https://arxiv.org/abs/2603.00454)针对 LLM/序列生成里的 prefix collapse 和 length bias，用根锚定的 absorbed-prefix 监督加强早期前缀信用，并用 submodular replay 同时追求高奖励与多样性。论文已列入 ICML 2026，但其优势仍应按具体序列与分子任务解读，不能替代一般终止分布评估。

### 8.4 探索—利用可控化

[Controlling Exploration–Exploitation in GFlowNets via Markov Chain Perspectives](https://arxiv.org/abs/2602.01749)提出 \(\alpha\)-GFN，通过 Markov-chain/reversibility 视角调节探索与利用，并给出相应唯一 flow 性质。它回应了一个长期问题：目标分布固定不代表训练期 proposal 只能有一种形态。当前应视为有吸引力的预印本方向。

### 8.5 把成熟 RL 优化器迁入 GFlowNet

[Proximal Policy Optimization for Amortized Discrete Sampling](https://arxiv.org/abs/2606.15793)基于 GFlowNet–entropy-RL 联系推导 policy-gradient 与 PPO 训练，报告更快收敛和更高数据效率。它说明“balance regression”不是唯一优化接口，但正确性仍来自特定改写后的分布匹配 reward/regularizer，不是直接用普通 PPO 最大化原始 \(R(x)\)。

### 8.6 最小流、最短路和 OT

[Your GFlowNet Secretly Learns an Optimal Transport Plan](https://arxiv.org/abs/2606.06272)是 2026 年最漂亮的结构性结果之一：minimum-flow 不只是让路径短，它在固定源/目标边缘时恰好选择图最短路代价下的 OT coupling。它也为“在许多正确内部流中应选择哪一个”提供了可优化的归纳原则。

发表状态必须写准确：该文是 **ICML 2026 SPIGM Workshop**，不是 ICML 2026 主会论文。它的理论价值不因 venue 降低，但现阶段证据强度仍是“一篇很新的 workshop/arXiv 论文提出的结构性等价”，尚未形成被多篇独立工作验证的成熟研究线。

### 8.7 大语言模型规模化

[GFlowRL: Scaling Distribution-Matching RL to Large Language Models](https://arxiv.org/abs/2607.13394)于 2026-07-15 提交，是本次检索中截止日之前很新的应用/系统方向。它：

- 用 rollout group 内的 Monte Carlo 估计替代额外的 prompt-conditional \(Z\) 网络；
- 加入 rollout/trainer drift 的 importance correction；
- 对 flow gap 使用非对称 clipping；
- 报告在 dense 与 MoE LLM 上的扩展结果。

这更像“大模型后训练中的 GFlowNet-style distribution-matching RL”，不应反过来改写基础 GFlowNet 定义。特别是以 batch 估计替代显式 \(Z_\theta\) 是特定大模型训练构造，不意味着 Foundations 中的总流概念不重要。

### 8.8 2024–2026 顶会主线

完整的逐 venue 主会核验表见[《GFlowNet 论文、课程与代码清单》§5](GFLOWNET_RESEARCH_RESOURCE_CATALOG_CN.md#5-20242026-顶会覆盖审计)。如果只想抓住最近两届最有理论含量的变化，优先读：

| 主线 | 2024 | 2025 | 2026 |
|---|---|---|---|
| 正确性与 loss | [Divergence Measures（NeurIPS）](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8aaae73117c9266b35eb977e70fcf95f-Abstract-Conference.html) | [When Do GFlowNets Learn the Right Distribution?（ICLR）](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a48f8928f78a58399ef0049453c14b02-Abstract-Conference.html)、[Beyond Squared Error（ICLR）](https://proceedings.iclr.cc/paper_files/paper/2025/hash/353ec686503cd7020460d2829578ee4e-Abstract-Conference.html) | [\(f\)-TB（ICML）](https://icml.cc/virtual/2026/poster/61247)、[Evaluation Balance（ICLR）](https://iclr.cc/virtual/2026/poster/10007783) |
| \(P_B\)、信用与探索 | [Pessimistic \(P_B\)（NeurIPS）](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c1ab28d0fe0bfb53067a1af7e578cd7d-Abstract-Conference.html)、[QGFN（NeurIPS）](https://proceedings.neurips.cc/paper_files/paper/2024/hash/948d8ba4e30c8c3a800cf436b31f376e-Abstract-Conference.html) | [Optimizing Backward Policies（ICLR）](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f3efbcfe76bed022a37c5aeb1daf2326-Abstract-Conference.html)、[Sibling GFN（ICLR）](https://iclr.cc/virtual/2025/poster/30233) | [ACE（ICML）](https://icml.cc/virtual/2026/poster/62783)、[RapTB（ICML）](https://icml.cc/virtual/2026/poster/65366)、[TD-GFN（ICML）](https://icml.cc/virtual/2026/poster/62632) |
| 空间与内部流 | [Non-Acyclic Theory（AAAI）](https://ojs.aaai.org/index.php/AAAI/article/view/28989) | [Revisiting Non-Acyclic（ICML）](https://proceedings.mlr.press/v267/morozov25a.html)、[Symmetry-Aware GFN（ICML）](https://proceedings.mlr.press/v267/kim25s.html) | shortest path 与 GFN–OT 仍是预印本/Workshop 线，不应伪装成主会成果 |
| 规模化与应用 | [EP-GFN（ICML）](https://proceedings.mlr.press/v235/silva24a.html)、[RGFN（NeurIPS）](https://proceedings.neurips.cc/paper_files/paper/2024/hash/53704142f230054140418ecd8857f391-Abstract-Conference.html) | [Nabla-GFN（ICLR）](https://iclr.cc/virtual/2025/poster/30600)、[GFlowVLM（CVPR）](https://openaccess.thecvf.com/content/CVPR2025/html/Kang_GFlowVLM_Enhancing_Multi-step_Reasoning_in_Vision-Language_Models_with_Generative_Flow_CVPR_2025_paper.html) | [Stable-GFN（ICML Spotlight）](https://icml.cc/virtual/2026/poster/64302)、[Flow of Spans（ICLR）](https://iclr.cc/virtual/2026/poster/10007998) |

从密度上看，近两年的 GFlowNet 前沿主要集中在 **ICML、ICLR、NeurIPS 和 AAAI**。CVPR 2025 有 GFlowVLM 和 PAG 两篇直接应用；截至核验日，ICCV 2025 与 ECCV 2024 官方 proceedings 未见同等直接的 GFlowNet 核心论文，CVPR 2024/2026 也未检索到直接核心论文。这里没有用 normalizing flow、rectified flow、flow matching 或光流论文补数量。

### 8.9 当前真正开放的问题

综合经典和最新工作，理论上最值得追踪的是：

1. **有限覆盖理论**：训练只看到指数空间的极小部分时，什么结构能保证终止分布泛化？
2. **内部流选择**：除固定 \(P_B\)、guide、minimum-flow 外，什么原则能稳定地产生有利于泛化的 flow？
3. **可计算证书**：不先从真实目标分布采样，能否可靠上界全局分布误差？
4. **探索保证**：在黑盒、多峰、昂贵奖励下，怎样避免 mode/prefix collapse？
5. **长轨迹尺度**：如何控制 log-ratio、\(Z\)、长度偏置和梯度方差？
6. **非无环与连续空间**：如何同时保证吸收、有限 expected visits、数值稳定和可扩展性？
7. **条件化与 reward scale**：单一模型跨温度、任务或 context 时，如何校准 \(Z(y)\) 与相对概率？
8. **评价标准**：大空间中无法枚举 \(P^\star\) 时，哪些可计算指标真的反映对象级分布拟合？

## 9. 博客、教程和代码资源怎么用

### 9.1 入门材料

建议按以下顺序，而不是一开始硬啃 Foundations：

1. [The GFlowNet Tutorial](https://milayb.notion.site/The-GFlowNet-Tutorial-95434ef0e2d94c24aab90e69b30be9b3)  
   Yoshua Bengio、Nikolay Malkin、Moksh Jain 的高层教程。先抓住“逐步构造、概率正比于奖励、同一对象多条路径”。

2. [Emmanuel Bengio 的早期 GFlowNet 博客](https://folinoid.com/w/gflownet/)  
   水流直觉、原始 FM、摊销采样、温度 \(R^\beta\)、分子设计都解释得很直观。

3. [Yoshua Bengio：Generative Flow Networks](https://yoshuabengio.org/en/blog/generative-flow-networks)  
   了解这条研究线为何处于 RL、生成模型、能量模型和 VI 的交叉点。博客中的长期 AI 愿景属于研究判断，不要与已经证明的定理混为一谈。

4. [Mila：What do GFlowNets and Variational Inference Have in Common?](https://mila.quebec/en/article/what-do-gflownets-and-variational-inference-have-in-common)  
   适合在学完 TB 后建立 VI 直觉。

5. [Mila GFlowNet Workshop 资源页](https://www.gflownet.org/resources.html)  
   有基础理论、训练、条件化、应用和 live coding 的视频/幻灯片入口。

### 9.2 实现资源

- [torchgfn](https://github.com/GFNOrg/torchgfn)：模块化 PyTorch 库，环境、sampler、estimator 与 loss 分离，适合学习和算法原型；[文档](https://gfn.readthedocs.io/en/latest/)。
- [Recursion/Valence GFlowNet](https://github.com/recursionpharma/gflownet)：偏图和分子生成，包含 FM、TB、SubTB、online/offline 混合训练；[文档](https://gflownet.readthedocs.io/en/latest/)。
- [gfnx](https://github.com/d-tiapkin/gfnx)：JAX 实现，强调快速和可扩展 benchmark；适合已理解算法后比较性能。
- [Awesome-GFlowNets](https://github.com/zdhNarsil/Awesome-GFlowNets)：社区维护的论文与资源索引，可用于继续跟踪，但引用具体结论时应回到原论文。

第一次实现建议选择 `torchgfn` 的 HyperGrid，而不是直接做分子。只有在可以枚举终止空间时，你才能清楚区分“loss 下降”“平均奖励上升”和“对象分布真的正确”。

## 10. 六周学习路线

每周按 6–10 小时设计；若数学基础较强，可以压缩到三周。

### 第 0 周：补齐先修概念

需要能解释：

- 未归一化密度、配分函数、能量 \(E(x)=-\log R(x)\)；
- DAG、拓扑序、Markov policy；
- KL、TV、JS 和 importance sampling；
- TD learning、自举和 credit assignment；
- MCMC 的 detailed balance 与 mixing；
- Kantorovich OT 的 coupling 线性规划。

通过标准：能写出一个 \(2\times2\) 离散 OT LP，并能说明最大熵 RL 与普通 RL 的目标差别。

### 第 1 周：流守恒和终止分布

阅读：

- Notion GFlowNet Tutorial；
- Emmanuel Bengio 博客；
- 原始 NeurIPS 2021 论文第 1–3 节；
- Foundations 第 2 节。

手推：

1. 在一个 diamond DAG 上列出所有轨迹。
2. 从轨迹流计算状态流、边流、\(P_F,P_B\)。
3. 证明守恒与终止奖励推出 \(P_T=R/Z\)。
4. 改变内部路径分流比例，验证 \(P_T\) 不变。

通过标准：不看资料写出核心定理，并解释为什么图是 DAG 而不是神经网络结构。

### 第 2 周：DB、TB 和 SubTB

阅读：

- Foundations 第 2.4–3.3 节；
- TB 论文；
- SubTB 论文。

手推：

1. DB 沿路径 telescoping 得到 TB。
2. 对一条长度 3 的路径写出全部 6 个非空 SubTB 区间。
3. 说明 \(\lambda\) 改变的是优化信用尺度，而不是精确终点目标。

实现：

- 在同一小型 HyperGrid 上比较 FM/DB/TB/SubTB；
- 每隔固定步数精确枚举 \(P_T\)，画 TV、平均奖励、loss 三条曲线。

通过标准：能从现象上举例说明低 loss、较高平均奖励、低 TV 为什么不是同一件事。

### 第 3 周：训练行为与内部流

精读 Shen et al. 2023：

- 第 3 节：低奖励过采样与评估；
- 第 4 节：flow distribution 和 generalization；
- 第 5–6 节：共享子结构与 GTB；
- 附录中的简化假设。

实验：

- 添加 reward-prioritized replay；
- 固定 uniform \(P_B\) 与学习 \(P_B\) 对比；
- 记录终点 TV、轨迹熵、中间状态流与模式覆盖；
- 构造“相同奖励、不同对象”的终点，观察 reward histogram 指标失灵。

通过标准：能解释“\(P_B\) 不影响精确目标分布”与“\(P_B\) 会强烈影响训练”为何不矛盾。

### 第 4 周：VI、RL 与 MCMC

阅读：

- GFlowNets and Variational Inference；
- A Variational Perspective；
- Entropy-Regularized RL；
- Discrete Probabilistic Inference as Control。

手推：

1. 写出前向轨迹分布 \(Q(\tau)\)。
2. 写出反向目标轨迹分布 \(P(\tau)\propto R(x)P_B(\tau|x)\)。
3. 比较 TB log-ratio 与 \(\log Q-\log P\)。
4. 在树和多路径 DAG 上比较朴素 MaxEnt RL 终点分布。

通过标准：每次说“等价”时，都能补出是目标值、零点、期望梯度，还是某种奖励校正后的算法等价。

### 第 5 周：连续与非无环 GFlowNet

阅读：

- Continuous GFlowNet；
- AAAI 2024 non-acyclic theory；
- ICML 2025 revisiting non-acyclic。

重点问题：

- DAG 证明哪里使用了“每条轨迹有限且不重复状态”？
- 有环时为何 flow 变为 expected visit count？
- 什么条件保证最终吸收？
- 为什么无效环会增加总流/推断成本？

通过标准：能说明把一个可逆编辑环境直接套用 DAG TB 会有什么理论缺口。

### 第 6 周：2026 前沿与复现

阅读顺序：

1. Stable GFlowNets；
2. \(f\)-TB；
3. Learning Shortest Paths；
4. OT 论文；
5. 根据兴趣选 RapTB、PPO 或 GFlowRL。

建议复现一个最小 OT 例子：

- 两个源状态、两个终止状态；
- 手算最短路代价矩阵；
- 用 `scipy.optimize.linprog` 或 POT 求 OT；
- 解 edge-flow LP；
- 比较两个最优值和诱导 coupling；
- 再用神经 TB + flow regularization，观察近似误差和路径长度折中。

通过标准：能准确列出 OT 定理的全部前提，并给出一个不满足前提的普通 GFlowNet 反例。

## 11. 建议完成的十个推导与实验

### 练习 1：两条路径、一个终点

证明第 2.5 节的 \(q\)-参数族全部 reward-matching，并计算不同 \(q\) 下的 \(P_B(a|x)\)、\(P_B(b|x)\)。

### 练习 2：路径数偏置

令两个终点 \(x_1,x_2\) 奖励相同，但分别有 1 条和 10 条等长路径。在对轨迹做均匀或朴素最大熵采样时计算终点概率，再与 GFlowNet reward matching 比较。

### 练习 3：TB 的 telescoping

从逐边 DB 写到完整 TB，不跳过任何 \(F(s_t)\) 的消去步骤。

### 练习 4：固定 \(P_B\) 的唯一 flow

从终止流 \(R(x)\) 开始，按逆拓扑序递推

\[
F(s\to s')=F(s')P_B(s|s'),
\]

并求 \(F(s)\) 和 \(P_F\)。这给出 DAG 情形唯一性的构造性证明。

### 练习 5：相同奖励分布、错误对象分布

取四个对象，奖励分别为 \((1,1,10,10)\)。构造一个与目标具有相同 reward histogram、但在每个同奖励对象内部严重偏置的模型分布。

### 练习 6：off-policy 支持反例

构造只从一条路径收集的离线数据。让模型在该路径 TB loss 为零，却在未覆盖终点上任意错误，说明 full support 不能从定理中删除。

### 练习 7：reward scale 与温度

验证：

\[
\frac{cR(x)}{\sum_{x'}cR(x')}=\frac{R(x)}{Z},
\]

所以乘正常数不改变目标；但

\[
P_\beta(x)=\frac{R(x)^\beta}{\sum_{x'}R(x')^\beta}
\]

会改变温度、熵和模式质量。

### 练习 8：TV 与极端 TB loss

在一个已有良好分布上添加目标质量为 \(\varepsilon\) 的新模式，而模型给它远小于 \(\varepsilon\) 的概率。观察整体 TV 可很小，但该模式相关 log-ratio 可任意大。

### 练习 9：最小流即最短路

固定一个源与一个终点，证明在所有能送 1 单位质量的可行流中，最小总边流等于最短路长度。

### 练习 10：从最短路到 OT

推广到源分布 \(L\) 和目标分布 \(R\)，分别完成 OT coupling \(\to\) edge flow、edge flow \(\to\) coupling 的两个方向。

## 12. 第一个严谨实验的建议配置

### 12.1 环境

- 2D 或 3D HyperGrid；
- 边长小到可枚举全部终点；
- 奖励含多个已知峰；
- 至少设计一组多路径状态合并。

### 12.2 对比

- FM；
- DB + fixed uniform \(P_B\)；
- DB + learned \(P_B\)；
- TB；
- SubTB(\(\lambda\))；
- TB + prioritized replay。

### 12.3 必报指标

\[
\operatorname{TV}(P_\theta,P^\star)
=\frac12\sum_x|P_\theta(x)-P^\star(x)|.
\]

再报告：

- object-level \(L_1\)、JS 或 KL（注意零概率处理）；
- \(|\log Z_\theta-\log Z|\)；
- mode recall / precision；
- mean reward 与 reward histogram；
- 平均及尾部轨迹长度；
- batch mean loss、max loss、分位数；
- 每个结果至少多个随机种子。

在不可枚举任务中，可用 held-out 可枚举子空间、重要性估计或双向轨迹诊断，但必须明确它们不是全局 TV 的无偏替代。

### 12.4 工程检查表

- 全部流、奖励和概率乘积在 log space 计算。
- action mask 在归一化前正确应用。
- 统一“终止对象 \(x\)”与“形式汇点 \(s_f\)”的约定。
- 确保每个终止对象的所有合法父路径都被表示。
- 对图/集合对象做 canonicalization，避免把同一对象误当多个终点。
- 明确 stop action：何时可停、是否会造成长度偏置。
- 记录 reward clipping、下界 \(\epsilon\) 和温度 \(\beta\)；它们会改变实际目标。
- 检查 replay 是否覆盖正目标质量区域，而不只是高奖励 top-\(k\)。
- 分开优化 \(\log Z\) 与策略时，记录各自学习率和梯度尺度。
- 不以 training loss 单独早停；至少同时看分布或覆盖代理指标。

一个最小 TB 训练循环的逻辑是：

```text
repeat:
    用探索/前向策略采样完整轨迹 tau，得到终点 x
    log_forward  = sum log P_F(action_t | state_t)
    log_backward = sum log P_B(reverse_action_t | next_state_t)
    residual = logZ + log_forward - logR(x) - log_backward
    loss = residual^2
    更新 P_F、P_B、logZ
```

真正困难的部分不在这六行，而在环境的状态等价类、反向动作、探索分布、reward scale、replay 和对象级评估。

## 13. 常见误区速查

### “GFlowNet 会均匀地产生所有高奖励解”

不会。它按 \(R(x)\) 比例采样。只有奖励在某个候选集合内相同，才在该集合内理想均匀。

### “乘大一点 reward 会更偏向高奖励”

把所有奖励乘同一个常数不改变目标分布，只改变 \(Z\) 的尺度。使用 \(R^\beta\)、\(\exp(\beta\log R)\) 或其他非线性变换才改变温度。

### “TB 比 DB 理论上更正确”

在精确满足条件时都可得到正确终止分布。差别主要在参数化、信用尺度、随机梯度和训练可达性。

### “off-policy 不需要 importance weights，所以任何 replay 都无偏”

正确零点可以不变，不等于有限函数逼近下的 SGD 更新没有采样加权影响；无支持的数据更不可能约束未见区域。

### “平均奖励达到目标均值，说明分布学对了”

一个标量矩无法识别整个对象分布。即使完整 reward histogram 一致，也可能在等奖励对象之间分配错误。

### “GFlowNet 保证比 MCMC 更快找到新模式”

没有这种无条件保证。GFlowNet 把成本摊销并可能跨已学结构泛化；MCMC 有自己的渐近性质。哪种更好取决于目标结构、训练预算、混合和样本复用次数。

### “所有 GFlowNet 都偷偷学了 OT”

不对。OT 等价要求非无环 minimum-flow、固定源分布、匹配总质量、图最短路代价和全局最优。

### “\(P_T(x)\) 与 \(P_F(\tau)\) 是同一个分布”

不是。前者是对象边缘概率：

\[
P_T(x)=\sum_{\tau\to x}P_F(\tau).
\]

多路径结构正是必须把二者分开的原因。

### “训练后样本是精确 i.i.d.”

固定模型后，不同随机调用可彼此独立；但每个样本来自近似的 \(P_{T,\theta}\)，未必等于真实 \(R/Z\)。

## 14. 一页式知识地图

```text
未归一化目标 R(x)
        |
        v
状态图 + 多条构造轨迹
        |
        v
轨迹流 F(tau)
   |          |
   v          v
状态/边流     PF 与 PB
   |          |
   +----+-----+
        |
        v
守恒 + 终止流=R
        |
        v
PT(x)=R(x)/Z

如何近似这些约束？
FM（状态） -> DB（边） -> SubTB（子轨迹） -> TB（完整轨迹）
                                      |
                                      +-> GTB：指定内部信用
                                      +-> f-TB：指定 divergence 几何

精确解之外的关键问题
覆盖 / 探索 / 泛化 / PB / replay / 梯度方差 / 长度
        |
        +-> Shen 2023：欠拟合与内部流
        +-> Stable 2026：loss 与 TV 证书
        +-> minimum-flow 2025–2026：最短路
        +-> OT 2026：固定源边缘时的最优 coupling
```

## 15. 本次调研方法与本地资料

本次使用了 [auto-claude-code-research-in-sleep](https://github.com/wanshuiyin/auto-claude-code-research-in-sleep) 的 `research-lit` 工作流作为检索和证据整理框架：

1. 明确问题与截止日期；
2. 从 arXiv、会议/期刊官网、作者教程和官方代码文档收集材料；
3. 把“论文明确声称”“跨论文综合”和“本报告推论”分开；
4. 对核心论文下载全文并检索定义、定理、实验限制；
5. 用仓库的 deterministic verifier 核验 arXiv 标识符存在性。

本地核心 PDF：

- [原始 GFlowNet（2106.04399）](literature/core/2106.04399.pdf)
- [GFlowNet Foundations（2111.09266）](literature/core/2111.09266.pdf)
- [Trajectory Balance（2201.13259）](literature/core/2201.13259.pdf)
- [Shen et al. 2023（2305.07170）](literature/core/2305.07170.pdf)
- [GFlowNet–OT（2606.06272）](literature/core/2606.06272.pdf)

检索与核验痕迹：

- [arXiv 论文核验结果](research/verified_papers.json)
- [核心 PDF 抽取文本](research/text/)
- [原始 arXiv 检索结果](research/raw/)

核验 JSON 的 “verified” 只表示 arXiv ID 和元数据可由 arXiv 回查，不表示论文中的定理或实验结论已被独立证明。报告中的发表状态优先以 JMLR、PMLR、NeurIPS、ICML、ICLR、AAAI、CVF 等官方页面为准；主会与 Workshop 分开记录，没有官方录用页面的 2026 工作按预印本处理。

## 16. 最后应记住的五句话

1. GFlowNet 通过图上的流守恒，把未归一化奖励转成可采样的终止分布。
2. 多路径求和是它相对朴素自回归/MaxEnt RL 表述最关键的结构问题。
3. FM、DB、SubTB、TB 的精确零点可一致，但有限训练行为并不一致。
4. 反向策略和内部 flow 决定信用如何分配；这常常比换一个表面上的 loss 名称更重要。
5. 2026 OT 结果揭示了 minimum-flow 的深层几何意义，但它是条件严格的特例，不是对所有 GFlowNet 的重新定义。
