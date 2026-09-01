# T02 · GFlowNet 数学基础

> **GFlowNet Foundations**
> 作者：Yoshua Bengio*, Salem Lahlou*, Tristan Deleu*, Edward J. Hu, Mo Tiwari, Emmanuel Bengio（*同等贡献） · JMLR 24 (2023) 1–76 · [arXiv](https://arxiv.org/abs/2111.09266) · [代码](https://github.com/GFNOrg/torchgfn)（论文本身无实验代码；该库由同组维护，见 N099）

## 一句话
把 GFlowNet 从「DAG 上的流匹配技巧」重建为「pointed DAG 上轨迹测度（trajectory measure）的理论」，给出 Markovian flow 的三种等价参数化、detailed balance 这一新的局部目标，以及条件流（conditional flow）用于自由能、熵、互信息估计的完整机器。

## 问题与动机

T01 只证明了「流守恒 ⟹ 正比采样」和一个 flow matching 损失。留下的问题包括：

- 「流」到底是什么数学对象？T01 里 $F(s)$、$F(s,a)$ 是并列定义的启发式量，没有统一的测度论基础。
- T01 的 flow matching 损失要对某状态的所有父节点求和，动作空间大或状态连续时不可行。有没有不含求和的局部目标？
- T01 附录承认内部流有无穷多解，但没说清这个自由度由什么参数化、能不能被利用。
- 流除了采样，还能算什么？$F(s_0) = Z$ 是配分函数，那内部状态的 $F(s)$ 是不是「从 $s$ 可达的终止奖励之和」？（答案是否）
- 如何把 GFlowNet 用于条件分布、边缘化、熵/互信息估计、多目标 Pareto 采样、随机环境、连续动作？

## 方法核心

### 2.1 从图论到轨迹测度

**pointed DAG（Definition 3）**：有向无环图 $G = (\mathcal{S}, \mathcal{A})$ 存在唯一的源状态 $s_0$ 与汇状态 $s_f$，满足 $\forall s \neq s_0: s_0 < s$ 且 $\forall s \neq s_f: s < s_f$，其中 $<$ 是可达关系诱导的严格偏序。**完整轨迹（complete trajectory）**从 $s_0$ 到 $s_f$；$\mathcal{T}$ 是全部完整轨迹的集合。**终止状态（terminating state）**是 $s_f$ 的父节点，集合记 $\mathcal{S}_f = \mathrm{Par}(s_f)$；边 $s \to s_f$ 称为**终止边**。论文强调「单源单汇」只是数学便利，一般 DAG 与之双射。

关键区分：终止状态 $s \in \mathcal{S}_f$ 是真实对象，$s_f$ 是形式化的汇点。一个终止状态可以同时有其他子节点（Fig. 3 里的 $s_7$），也就是说「继续构造」与「停止」是同一状态上的两个动作。

**轨迹流（Definition 6）**：$F: \mathcal{T} \to \mathbb{R}_+$ 是定义在完整轨迹上的非负函数，它在 $\sigma$-代数 $\Sigma = 2^{\mathcal{T}}$ 上诱导一个测度 $F(A) = \sum_{\tau \in A} F(\tau)$。这是整篇论文的支点：**流首先是轨迹上的测度，状态流和边流是这个测度在特定事件上的取值**：

$$F(s) := F(\{\tau \in \mathcal{T} : s \in \tau\}), \qquad F(s \to s') := F(\{\tau \in \mathcal{T} : s \to s' \in \tau\})$$

于此，Proposition 8 的两个等式（Eq. 8/9）就是把「过 $s$ 的轨迹集」按子节点或父节点做无交分解，不再需要「入流等于出流」这种物理直觉：

$$F(s) = \sum_{s' \in \mathrm{Child}(s)} F(s \to s'), \qquad F(s') = \sum_{s \in \mathrm{Par}(s')} F(s \to s')$$

总流 $Z := F(\mathcal{T})$，Proposition 10 给出 $F(s_0) = F(s_f) = Z$（因为每条完整轨迹都过 $s_0$ 和 $s_f$）。归一化后得到流概率 $P(A) = F(A)/Z$，并定义

$$P_F(s' \mid s) := \frac{F(s \to s')}{F(s)}, \qquad P_B(s \mid s') := \frac{F(s \to s')}{F(s')}$$

以及**终止状态概率**（Definition 13）$P_T(s) := P(s \to s_f) = F(s \to s_f)/Z$。Proposition 14 证明 $P_T$ 是 $\mathcal{S}_f$ 上的良定分布（$\sum_{s \in \mathcal{S}_f} P_T(s) = 1$），而 $P(s)$ 不是——$\sum_{s \in \mathcal{S}} P(s) \neq 1$，且 $P(s_0) = 1$。T01 里那句「状态不互斥」在这里有了准确表述：$P(s)$ 是「轨迹经过 $s$」这个事件的概率。

### 2.2 Markovian flow 与三种参数化

一般的流需要指定 $|\mathcal{T}|$ 个数（对边数指数级）。**Markovian flow（Definition 15）**要求流概率无记忆：

$$P(s \to s' \mid \tau) = P(s \to s' \mid s) = P_F(s' \mid s)$$

对任何从 $s_0$ 到 $s$ 的部分轨迹 $\tau$ 成立。Proposition 16 给出三条等价陈述：$F$ 是 Markovian $\iff$ 存在唯一的前向概率函数使 $P(\tau) = \prod_{t=1}^{n+1} \hat{P}_F(s_t \mid s_{t-1})$ $\iff$ 存在唯一的后向概率函数使 $P(\tau) = \prod_{t=1}^{n+1} \hat{P}_B(s_{t-1} \mid s_t)$，且这两个函数正是 $P_F$、$P_B$。Corollary 17：从 $s_0$ 反复按 $P_F$ 采样直到 $s_f$，终止在 $s$ 的概率就是 $P_T(s)$。

**Proposition 18（三种完整参数化）**：Markovian flow 由以下任一组合唯一确定：
1. 总流 $\hat{Z}$ + 全部边的前向转移 $\hat{P}_F$；
2. 总流 $\hat{Z}$ + 全部边的后向转移 $\hat{P}_B$；
3. 全部终止流 $\hat{F}(s \to s_f)$ + 非终止边的后向转移 $\hat{P}_B$。

第 3 种是理解 §2.6 的钥匙：给定要匹配的终止流（即奖励），仍需指定非终止边上的 $P_B$ 才能定死整个流。

### 2.3 两个局部约束

**flow matching（Proposition 19）**：非负函数 $\hat{F}$ 对应一个流当且仅当

$$\forall s' > s_0:\ \hat{F}(s') = \sum_{s \in \mathrm{Par}(s')} \hat{F}(s \to s'), \qquad \forall s' < s_f:\ \hat{F}(s') = \sum_{s'' \in \mathrm{Child}(s')} \hat{F}(s' \to s'')$$

此时唯一的 Markovian flow 满足 $F(\tau) = \prod_{t=1}^{n+1} \hat{F}(s_{t-1} \to s_t) \big/ \prod_{t=1}^{n} \hat{F}(s_t)$（Eq. 23）。

**detailed balance（Proposition 21）**：状态流 $\hat{F}$ 与前后向转移 $\hat{P}_F, \hat{P}_B$ 三者共同对应一个流当且仅当

$$\forall s \to s' \in \mathcal{A}:\quad \hat{F}(s)\,\hat{P}_F(s' \mid s) = \hat{F}(s')\,\hat{P}_B(s \mid s')$$

其中 $\hat{F}(s)$ 是状态流估计、$\hat{P}_F(s'|s)$ 是前向策略、$\hat{P}_B(s|s')$ 是后向策略。这是本文命名的新目标，与 MCMC 的 detailed balance 同形。它的优势是**每条边一个方程，不含对父节点或子节点的求和**，因此适用于大动作空间和连续状态。对应损失（Example 5）：

$$\mathcal{L}_{DB}(\hat{F}, \hat{P}_F, \hat{P}_B, s \to s') = \left( \log \frac{\delta + \hat{F}(s)\hat{P}_F(s' \mid s)}{\delta + \hat{F}(s')\hat{P}_B(s \mid s')} \right)^2 \quad (s' \neq s_f), \qquad \left( \log \frac{\delta + \hat{F}(s)\hat{P}_F(s' \mid s)}{\delta + R(s)} \right)^2 \quad (s' = s_f)$$

论文顺带指出（Eq. 31 附近）：$\hat{P}_B$ 不受约束时看似可以用 $\hat{P}_B(s|s') = \hat{P}_F(s'|s)\hat{F}(s)/\hat{F}(s')$ 平凡地满足 DB，但 $\sum_{s \in \mathrm{Par}(s')} \hat{P}_B(s|s') = 1$ 的归一化要求会强制 $\sum_{s \in \mathrm{Par}(s')} \hat{P}_F(s'|s)\hat{F}(s) = \hat{F}(s')$，即流仍须与前向转移一致。

### 2.4 GFlowNet 的形式定义

**flow parametrization（Definition 24）**是三元组 $(\mathcal{O}, \Pi, H)$：$\mathcal{O}$ 是配置空间，$\Pi: \mathcal{O} \to \Delta(\mathcal{T})$ 把配置映到轨迹分布，$H: \mathcal{F}_{\mathrm{Markov}}(G,R) \to \mathcal{O}$ 是单射，且 $\Pi(H(F))$ 等于 $F$ 诱导的概率测度。**GFlowNet（Definition 25）**是五元组 $(G, R, \mathcal{O}, \Pi, H)$。**flow-matching loss（Definition 26）**是任意 $L: \mathcal{O} \to \mathbb{R}_+$ 满足 $L(o) = 0 \iff o \in H(\mathcal{F}_{\mathrm{Markov}}(G,R))$，并按可分解性分为 edge-decomposable、state-decomposable、trajectory-decomposable 三类。

这个抽象让「找正确流」变成「在 $\mathcal{O}$ 上最小化 $L$」，并把 FM（state-decomposable）、DB（edge-decomposable）、TB（trajectory-decomposable，Example 6 引用 Malkin et al. 2022，即 T03）纳入同一框架：

$$\mathcal{L}_{TB}(\hat{Z}, \hat{P}_F, \hat{P}_B, \tau) = \left( \log \frac{\hat{Z} \prod_{t=1}^{n+1} \hat{P}_F(s_t \mid s_{t-1})}{R(s_n) \prod_{t=1}^{n} \hat{P}_B(s_{t-1} \mid s_t)} \right)^2$$

### 2.5 条件流与自由能

**自由能（Definition 27）**：给定能量 $E$，状态 $s$ 的自由能定义为 $e^{-\mathcal{F}(s)} := \sum_{s' \geq s} e^{-E(s')}$。

Fig. 5 给了反例说明 $F(s)$ **不**做这种边缘化：图中 $F(s_2) = 4$，而从 $s_2$ 可达的终止流之和为 6，差额来自路径 $(s_0, s_1, s_5)$——它贡献了 $F(s_5 \to s_f)$ 但不经过 $s_2$（$s_1$ 与 $s_2$ 之间无序关系）。

修正办法是**state-conditional flow network（Definition 30）**：对每个 $s$，取子图 $G_s$（包含所有 $s' \geq s$ 的状态），要求 $F_s(s' \to s_f) = F(s' \to s_f)$。Proposition 31 证明这样的条件流存在。此时（Proposition 32）

$$F_s(s_0 \mid s) = F_s(s) = \sum_{s' \geq s} F(s' \to s_f) = \exp(-\mathcal{F}(s))$$

即条件流网络的「初始流」就是自由能。Corollary 33 进一步给出 $P_T(s' \mid s) = \mathbb{1}_{s' \geq s}\, e^{-E(s') + \mathcal{F}(s)}$。

**熵与互信息（§4.7）**：定义熵奖励 $R'(s) = -R(s) \log R(s)$（需 $R(s) < 1$ 保证正值）。训练第二个流 $F'$ 匹配 $R'$，则（Proposition 35）

$$H[S] = \frac{F'(s_0)}{F(s_0)} + \log F(s_0)$$

条件化后得条件熵 $H[S \mid x] = F'(s_0|x)/F(s_0|x) + \log F(s_0|x)$，两者相减得互信息 $\mathrm{MI}(S; X)$（Proposition 36，Eq. 57）。

## 理论结果

- **Proposition 8 / 10 / 14**：状态流与边流的分解、$F(s_0) = F(s_f) = Z$、$P_T$ 是良定分布。条件：pointed DAG、有限状态。
- **Proposition 16 / 18**：Markovian flow 的等价刻画与三种唯一参数化。这把「学一个流」从指数级参数降到边数级参数。
- **Proposition 19 / 21**：flow matching 与 detailed balance 分别是「非负函数构成流」的充要条件。DB 的价值在于消掉求和。
- **Proposition 23（流等价类）**：定义两个流等价当且仅当它们在所有边流上相同（Definition 22）。则（i）两个等价的 Markovian flow 必相等；（ii）任意流 $F'$ 的等价类中存在唯一的 Markovian flow。Fig. 4 给出具体数值：同一个五状态 DAG 上四个流 $F_1, \ldots, F_4$，$F_1 \sim F_2$、$F_3 \sim F_4$，其中 $F_2, F_4$ 是 Markovian、$F_1, F_3$ 不是，而四者在终止流上完全一致。这个命题是「只研究 Markovian flow 不损失一般性」的正式理由。
- **§2.6：后向转移可自由选择**。终止流不完全确定流；不同做事顺序的偏好由 $P_B$ 表达（$P_B(s \mid s_f)$ 除外，它由终止流与 $Z$ 决定）。论文明确列出三种用法：给所有父节点等权、偏好更短路径（可在状态里记最短路长度）、让学习器自己找一个使 $P_F$ 或 $F$ 更易学的 $P_B$。这三条分别预示了后来的 uniform-$P_B$ 惯例、O07 的最短路方向、T31/T25 的 $P_B$ 优化。
- **Proposition 39/40/41（附录 A，直接信用分配）**：在流已匹配的极限下，$\frac{d \log F(s')}{d \log F(s)} = P(s \mid s')$、$\frac{d \log F(s')}{d \log F(s \to s')} = P_B(s \mid s')$。据此构造两个渐近无偏的总导数估计 $G_1$（沿采样轨迹的边）、$G_2$（对父节点按 $P_B$ 加权），以及任意凸组合 $G = \lambda G_1 + (1-\lambda) G_2$。论文自己评价这「非常接近 policy gradient」，但只在 on-policy 且流已匹配时无偏，off-policy 需要重要性权重。这条线后来变成 T16（policy gradient 训练 GFlowNet）。
- **Proposition 52/53/55（附录 D，与最大化奖励的关系）**：定义 $V_{P_\pi}(s) := \mathbb{E}_{P_\pi}[R(S) \mid S \geq s]$。在 $P_T$ 下 $V_{P_T}(s) = \sum_{s' \geq s} R(s')^2 \big/ \sum_{s' \geq s} R(s')$，可以通过训练第二个匹配 $R^2$ 的流得到：$V_{P_T}(s) = F'(s|s)/F(s|s)$。Proposition 53 证明对 $V_{P_\pi}$ 贪心的策略 $\bar\pi$ 满足 $V_{P_{\bar\pi}}(s) \geq V_{P_\pi}(s)$（中间奖励为 0、$\gamma = 1$ 时的 policy improvement 类比），Corollary 54 由此得到存在最优贪心策略。也就是说，训练好的 GFlowNet 可以「读出」一个奖励最大化策略。
- **Proposition 62（附录 F，事后指定奖励）**：outcome-conditioned GFlowNet 按结果 $y = f(s)$ 条件化训练完成后，对任意事后给定的 $R(s) = r(f(s))$，无需重训即可得 $F_{r \circ f}(A) = \sum_y r(y) F(A \mid y)$，策略为 $\pi_{r \circ f}(a|s) = \sum_y r(y) F((s,a)|y) \big/ \sum_y r(y) F(s|y)$。代价是运行时要对结果空间求和（可 Monte-Carlo 近似），存在计算量与精度的权衡。Definition 63/64 给出 Pareto 加性奖励 $R_\omega(s) = \sum_i \omega_i f_i(s)$ 与乘性奖励 $R_\omega(s) = e^{-\sum_i \omega_i e_i(s)}$，训练以凸权重 $\omega$ 为条件的 GFlowNet 后，先采 $\omega$ 再采轨迹即可从 Pareto 前沿取样。
- **§7.2 与 MaxEnt RL 的定量分歧**：MaxEnt RL 学到 $P_T(s) \propto n(s) R(s)$（$n(s)$ 为到 $s$ 的路径数），仅当 DAG 去掉 $s_f$ 后是以 $s_0$ 为根的树时两者一致。论文另给一个等价表述：训练 $P_T(s) \propto R(s)$ 等价于最大化 $r(s,a) = \log R(s,a) - \log d^\pi(s,a)$，但 $d^\pi$（状态占用度）一般不可计算。

## 实验与证据

**这是一篇纯理论论文，没有自己的实验。** 全部实证支撑来自引用：T01 的分子与 hypergrid 实验、Deleu et al. 2022（A17，DAG-GFlowNet 逼近贝叶斯网络结构后验）、Jain et al. 2022/2023（A02/A03）、Zhang et al. 2022（联合训练 EBM 与 GFlowNet）、Malik et al. 2023（用集合 GFlowNet 做主动学习批量选择，并验证互信息奖励可用神经网络近似）、Hu et al. 2023（T10，GFlowNet-EM）、Pan et al. 2023（T06，Forward-Looking GFlowNet 利用能量模块化，报告比常规损失更好的分布近似）。

证据强度需要分层看待。detailed balance 的正确性有完整证明，且已被 A17 等工作大规模验证；而熵/互信息估计（需要训练第二个流并要求 $R < 1$）、模块化能量分解、GFlowNets-in-GFlowNets、Pareto 事后重加权这几节只有构造和命题，论文在 §8 明确写「本文许多数学表述仍需实证验证才能确认其有用性」。读者应把后半篇当作研究纲领而非已验证结论。

版本差异值得注意：arXiv v5（2026-01）比 JMLR 2023 版增补了 Appendix B 对条件 GFlowNet 的形式化（Definition 42–43、Remark 44）与 Example 9，其中 Remark 44 明确指出「摊销参数化下各条件之间的优化是耦合的，有限容量时 $L = 0$ 一般不可达，实际最优是各条件间的折中」。这是对条件 GFlowNet 一个诚实的限定，早期版本没有。

## 与谁对话

- **直接建立在 T01 之上**，把 T01 的 Proposition 2/3 重述为测度论结果，把 T01 附录里「内部流无穷多解」的一句话展开为流等价类（Proposition 23）+ $P_B$ 自由度（§2.6）两个正式结果。
- **detailed balance 成为第二条主线**：T03 的 TB 在本文里被收进 Example 6；T05（SubTB）用「任意子轨迹上的 balance」把 DB 与 TB 连成一族；T47（evaluation balance）继续沿用这套 balance 语言。
- **条件 GFlowNet 与自由能**是 A03（多目标偏好条件）、A04（多保真度）、T22（预训练/微调）、T28（温度条件）、A05/N017（分子构象条件化）的理论入口；Proposition 62 的「事后指定奖励」正是 T22 outcome-conditioned 预训练的原型。
- **$P_B$ 自由选择这条线**：T25（悲观后向策略）、T31（轨迹似然最大化优化 $P_B$）、T36/O07/O08（minimum flow 与最优传输选择内部流）全部以 §2.6 为起点。
- **训练分布 $\pi_T$ 的建议（§3.3.3）**：论文写道「$\pi_T$ 可以是第二个 GFlowNet 的策略，它主要匹配一个在主 GFlowNet 损失大处取高值的奖励函数」。这句话是 N024（adaptive teachers）、T35（sibling augmented）、T48（loss-guided auxiliary agents）、T50（ACE / divergent TB）的共同祖先。
- **被扩展/被修正**：T12（Lahlou et al. 2023）把理论推广到一般测度空间（本文 §6.2 已加注引用）；T13/T18 处理随机环境（本文附录 C 的 Proposition 49 只给出「任意策略都能产生 Markovian flow，但可能无法完美达到目标终止流」这一负面结论）；T19/T36 处理非无环情形（本文 §3.3.1 只提供了加时间戳这一朴素办法）；T08/T09 建立与变分推断的联系（本文 §1.3 已引用 Malkin et al. 2023）；T82 挑战 Markovian flow 这个核心假设本身。
- **与 RL 的接口**：附录 A 的 policy-gradient 式估计 → T16；附录 D 的 $V_{P_T}$ 与贪心策略 → T26（QGFN 用动作价值调节 greediness）、T37；§7.2 的 $n(s)$ 分歧 → T14/T15/N003 的精确等价研究。

## 局限与批判

- **零损失理论，不涉及有限训练**。全篇的结论形式都是「$L(o) = 0 \iff$ 流正确」。损失小但不为零时终止分布偏多少，本文完全没有讨论。T07、T32、N001、T51 都是补这个洞。
- **随机环境的结论是负面的且被搁置**。Proposition 49 说明在随机环境中「任意策略都可产生 Markovian flow，且可能无法精确匹配目标终止流」，然后只写「有足够训练时间和容量时前后向转移可以变得相容」。真正的处理留给 T13/T18。
- **连续情形只有方案草图**。§6 讨论用高斯、混合分布、自回归/归一化流、扩散式多步重采样来参数化连续条件密度，但没有可测性、密度存在性的严格论证；论文自己加注说「本文审稿期间 Lahlou et al. (2023) 把理论推广到了更一般的状态空间」。T12 明确指出过 CFlowNets（N015）在可达性与密度参数化上的假设不严谨，本文 §6 处于同一层次的非严格状态。
- **熵估计的可用性存疑**。要求 $R(s) < 1$ 对所有终止状态成立（否则熵奖励为负），这在实践中需要重新缩放奖励；且需要训练第二个流，误差如何复合没有分析。
- **非无环只有加时间戳这一招**。$\mathcal{S}' = \mathcal{S} \times \mathbb{N}$ 会让状态空间随最大轨迹长度膨胀，且无法处理「同一状态需要被访问多次」这类真正的循环语义。T19/T36 的 expected visit flow 才是正解。
- **模块化能量分解（§5.4）与 GFlowNets-in-GFlowNets（§6.2）纯属构想**。前者把因子图作为 GFlowNet 的生成对象、能量按因子求和，后者用内层 GFlowNet 表示外层的一条边流。两节都没有算法细节、复杂度分析或实验，且与注意力机制的类比停留在修辞层面。
- **一个概念负担**：论文对同一个符号 $F$ 反复重载（轨迹流、测度、状态流、边流、条件流 $F_s$），并把 Proposition 编号与 Theorem 交替引用（正文里出现 "Theorem 18" 指向 Proposition 18）。这在阅读长证明时容易出错。

## 对后续研究的启示

- **「流 = 轨迹测度」这个视角比「流 = 守恒量」更能生长**。一旦把 $F(s)$ 定义为事件概率，条件流、自由能、熵、互信息都是同一台机器的不同读数；非无环推广（用 expected visit flow）和连续推广（换成一般测度空间）也都在这个视角下自然。
- **约束的选择就是信用分配尺度的选择**。FM（状态级）、DB（边级）、TB（轨迹级）零点相同但优化行为不同，这一观察直接催生了 T05 的 $\lambda$ 插值、T34 的损失函数设计、T49 的 $f$-divergence 对应。本文提供了统一的语言（decomposability），但没有比较它们的有限样本性质。
- **$P_B$ 是设计变量，不是无关的辅助量**。§2.6 是全篇最有生产力的一节半页。任何以 uniform $P_B$ 为默认设定的工作，都应先问「这个选择对可学习性和内部流意味着什么」。
- **条件化是复用的通用手段**。Proposition 62 给出的「训练一次、事后定义奖励」是 foundation sampler 思路的最早形式化；T55（组合预训练 GFlowNet 做多目标）、T28（温度条件）都在这条路上，但都还没解决 Remark 44 指出的「有限容量下各条件相互折中」。
- **理论纲领需要配套的证伪机制**。本文最大的方法论教训是：一篇 76 页的论文里，有严格证明的部分（DB、流等价类、参数化唯一性）与只有构造的部分（模块化能量、GFlowNets-in-GFlowNets、熵估计）在文体上没有区分。后来者引用时应逐项核对是「已证 + 已验证」「已证 + 未验证」还是「仅构造」。
