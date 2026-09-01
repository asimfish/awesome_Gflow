# T01 · 基于流网络的生成模型：非迭代式的多样候选生成

> **Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation**
> 作者：Emmanuel Bengio, Moksh Jain, Maksym Korablyov, Doina Precup, Yoshua Bengio · NeurIPS 2021 · [arXiv](https://arxiv.org/abs/2106.04399) · [代码](https://github.com/bengioe/gflownet)

## 一句话
把「按序列动作构造对象」的生成过程看成 DAG 上的流网络（flow network），用局部流守恒（flow matching）作为训练目标，证明全局最优解对应 $\pi(x) \propto R(x)$ 的采样策略，从而在同一对象有多条构造路径时也能得到正确的终止分布。

## 1. 要解决的问题

目标是学一个策略 $\pi$，使采样某个终止对象 $x$ 的概率满足

$$\pi(x) \approx \frac{R(x)}{Z} = \frac{R(x)}{\sum_{x' \in \mathcal{X}} R(x')}$$

其中 $R(x) > 0$ 是终止状态的奖励（论文强制正值），$Z$ 是配分函数。这与 RL 的期望回报最大化目标不同：后者把概率质量压在单条最高回报的动作序列上，在药物设计里等价于只输出一个分子。

动机来自迭代式黑盒优化：每轮可以对一大批候选调用 oracle（细胞实验、docking 模拟），oracle 本身有噪声且是下游更贵实验的 proxy，因此需要覆盖 $R$ 多个 mode 的一批候选，而不是单个最优点。论文强调这条路线的计算成本对批量大小是线性的，与需要两两比较候选的 batch Bayesian optimization（至少二次）不同。MCMC 能采到 $\pi \propto R$，但在高维、mode 之间被低奖励区域隔开时混合极慢；GFlowNet 把搜索代价摊销（amortize）到训练阶段，推断时一次前向生成。

**核心技术障碍是 DAG 而非树。** 设 $\mathcal{S}$ 为状态集，$\mathcal{X} \subset \mathcal{S}$ 为终止状态集，$C: \mathcal{A}^* \to \mathcal{S}$ 把允许的动作序列映到状态。当 $C$ 是双射时状态空间是树；当 $C$ 非单射（多个动作序列给出同一个分子图）时是 DAG。

论文用 Proposition 1 把这个障碍量化。定义树式伪值函数 $\tilde{V}(s) = \sum_{\vec{b} \in \mathcal{A}^*(s)} R(s + \vec{b})$（$s$ 可达的所有终止状态奖励之和），并令 $\pi(a|s) = \tilde{V}(s+a) / \sum_{b \in \mathcal{A}(s)} \tilde{V}(s+b)$，则：

- (a) $\pi(s) = \sum_{\vec{a}_i : C(\vec{a}_i) = s} \pi(\vec{a}_i)$；
- (b) $C$ 双射时 $\pi(s) = \tilde{V}(s)/\tilde{V}(s_0)$，特别地 $\pi(x) = R(x)/\sum_x R(x)$；
- (c) $C$ 非单射且有 $n(x)$ 条不同动作序列到达 $x$ 时，

$$\pi(x) = \frac{n(x) R(x)}{\sum_{x'} n(x') R(x')}$$

也就是说，把 DAG 当树处理会引入路径计数因子 $n(x)$。论文指出这在组合空间里随轨迹长度指数恶化：更大的分子仅因为有指数级更多的构造顺序就被指数级更常采到。Buesing et al. (2019) 的方法、MaxEnt RL、自回归方法都落在这个陷阱里；Soft Q-Learning 的问题被明确点出——它在 in-flow 里只算轨迹中包含的那一个父节点，导出的是 $P(\tau) \propto R(\tau)$ 而不是 $P(x) \propto R(x)$。

## 2. 核心方法

**流网络构造。** 单一源点是初始状态 $s_0$，入流为 $Z$；每个终止状态 $x$ 是一个汇点，出流固定为 $R(x) > 0$。记 $T(s,a) = s'$ 表示在 $s$ 执行 $a$ 到达 $s'$（环境确定性），$F(s,a)$ 是边 $(s \to s')$ 上的流，$F(s)$ 是穿过 $s$ 的总流。因为 $C$ 非单射，一个节点可以有多个父节点 $|\{(s,a) : T(s,a) = s'\}| \geq 1$（根节点除外）。

**流守恒方程（Eq. 4）。** 约定内部节点 $R(s) = 0$、叶节点 $\mathcal{A}(s) = \emptyset$，则对所有 $s'$ 要求入流等于出流：

$$\sum_{s,a\,:\,T(s,a)=s'} F(s,a) = R(s') + \sum_{a' \in \mathcal{A}(s')} F(s', a')$$

左边是 $s'$ 的全部入流（对所有父边求和，这是与 Soft Q-Learning 的关键分歧点），右边是终止奖励加上全部出边流。论文要求 $F(s,a) > 0$，零流的边通过父状态的动作掩码排除，以便对流取对数。

**Proposition 2（正确性）。** 若流守恒成立，取策略

$$\pi(a|s) = \frac{F(s,a)}{F(s)}$$

其中 $F(s) = R(s) + \sum_{a \in \mathcal{A}(s)} F(s,a)$、终止节点 $F(x) = R(x)$，则

- (a) $\pi(s) = F(s)/F(s_0)$（对根做归纳，$\pi(s_0)=1$）；
- (b) $F(s_0) = \sum_{x \in \mathcal{X}} R(x)$；
- (c) $\pi(x) = R(x) / \sum_{x'} R(x')$。

证明的关键一步是 Eq. 9：$\pi(s') = \sum_{s,a: T(s,a)=s'} \frac{F(s,a)}{F(s)} \cdot \frac{F(s)}{F(s_0)}$ 中 $F(s)$ 消掉，剩下 $s'$ 的入流除以 $F(s_0)$。论文提醒 $\sum_{s \in \mathcal{S}} \pi(s) \neq 1$——非双射情形下不同状态并不互斥，$\pi(s)$ 是访问概率而非 RL 里马尔可夫链的平稳分布。

**训练目标。** 直接把守恒方程的残差平方（Eq. 11）会遇到量纲问题：靠近根的流是 $|\mathcal{X}|$ 量级、靠近叶的流是 $R(x)$ 量级，在高维空间里差指数级倍，神经网络无法同时输出。论文改为在对数尺度上匹配，网络输出 $F^{\log}_\theta(s,a) = \log F(s,a)$，得到 log-scale flow matching 目标（Eq. 12）：

$$\mathcal{L}_{\theta,\epsilon}(\tau) = \sum_{s' \in \tau \neq s_0} \left( \log \Big[\epsilon + \sum_{s,a: T(s,a)=s'} \exp F^{\log}_\theta(s,a)\Big] - \log\Big[\epsilon + R(s') + \sum_{a' \in \mathcal{A}(s')} \exp F^{\log}_\theta(s',a')\Big] \right)^2$$

匹配对数等价于把入流与出流之比推向 1，从而给大流和小流相同的梯度权重。$\epsilon$ 有两个作用：避免对极小流取对数，以及作为超参调节「更在意大流还是小流」——实验中设为接近 $R$ 的最小可能取值（分子任务 $\epsilon = 2.5 \times 10^{-5}$）。$\epsilon$ 不改变全局最小点。

**Proposition 3（off-policy / offline）。** 若训练轨迹来自任意与最优 $\pi$ 同支撑（same support）的探索策略 $P$，且模型族足够丰富（$\exists\theta: F_\theta = F^*$），则期望损失的全局最优满足 $F_{\theta^*} = F^*$、$\mathcal{L}_{\theta^*}(\tau) = 0$，进而 $\pi_{\theta^*}(x) = R(x)/Z$。预测的流不依赖采样策略，只要样本覆盖足够；论文把这类比为异步动态规划（asynchronous DP）。

## 3. 理论结果

- **Proposition 1**：树式伪值方法在非单射环境下产生 $\pi(x) \propto n(x) R(x)$ 的系统偏差。条件：确定性环境、$\tilde{V}$ 定义为后代奖励之和。证明手法是把 DAG 展开成以动作序列为状态的树，同一个 $x$ 在树里重复 $n(x)$ 次。
- **Proposition 2**：流守恒 $\Rightarrow$ 正比采样。条件：DAG（无确定性环）、有限轨迹长度、$R(x) > 0$、$F(s,a) > 0$、策略按出边流归一化。
- **Proposition 3**：off-policy 收敛。条件：行为策略支撑覆盖最优策略的支撑、模型容量充足、损失在流匹配时取到最小值。
- **内部流不唯一（Appendix A.1 末段）**：论文自己给了显式例子——两条轨迹 $s_0 \to s_A \to s_T$ 与 $s_0 \to s_B \to s_T$ 都到 $s_T$（奖励 $r$），则 $F(s_A) = u$、$F(s_B) = r - u$、$u \in [0, r]$ 是一族解。终止分布唯一，内部流有无穷多解。这一句话是后续 T02 内部流刻画、O07/O08 的 minimum-flow 与 OT 选择原则的直接起点。
- **Proposition 4（附录 A.2）**：双射情形下，令 $\mu$ 为均匀策略 $\mu(a|s) = 1/|\mathcal{A}(s)|$、$f(x) = \prod_{t=0}^{n} |\mathcal{A}(s_t)|$、$\hat{R}(x) = R(x) f(s_{n-1})$，则 $Q^\mu(s,a;\hat{R}) = F(s,a;R) f(s)$。即树 MDP 下流等于均匀策略的动作值函数。非单射情形因为流不唯一，论文只给出一个猜想（conjecture）：存在依赖 $n(s)$、$\mathcal{A}(s)$、父节点数 $n_p(s)$ 的 $f$ 使等价关系成立。这条线索被 T37（random policy evaluation）和 T14 系列继续追。

## 4. 实验与证据

**Hypergrid（可精确计算 $Z$）。** $n$ 维、边长 $H$ 的超立方格，动作是把某一维坐标加一，另有 stop 动作；多条动作序列到同一坐标，因此是 DAG。奖励

$$R(x) = R_0 + R_1 \prod_i \mathbb{I}(0.25 < |x_i/H - 0.5|) + R_2 \prod_i \mathbb{I}(0.3 < |x_i/H - 0.5| < 0.4)$$

取 $R_1 = 1/2$、$R_2 = 2$，恰好有 $2^n$ 个 mode 位于角落附近；把 $R_0$ 调小（$10^{-1}, 10^{-2}, 10^{-3}$）人为加大 mode 间隔离。主实验 $n=4, H=8$（16 个 mode）。指标是经验 $L_1$ 误差 $\mathbb{E}[|p(x) - \pi(x)|]$ 与已访问 mode 数。结论：GFlowNet 对 $R_0$ 鲁棒；Metropolis-Hastings MCMC 需要指数级更多样本才达到同一 $L_1$ 水平，且 $R_0$ 越小越难访问每个 mode；PPO 需要把熵正则系数调到 0.5（远高于常用的 $\ll 1$）才能找到全部 mode，仍显著慢于 GFlowNet。SAC 在 $n=4, H=8$ 下最多只找到 16 个 mode 中的 10 个（附录 A.6）。

**分子生成（sEH docking）。** 片段式（fragment-based）设计：72 个预定义 block（考虑对称群复制后每个 stem 有 105 个动作），最多拼 8 个 block，状态空间达 $10^{16}$，每状态 100–2000 个动作。奖励来自在 30 万随机/半随机分子上预训练的 MPNN proxy（test MSE 0.6），预测与 sEH（soluble epoxide hydrolase，4JNC 抑制剂）的结合能，取反并归一化到大致 0–10。训练最多 $10^6$ 个分子，行为策略为 $0.95\,\pi + 0.05\,\text{uniform}$。

关键数字（3 次运行均值±标准差）：

- $10^5$ 样本时 top-10 / top-100 / top-1000 奖励：GFlowNet $8.36 \pm 0.01$ / $8.21 \pm 0.03$ / $7.98 \pm 0.04$；MARS $8.05 \pm 0.12$ / $7.71 \pm 0.09$ / $7.13 \pm 0.19$；PPO $8.06 \pm 0.26$ / $7.87 \pm 0.29$ / $7.52 \pm 0.26$。
- $10^6$ 样本时 GFlowNet $8.45 \pm 0.03$ / $8.34 \pm 0.02$ / $8.17 \pm 0.02$。JT-VAE+BO 在同等计算下只能生成约 $10^3$ 个分子，top-10 仅 6.03。
- top-1000 平均两两 Tanimoto 相似度：GFlowNet $0.44 \pm 0.01$，MARS $0.59 \pm 0.02$，PPO $0.62 \pm 0.03$；随机 agent 为 0.231（奖励很差）。
- mode 数（Bemis-Murcko scaffold，$R > 8$）：GFlowNet $> 1500$，MARS $< 100$。最好一次运行找到 2339 个 score $> 8$ 的独特分子，其中只有 39 个在训练数据集中（数据集最大奖励 10，仅 233 个样本 $> 8$）。
- 用 $R^\beta$（$\beta = 4$）训练时奖励密度整体右移，与 $\pi \propto R^\beta$ 的预期一致。
- 主动学习（multi-round）：1800 次 docking 后 GFlowNet top-10 $8.83 \pm 0.15$、top-100 $7.76 \pm 0.11$；MARS $8.27 \pm 0.20$ / $7.08 \pm 0.13$。PPO 训练不稳定、持续发散，未报告数字。

**证据强弱的诚实之处。** 论文承认在大规模域无法直接验证 $\pi_\theta(x) \propto R(x)$——计算真实 $p_\theta(x)$ 需要对所有到 $x$ 的轨迹求和（many-paths problem），只能给间接证据。Figure 16 把叶节点入流与目标分数做 log-log 回归，斜率 $a = 0.58$、$r = 0.69$：斜率小于 1 说明模型系统性低估高奖励（高奖励样本稀有、访问少）。Figure 18 显示分子任务的 loss 始终不收敛到 0（hypergrid 上会收敛），类似深度 RL 里价值损失不归零。这些是「分布拟合仅近似」的自证，也正是 T07 与 T32 后来系统追问的地方。

## 5. 在 GFlowNet 版图中的位置

- **上游**：Buesing et al. (2019)（MCTS + 值函数做离散近似推断）是被 Proposition 1 直接反驳的对象；Soft Q-Learning / MaxEnt RL（Haarnoja et al., 2017）被指出只匹配 $P(\tau) \propto R(\tau)$；MCMC 方法（MARS、GWG）与 JT-VAE+BO 是实验基线。
- **直接后继**：T02（GFlowNet Foundations）把这里的 flow matching 提升为 Markovian flow 的一般理论，补上后向策略 $P_B$、detailed balance、reward matching 等约束族，并系统处理 T01 附录里那句「内部流有无穷多解」。T03（TB）针对 Eq. 12 的局部性与 bootstrapping 缺陷，改用轨迹级恒等式做信用分配；T05（SubTB）在两者之间插值；T07 则回头诊断这套目标在有限训练下的失效模式。
- **被挑战/被改写**：T14、T15、N003 把 GFlowNet 与最大熵 RL 建立更精确的对应（N003 通过构造修正奖励得到精确等价，而不是 T01 里「Soft Q-Learning 是错的」这种粗粒度区分）；T82（path-dependent）挑战马尔可夫假设本身。
- **内部流选择这条线**：T01 附录的非唯一性例子 → T02 的内部流刻画 → T25/T31 优化 $P_B$ → T36/O07/O08 的 minimum-flow 与最优传输解释。
- **应用谱系起点**：分子片段环境（72 blocks、sEH proxy）成为此后一大批工作的默认基准，A02、A03、A09、N055、N056 等都沿这条线走；multi-round 主动学习框架被 A02、A04 继承。

## 6. 局限与批判

- **bootstrapping 带来的优化困难**：论文自己列为主要局限。Eq. 12 是局部目标，靠自举把终点奖励逐步向上游传，长轨迹上信号衰减严重，这正是 T03 提出 TB 的动机。
- **奖励必须严格为正**、流必须严格为正，零流边只能靠动作掩码排除。这限制了对「无效终止」的建模，A14（LeakGFN）后来专门处理化学环境里大量无效终止状态造成的流泄漏。
- **假设环境确定性且无环**：动作唯一决定下一状态，且不存在确定性环。随机环境要等 T13/T18，非无环要等 T19/T36。
- **分布正确性缺乏直接验证**：大规模域只有 log-log 回归斜率 0.58 这类间接证据，且该斜率本身说明高奖励区被低估。论文没有讨论「loss 低是否等于分布对」，这一空缺到 T32 才被正面处理。
- **内部流完全不受约束**：Appendix A.6 的 Figure 10 显示到达 mode $(6,6)$ 的路径访问分布既不均匀也无特定结构，论文只写「我们的损失不强制任何流分布，均匀流也不一定可取」，把问题留给后人。
- **多样性指标是启发式的**：Tanimoto 相似度阈值 0.7、Bemis-Murcko scaffold 计数都是化学惯例而非分布距离，无法排除「找到很多高奖励分子但分布仍偏」的情形。N080 后来专门研究该用什么指标。
- **超参依赖**：分子实验用了 $\lambda_T = 10$（终止转移损失加权）、reward $\beta = 10$、$T = 8$、$R_{\min} = 0.01$ 等一串技巧，论文承认 $\lambda_T > 1$ 只是「实验发现有帮助」，机制解释停留在「优先修正端点，再由自举向上传播」。

## 7. 对后续研究的启示

- **「按奖励成比例采样」是独立于「最大化奖励」的问题设定**，这一框定本身是论文最持久的贡献。它让 diversity 从事后正则变成目标函数的一部分。
- **多路径结构必须显式处理**。Proposition 1 的 $n(x)$ 因子是一个可复用的检验：任何声称做离散分布匹配的方法，都应该被问「在同一对象有多条构造路径时你的边缘分布是什么」。T15、T38（对称性）、N083 都可看作这条检验的延伸。
- **零损失条件与有限训练性质要分开谈**。T01 只证明了「全局最优 $\Rightarrow$ 正确」，而 Figure 16/18 显示实际训练远离全局最优。T07、T32、T34 全都建立在这个缝隙上。
- **内部流的自由度是可利用的资源，不是缺陷**。同样的终止分布对应一族内部流，选哪一个可以由额外准则（最小总流、最短路、可学习性）决定——O07/O08 把它接到最优传输，T31 用轨迹似然最大化选 $P_B$。
- **摊销 vs 迭代的权衡需要按 reward 地形判断**。论文的卖点是「MCMC 混合慢时摊销更优」，但 N081 后来指出在尖峰、低维、近 Dirac 目标上 MCMC 反而更稳更快约 500 倍。T01 的对比实验默认了宽峰多模态设定，读时应记住这个前提。
