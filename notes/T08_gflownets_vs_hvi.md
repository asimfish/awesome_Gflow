# T08 · GFlowNet 与变分推断的梯度等价

> **GFlowNets and Variational Inference**
> 作者：Nikolay Malkin*, Salem Lahlou*, Tristan Deleu, Xu Ji, Edward Hu, Katie Everett, Dinghuai Zhang, Yoshua Bengio（*共同一作；Mila–Université de Montréal / Google Research） · ICLR 2023 · [arXiv](https://arxiv.org/abs/2210.00580) · [代码](https://github.com/GFNOrg/GFN_vs_HVI)

## 一句话

证明 on-policy 的 Trajectory Balance（TB）训练在期望梯度意义上等价于带最优 baseline 的层级变分推断（hierarchical variational inference, HVI），而两者的真正分野在于 GFlowNet 可以不借助重要性采样（importance sampling, IS）做 off-policy 训练——这正是它在多峰目标分布上更能发现模式的原因。

## 问题与动机

GFlowNet（T01/T02）被提出时定位为一种强化学习式算法：在 pointed DAG $G=(\mathcal{S},\mathcal{A})$ 上从初始态 $s_0$ 出发顺序执行动作，使终态边缘分布

$$P_F^\top(x) = \sum_{\tau \in \mathcal{T}: x_\tau = x} P_F(\tau) \;\propto\; R(x)$$

（$\mathcal{T}$ 为完整轨迹集合，$x_\tau$ 为轨迹终态，$R:\mathcal{X}\to\mathbb{R}^+$ 为奖励即非归一化目标密度）。而 HVI 走的是另一条路：把样本空间建成定长序列 $(z_1,\dots,z_n)$ 的马尔可夫链 $q(z_1)q(z_2\mid z_1)\cdots q(z_n\mid z_{n-1})$，最小化边缘 $q(z_n)$ 与目标之间的散度。两个社区各自发展，关系不明。本文回答三个问题：

1. GFlowNet 的 TB / DB / SubTB 目标与 HVI 的 KL 类目标在数学上是什么关系？
2. 若二者在某种意义上等价，剩余的差异是什么、来自哪里？
3. 这些差异在离散多峰分布（分子生成、贝叶斯结构学习）上会产生什么实验后果？

动机是双向的：为用过 GFlowNet 的任务补上缺失的 HVI baseline，也为 VI 社区引入「无 IS 的 off-policy 训练」这一新能力。

## 方法核心

**建立对象层面的双射。** 任意 pointed DAG 可通过插入哑状态规范化为分层（graded）DAG（§A，构造是幂等的），因此不失一般性只讨论分层情形。此时：HVM（hierarchical variational model）的第 $i$ 层随机变量 $z_i$ ↔ 轨迹中第 $i$ 步状态 $s_i$；HVM 条件分布 $q(z_{i+1}\mid z_i)$ ↔ 前向策略 $P_F(s_{i+1}\mid s_i)$；HVM 边缘 $q(z_n)$ ↔ 终态分布 $P_F^\top$。反向策略 $P_B$ 的意义：它把 $\mathcal{X}$ 上的非归一化目标 $R$ 抬升为轨迹空间上的目标分布（式 (4)）

$$P_B(\tau) \propto R(x_\tau)\, P_B(\tau \mid x_\tau), \qquad \hat{Z} = \sum_{x\in\mathcal{X}} R(x),$$

$\hat Z$ 是未知真实配分函数；$P_B(\tau)$ 的终态边缘恰为 $R(x)/\hat Z$，所以「$P_F = P_B$（作为轨迹分布）」就蕴含采样正确。

**两类目标函数。** TB 目标（式 (3)，源自 T03）：

$$\mathcal{L}_{\mathrm{TB}}(\tau; P_F, P_B, Z) = \left( \log \frac{Z\, P_F(\tau)}{R(x_\tau)\, P_B(\tau\mid x_\tau)} \right)^2,$$

$Z$ 为可学习标量（以 $\log Z$ 参数化）；对所有 $\tau$ 取零当且仅当 $P_F^\top \propto R$ 且 $Z=\hat Z$。HVI 目标（式 (5)）是轨迹空间上的 $f$-散度

$$\mathcal{L}_{\mathrm{HVI},f}(P_F,P_B) = D_f(P_B \Vert P_F) = \mathbb{E}_{\tau\sim P_F}\!\left[ f\!\left( \frac{P_B(\tau)}{P_F(\tau)} \right) \right],$$

常用取法为 forward KL（$f: t\mapsto t\log t$，得 $D_{\mathrm{KL}}(P_B\Vert P_F)$）与 reverse KL（$f: t\mapsto -\log t$，得 $D_{\mathrm{KL}}(P_F\Vert P_B)$）。数据处理不等式（式 (6)，要求 $f$ 凸）保证轨迹级散度是终态级散度的上界：$D_f(R/\hat Z \,\Vert\, P_F^\top) \le D_f(P_B \Vert P_F)$，所以轨迹级目标是合理的替代损失。

**算法谱系（Table 1）。** 训 $P_F$（采样器）与训 $P_B$（后验）可用不同散度，只要都在 $P_F = P_B$ 处取零：

- REVERSE KL：两者都用 $D_{\mathrm{KL}}(P_F\Vert P_B)$；
- FORWARD KL：两者都用 $D_{\mathrm{KL}}(P_B\Vert P_F)$；
- WAKE-SLEEP（WS）：$P_F$ 用 $D_{\mathrm{KL}}(P_B\Vert P_F)$、$P_B$ 用 $D_{\mathrm{KL}}(P_F\Vert P_B)$；
- REVERSE WAKE-SLEEP：反过来；
- on-policy TB：$P_F$ 侧等价于 $D_{\mathrm{KL}}(P_F\Vert P_B)$，$P_B$ 侧见式 (9)。

**主定理（Proposition 1）。** 记 $\theta,\phi$ 分别为 $P_F, P_B$ 的参数，则

$$\nabla_\phi D_{\mathrm{KL}}(P_B \Vert P_F) = \tfrac{1}{2}\, \mathbb{E}_{\tau\sim P_B}\big[\nabla_\phi \mathcal{L}_{\mathrm{TB}}(\tau)\big] \quad (7), \qquad \nabla_\theta D_{\mathrm{KL}}(P_F \Vert P_B) = \tfrac{1}{2}\, \mathbb{E}_{\tau\sim P_F}\big[\nabla_\theta \mathcal{L}_{\mathrm{TB}}(\tau)\big] \quad (8).$$

式 (8) 说明 on-policy TB 对 $P_F$ 的期望梯度就是 reverse KL 的梯度（且与 $Z$ 的当前估计值无关——证明的直接推论）。式 (7) 的期望取自 $P_B$ 而非 $P_F$，直接优化需要 IS；若用 (7) 训 $P_B$、用 (8) 训 $P_F$，恰好复现 REVERSE WAKE-SLEEP 的期望梯度。而真正 on-policy 的 TB 对 $P_B$ 的梯度对应一个更奇特的目标（式 (9)）：

$$\mathbb{E}_{\tau\sim P_F}[\nabla_\phi \mathcal{L}_{\mathrm{TB}}(\tau)] = \nabla_\phi\Big[ D_{\log^2}(P_B \Vert P_F) + 2(\log Z - \log \hat{Z})\, D_{\mathrm{KL}}(P_F \Vert P_B) \Big],$$

其中 $D_{\log^2}$ 由 $f(x)=(\log x)^2$ 定义，是伪 $f$-散度（大 $x$ 处非凸）。

**$\log Z$ 即最优 baseline（§2.3）。** Reverse KL 的 REINFORCE（score function）梯度估计为 $\Delta(\tau) = \nabla_\theta \log P_F(\tau;\theta)\, c(\tau)$，$c(\tau) = \log\frac{P_F(\tau)}{R(x_\tau) P_B(\tau\mid x_\tau)}$（式 (10)），方差大；标准做法是从 $c(\tau)$ 中减 baseline $b \approx \mathbb{E}_{\tau\sim P_F}[c(\tau)]$（最优 baseline 的精确形式带梯度范数加权）。批内局部 baseline 为 $b_{\mathrm{local}} = \frac1B \sum_i c(\tau_i)$（式 (11)）；全局 baseline 用滑动平均维护：

$$b_{\mathrm{global}} \leftarrow (1-\eta)\, b_{\mathrm{global}} + \eta\, b_{\mathrm{local}} \quad (12),$$

这与 TB 中 $\log Z$ 以学习率 $\eta/2$ 做 SGD 的更新式完全一致（TB 对 $\log Z$ 是二次的）。于是 on-policy TB = reverse KL + 自动学得的全局控制变量（control variate）；$P_B$ 固定时二者只差方差缩减方式。

**关键差异：off-policy 机制。** HVI 目标是期望形式，行为策略（behavior policy）$\pi \ne P_F$ 时必须做 IS 加权 $P_B(\tau_i)/\pi(\tau_i)$，方差随 $\pi$ 与 $P_B$ 的差异增大。TB 是逐轨迹的平方损失，任何轨迹来源只改变多目标问题的标量化权重（scalarization weights），不引入 IS 方差——这是 GFlowNet 从 RL 继承的结构性优势。

## 理论结果

- **Proposition 1**（§2.3，证明在 §B）：见上。成立条件：分层 DAG（一般 DAG 先规范化）、$Z$ 为不依赖轨迹与参数的标量。推论：on-policy TB 对 $P_F$ 的期望梯度与 $\log Z$ 估计值无关。
- **式 (9)**：on-policy TB 对 $P_B$ 的梯度 = $D_{\log^2}$ 伪散度梯度 + 系数为 $2(\log Z - \log\hat Z)$ 的 reverse KL 梯度修正项；$\log Z$ 估准时修正项消失。
- **Lemma 1 + Proposition 2**（§C）：把 nested variational inference（NVI，Zimmermann et al. 2021）自然推广到子轨迹。在分层 DAG 上选 $K+1$ 个 junction layer（含首层 $m_0=0$ 与末层 $m_K=L$），每个非终止 junction layer 配状态流函数 $F_k$，定义子轨迹上的一对分布 $\hat p_k(\tau_k) \propto F_k(s_{m_k}) P_F(\tau_k)$ 与 $\check p_k(\tau_k) \propto F_{k+1}(s_{m_{k+1}}) P_B(\tau_k \mid s_{m_{k+1}})$（式 (20)(21)，$F_K$ 固定为 $R$）。Lemma 1：所有 $\hat p_k = \check p_k$ 时 $P_F^\top \propto R$。Proposition 2：SubTB（式 (17)）与 SubNVI 目标 $\sum_k D_f(\check p_k \Vert \hat p_k)$（式 (26)）在期望梯度上等价（式 (27)(28)）——$P_B$ 侧对应 forward KL、$P_F$ 侧对应 reverse KL，各差一个因子 2。$K=1$ 退化为 Prop. 1（TB↔HVI）；$K=L$（全部层都是 junction layer）给出 DB↔NVI。
- 相关工作中指出：Richter et al. 2020 的 log-variance loss 与 batch 最优 $\log Z$ 下的 on-policy TB 在 $P_F$ 期望梯度上等价。

## 实验与证据

三个离散域 + 一个连续附录实验，统一支撑两个观察：**Observation 1**——on-policy 下 TB 与 HVI 行为接近，且 TB 在 mode-seeking 的 reverse KL 与 mean-seeking 的 forward KL 之间取得更好折中；**Observation 2**——需要探索时 off-policy TB 全面胜出，因为无 IS 方差。

**Hypergrid（128×128，$R_0=10^{-3}$，4 个角落模式）**：

- 奖励为式 (31)：$R(s^\top) = R_0 + 0.5\prod_d \mathbb{1}[\,|s_d/(H{-}1)-0.5| \in (0.25,0.5]\,] + 2\prod_d \mathbb{1}[\,|s_d/(H{-}1)-0.5| \in (0.3,0.4)\,]$，$R_0$ 越小探索越难。
- 评估：状态空间可枚举时用流传播动态规划精确算 $P_F^\top$（式 (32)(33)，复杂度线性于 $|\mathcal{S}|+|\mathcal{A}|$），再算与目标的 JSD（对称，避免偏袒任一 KL 方向；式 (34)(35)）。
- 结果（Fig. 1）：FORWARD KL 与 WS 找到 4 个模式更快但精度差（mean-seeking 导致模式处出现纹理伪影）；REVERSE KL 系 mode-seeking，可能漏模式；TB 收敛到最低 JSD 且四模式全部精确建模。off-policy 行为策略（logits 减 $\epsilon$，余弦退火到 0）抑制提前终止、加速远端模式发现，所有方法都获益但 TB 获益最大。
- HVI 对学习率明显更敏感；作者做了两阶段贝叶斯超参搜索（先以 200K 轨迹处 JSD 为目标搜索、再以 JSD 曲线下面积选定）保证比较公平，并发现全局 baseline 普遍优于局部 baseline（Fig. D.1）。较易的 64×64 / 8×8×8×8（$R_0=0.1$）网格上 mode-seeking 一侧反而与 TB 持平（Fig. D.2）。

**分子生成（T01 的 block-by-block 环境，$R(x)=f(x)^\beta$，$f$ 为预训练结合能 proxy）**：

- 指标：持有集分子上 $\log P_F^\top(x)$（动态规划可算）与 $\log R(x)$ 的 Pearson 相关，完美采样器应为 1。
- $P_B$ 固定为均匀，故 on-policy TB 与 on-policy REVERSE KL 期望梯度相同，实测性能几乎重合（Obs. 1）；off-policy REVERSE KL 因 IS 方差反而比其 on-policy 版更差（小 $\beta$ 即高熵目标时尤甚）；off-policy TB 在 $\beta \in \{4,8,10,16\}$ 与 4 个学习率的全部 64 组设置下最好且方差最低（Fig. 2）。off-policy FORWARD KL/WS 的所有超参组合相关系数都不超过 0.1。

**贝叶斯结构学习（A17 的设定：DAG 逐边生成、线性高斯 BGe score、均匀先验、$P_B$ 固定均匀、$d\le5$ 可精确枚举后验）**：

- Table 2 的 JSD：$d=5$ 时 off-policy TB 为 $5.44\pm2.47\times10^{-4}$、Modified DB 为 $4.65\pm1.08\times10^{-4}$；on-policy TB 为 $0.277\pm0.040$、on-policy REVERSE KL 为 $0.306\pm0.042$（两者接近，印证 Obs. 1）；off-policy REVERSE KL 恶化到 $0.656\pm0.009$（IS 失效，印证 Obs. 2）。$d=3$ 时所有方法都准。
- 边缘概率 $P(X_i\to X_j\mid\mathcal{D})$ 的 RMSE（Fig. D.3）给出同样排序。off-policy 目标均使用回放缓冲，批大小 256，20 个种子。

**§F 连续控制（探索性）**：10 步 Euler–Maruyama 离散化的 SDE $d\mathbf{x}_t = f(\mathbf{x}_t,t)\,dt + \tfrac12 d\mathbf{w}_t$，目标为 8gaussians 密度。TB + 探索噪声 $\sigma_{\mathrm{exp}}=0.1$ 时 MMD 达 0.0005（on-policy 为 0.1111）；用 forward KL 训 $P_B$ 的两个算法（REVERSE WS、FORWARD KL）早期梯度即出 NaN。作者明确标注连续情形的 GFlowNet 理论当时仅为猜想（脚注 3）。

## 与谁对话

- 上游：T01（GFlowNet 提出、hypergrid 与分子环境）、T02（DB 目标与流理论）、T03（TB 目标，本文的分析对象）、T05（SubTB 实测）、A17（结构学习环境与 Modified DB）。
- 同期对话：T09（Zimmermann et al. 2022）是明确的 concurrent work，从 VI 视角独立建立 GFlowNet 与散度目标的联系并研究前后 KL 之间的插值；两文互引。T09 侧重重要性加权与散度谱系，本文侧重梯度等价、baseline 识别与 off-policy 机制。
- 下游：T12（连续 GFlowNet 理论）在本文评审期间出现，把 §F 的猜想变成定理；T17 系统研究用一般 $f$-散度训练 GFlowNet，直接延续本文「TB 是一种伪散度准则」的提法；T16 用 policy gradient 重新审视训练；T14/T15 把「GFlowNet vs RL」推到熵正则 RL 严格等价，其中 T15 对本文的 off-policy 论断给出了重要的再检验。Richter et al. 2020 的 log-variance loss 等价性后来成为 T20 等扩散采样器工作中 VarGrad 类目标的理论接口。
- 方法论影响：本文把「off-policy 无 IS」确立为 GFlowNet 区别于 VI 的核心卖点，此后行为策略设计（回放缓冲、局部搜索、tempering）成为独立研究线（T23、T25、T35 等）。

## 局限与批判

- 梯度等价只在期望意义下成立，单样本/小批量下两类估计器的方差结构不同，论文对方差的比较主要靠实验而非定量刻画；后续研究梯度方差的工作正是补这一块。
- Prop. 1 的 $P_B$ 侧结论（式 (7)）需要从 $P_B$ 采样，实际算法中不可直接执行，论文承认只能借 IS 或换目标；等价性的实践意义主要落在 $P_F$ 侧。
- 实验里 HVI 的 off-policy 版本一律用朴素 IS 加权，未与更强的方差控制手段（per-decision IS、截断/自归一化 IS）比较，「off-policy VI 不行」的结论强度受此限制。T15 后来进一步指出：给 RL 一侧配上恰当的目标与探索，off-policy 差距会大幅缩小。
- 分子与结构学习实验中 $P_B$ 固定为均匀，等价性最干净但回避了「学 $P_B$ 时两框架差异」这一更微妙的问题；hypergrid 上学 $P_B$ 的设置未单独消融其贡献。
- 连续实验（§F）规模很小（2D、10 步、固定方差），当时缺理论支撑，只能算探索性证据。
- 「off-policy TB 最好」依赖手工设计的探索策略（logit 减 $\epsilon$、混合均匀策略），行为策略怎么选没有理论指导，论文自己也把这列为 open problem。

## 对后续研究的启示

- 把训练目标从「选哪个散度」重构为「选哪个期望梯度场 + 哪个采样分布」，是本文留下的最有用的分析框架：任何新目标都可以先问它 on-policy 时等价于什么散度、off-policy 时是否免 IS。
- $\log Z$ = 学出来的全局控制变量这一识别提示可以反向操作：把 VI 里的方差缩减技术（多样本 baseline、leave-one-out 估计）移植回 GFlowNet（T17 与 VarGrad 类工作正是如此）。
- SubNVI 的构造说明 junction layer / 状态流函数是在「轨迹级散度」与「转移级散度」之间连续插值的旋钮，credit assignment 粒度可以当作独立设计维度（T05、T06、T21 的部分推理沿此展开）。
- 行为策略选择被明确抛出为 open problem，且实验证明其收益大于目标函数本身的选择；这直接开启了 GFlowNet 探索机制研究（replay、local search、增广探索等）。
- 连续域猜想给 T12 划定了任务书：什么条件下「概率换成密度」后 TB/DB 理论仍成立。
