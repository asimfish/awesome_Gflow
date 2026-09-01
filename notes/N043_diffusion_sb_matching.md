# N043 · 扩散薛定谔桥匹配：用迭代马尔可夫拟合求解 SB

> **Diffusion Schrödinger Bridge Matching**
> 作者：Yuyang Shi* (Oxford), Valentin De Bortoli* (ENS Ulm), Andrew Campbell (Oxford), Arnaud Doucet (Oxford)，前两人同等贡献 · NeurIPS 2023 · [arXiv](https://arxiv.org/abs/2303.16852) · [代码](https://github.com/yuyang-shi/dsbm-pytorch)

## 一句话

提出迭代马尔可夫拟合（Iterative Markovian Fitting, IMF）——在马尔可夫测度类与 reciprocal class 之间交替投影来解 Schrödinger bridge（SB），每步只需一次 Bridge Matching 回归；对应算法 DSBM 消除了 IPF 系 DSB 方法的时间离散误差与跨迭代误差累积，并把 Bridge/Flow Matching、Rectified Flow、OT-CFM 全部收编为特例或极限。

## 1. 要解决的问题

SB 问题：给定参考路径测度 \(Q\)（由 SDE \(dX_t = f_t(X_t)dt + \sigma_t dB_t\) 定义），求

\[
P^{\mathrm{SB}} = \arg\min_{P} \{\mathrm{KL}(P|Q) : P_0 = \pi_0,\ P_T = \pi_T\},\tag{6}
\]

即在两端边缘约束下离参考扩散最近的路径测度；其静态版 \(\Pi^{\mathrm{SB}}_{0,T}\) 是熵正则最优传输（entropy-regularized OT, EOT）的解。经典数值方法（Sinkhorn 的连续版 Iterative Proportional Fitting, IPF）驱动了 DSB（De Bortoli et al. 2021）等扩散式求解器，但有三个实际问题：(i) 每轮 IPF 要学上一轮过程的时间反转，需要缓存**整条轨迹**，时间离散误差进入训练目标；(ii) 迭代中参考桥被逐渐"遗忘"（Fernandes et al. 2021），误差跨迭代累积；(iii) IPF 迭代不保持两端边缘（只在极限满足）。另一头，Bridge Matching / Flow Matching 训练简单但**不解 OT/SB**——学到的传输图不保证接近最优。本文要同时拿到两边的好处。

## 2. 核心方法

**两个投影。** 设 \(\mathcal{M}\) 为马尔可夫扩散测度类，\(\mathcal{R}(Q)\) 为 \(Q\) 的 reciprocal class（与 \(Q\) 有相同桥 \(Q_{|0,T}\) 的测度：\(\Pi = \Pi_{0,T}Q_{|0,T}\)，Definition 3）。

- **马尔可夫投影（Definition 1）**：对混合桥测度 \(\Pi = \Pi_{0,T}Q_{|0,T}\)，其投影 \(M^\star = \mathrm{proj}_{\mathcal{M}}(\Pi)\) 由 SDE

\[
dX^\star_t = \{f_t(X^\star_t) + v^\star_t(X^\star_t)\}dt + \sigma_t dB_t,\quad v^\star_t(x_t) = \sigma_t^2\,\mathbb{E}_{\Pi_{T|t}}[\nabla\log Q_{T|t}(X_T|X_t)\,|\,X_t = x_t]
\]

给出。Proposition 2：\(M^\star = \arg\min_{M\in\mathcal{M}}\mathrm{KL}(\Pi|M)\)，且保持全部时刻边缘 \(M^\star_t = \Pi_t\)（特别地两端边缘不动）。
- **reciprocal 投影（Definition 3/Proposition 4）**：\(\Pi^\star = \mathrm{proj}_{\mathcal{R}(Q)}(P) = P_{0,T}Q_{|0,T}\)，即保留两端联合分布、把中间路径换成参考桥；它是 \(\arg\min_{\Pi\in\mathcal{R}(Q)}\mathrm{KL}(P|\Pi)\)。

**IMF（式 (8)）。** 由 Proposition 5——SB 是唯一同时满足"马尔可夫 + 属于 \(\mathcal{R}(Q)\) + 两端边缘正确"的测度——构造交替投影序列

\[
P^{2n+1} = \mathrm{proj}_{\mathcal{M}}(P^{2n}),\qquad P^{2n+2} = \mathrm{proj}_{\mathcal{R}(Q)}(P^{2n+1}),
\]

初始化 \(P^0 \in \mathcal{R}(Q)\) 且 \(P^0_0=\pi_0, P^0_T=\pi_T\)。与 IPF 恰成对偶（Table 1）：IPF 在"固定 \(\pi_0\)"与"固定 \(\pi_T\)"两个集合间交替、保持马尔可夫性和 reciprocal 性；IMF 在 \(\mathcal{M}\) 与 \(\mathcal{R}(Q)\) 间交替、**每一步都保持两端边缘**。

**DSBM（Algorithm 1）。** reciprocal 投影是免费的：从 \(M^n_{0,T}\) 采两端对 \((X_0,X_T)\)，再用解析桥（Brownian bridge：\(X^{0,T}_t = \frac{t}{T}x_T + (1-\frac{t}{T})x_0 + \sigma_t(B_t - \frac{t}{T}B_T)\)，式 (4)）插值。马尔可夫投影用 Bridge Matching 回归（式 (10)）：

\[
\theta^\star = \arg\min_\theta \int_0^T \mathbb{E}_{\Pi_{t,T}}\big[\|\sigma_t^2\nabla\log Q_{T|t}(X_T|X_t) - v_\theta(t,X_t)\|^2\big]/\sigma_t^2\,dt.
\]

纯前向迭代会在 \(\pi_T\) 端累积回归误差，因此用 Proposition 9（马尔可夫投影的时间对称性：同一 \(M^\star\) 既可写成前向 SDE (11) 也可写成从 \(\pi_T\) 出发的反向 SDE (12)）交替做前向/反向投影。反向投影用对称的回归目标（式 (14)）：

\[
\phi^\star = \arg\min_\phi \int_0^T \mathbb{E}_{\Pi_{0,t}}\big[\|\sigma_t^2\nabla\log Q_{t|0}(X_t|X_0) - v_\phi(t,X_t)\|^2\big]/\sigma_t^2\,dt,
\]

前向从 \(X_0\sim\pi_0\) 出发、反向从 \(Y_0\sim\pi_T\) 出发，两端偏差互相清零。相比 DSB：只缓存 \((X_0,X_T)\) 对而非整条轨迹、回归可在任意 \(t\) 评估（连续时间训练）、显式 reciprocal 投影杜绝桥遗忘。

**Brownian bridge 特例与 Flow Matching 极限。** 取 \(f_t=0,\sigma_t=\sigma\) 时桥有闭式（式 (4)），回归目标化简为（式 (5)）

\[
\mathbb{E}_{\Pi_{t,T}}\big[\|(X_T - X_t)/(T-t) - v_\theta(t,X_t)\|^2\big],
\]

令 \(\sigma\to 0\) 即 Flow Matching / Conditional Flow Matching 的目标（Appendix A.1 证明了 FM、CFM、Rectified Flow 第一轮在生成建模设定下三者等价）；stochastic interpolants（Albergo et al. 2023）的参数化 \(X_t = \bar\alpha_t x_0 + \bar\beta_t x_T + \bar\gamma_t Z\) 与本文桥 SDE 参数化的显式换算在 Appendix B.1 给出。

**统一图景（Figure 1 / Appendix A.2）。** 初始化耦合决定身份：独立耦合 \(\Pi^0_{0,T}=\pi_0\otimes\pi_T\) 为 DSBM-IMF；参考过程耦合 \(\Pi^0_{0,T}=Q_{0,T}\) 时最优迭代逐点等于 IPF 迭代（Proposition 10），得 DSBM-IPF；minibatch-EOT 耦合初始化得 DSBM-IMF+。第一轮迭代即 Bridge Matching；\(\sigma\to 0\) 且只做前向投影即 Rectified Flow；给定真 SB 静态耦合则一轮收敛（Somnath et al. 2023 的 aligned SB）。收敛后概率流 ODE 为 \(dZ^\star_t = \{f_t + \frac{1}{2}[v_{\theta^\star} - v_{\phi^\star}]\}dt\)。

## 3. 理论结果

- **Proposition 5**（SB 唯一刻画）：马尔可夫 + reciprocal + 两端边缘 ⟹ 唯一且等于 \(P^{\mathrm{SB}}\)。这是 IMF 的不动点依据。
- **Lemma 6**（勾股定理）：\(\mathrm{KL}(\Pi|M) = \mathrm{KL}(\Pi|\mathrm{proj}_{\mathcal{M}}(\Pi)) + \mathrm{KL}(\mathrm{proj}_{\mathcal{M}}(\Pi)|M)\)，reciprocal 投影有对称版本。
- **Proposition 7**（单调收敛）：\(\mathrm{KL}(P^{n+1}|P^{\mathrm{SB}}) \le \mathrm{KL}(P^n|P^{\mathrm{SB}}) < \infty\)，且 \(\lim_{n\to\infty}\mathrm{KL}(P^n|P^{n+1}) = 0\)；与 IPF 的经典结果（Rüschendorf 1995, Prop 2.1）互为 forward/reverse KL 镜像。
- **Theorem 8**（收敛性）：IMF 序列有唯一不动点 \(P^\star = P^{\mathrm{SB}}\)，\(\lim_{n\to\infty}\mathrm{KL}(P^n|P^\star) = 0\)。同期独立工作 Peluchetti (2023, Theorem 2)（其 IDBM 即 DSBM-IMF）先给出该结果，本文给出更简洁证明。
- **Proposition 9/10**：马尔可夫投影的前向/反向双表示；DSBM-IPF 与经典 IPF 迭代的逐点等价（在函数族足够丰富的假设下）。
- 定性区分 Rectified Flow：Proposition 5 只在 \(\sigma_t > 0\) 时成立，RF（\(\sigma=0\)）不保证收敛到动态 OT（Liu 2022 有反例），且 RF 只做前向投影导致 \(P^n_T\) 偏差随迭代恶化——SDE 化 + 双向投影正是修复此缺陷的理论解释（Appendix A.3）。
- 附录补充：Appendix D 给出 Gaussian 情形 IMF 的解析迭代（可对照 Bunne et al. 2023 的闭式 SB）；Appendix E 推导离散时间马尔可夫投影；Appendix G 提出前向/反向过程联合学习 + 一致性损失（强制 \(v_\phi(t,x) = -v_\theta(t,x) + \sigma_t^2\nabla\log P_t(x)\) 的两侧互为时间反转），作为交替式训练的替代。

## 4. 实验与证据

- **2D 传输**（Table 2，5 seeds，Euler 20 步）：moons/scurve/8gaussians/moons-8gaussians 上比 2-Wasserstein 和路径能量 \(\mathbb{E}[\int_0^T\|v(t,Z_t)\|^2 dt]\)。2-Wasserstein 上 DSBM 全面优于 DSB（moons：0.140±0.006 vs 0.190±0.049；moons-8gaussians：0.812±0.092 vs 0.987±0.324）；不用 OT solver 时优于 FM（0.212±0.025）/CFM（0.215±0.028）；用 minibatch solver 的 OT-CFM 在低维最强（moons-8gaussians 0.716±0.187），DSBM-IMF+ 次之（0.802±0.172）；RF 在 moons-8gaussians 上崩坏（1.522±0.304）。路径能量上 CFM 最差（moons-8gaussians 116.5±2.633），DSBM 三变体聚在 41–42，OT-CFM 30.50±0.626 最短——用 OT solver 换直路径在低维划算，高维见下。
- **高维 Gaussian**（\(d=50\)，真 SB 有闭式解）：DSB 与 IMF-b（只做反向投影的消融）的协方差估计随迭代漂移，DSBM 不漂移；Table 3 的 \(\mathrm{KL}(P_t|P^{\mathrm{SB}}_t)\times 10^{-3}\)：\(d=50\) 时 DSB 32.8±1.28、SB-CFM 49.4±3.91、DSBM-IPF **8.75±0.87**。
- **MNIST↔EMNIST 迁移**：DSB 与 RF 的 FID 随训练恶化，DSBM 不恶化且比 DSB 快约 30%；OT-CFM 高维失效（Figure 4a）。
- **CelebA 64×64（male/old ↔ female/young）**：\(\sigma^2 \in \{0.01,0.1,1,10\}\) 消融——FID 随 \(\sigma\) 先降后升，LPIPS（对齐度）单调变差（Figure 7）；同一 \(\sigma=1\) 在 128×128 上比 64×64 对齐更好，呼应"噪声表应随分辨率缩放"。
- **AFHQ 512×512 cat↔wild** 与**无配对流体下采样**（64×64 低分辨率 → 512×512）：DSBM 的 \(\ell_2\) 谱距离全频段低于 Diffusion-fb 基线（Figure 11），展示了 SB 类方法真正的应用甜区——非配对域迁移。

作者自报：CIFAR-10 生成任务上相对 Bridge/Flow Matching 只有轻微改进（Appendix I.6）——DSBM 的价值在一般传输而非纯生成。

## 5. 与 GFlowNet 生态位的关系

**问题设定正交，方法论威胁真实存在。**

- **数据条件不同**：DSBM 要求**两端边缘都能采样**（\(\pi_0,\pi_T\) 的样本），学的是二者之间的熵正则耦合；GFlowNet 只需要未归一化 reward \(R(x)\)，学的是单个目标分布的构造式采样器，并附带估计 \(Z\)。一个是 transport 问题，一个是 sampling 问题——在各自的原生设定里互不可替代。
- **但在"图/离散空间上的熵正则 OT"这个交叉生态位上是正面竞争**：若想做"GFlowNet 当作离散 SB/EOT 求解器"（本仓库 OT 分析文档 §5 列的高潜力方向），IMF 这套投影机制是现成的、有收敛定理的替代技术——离散扩展 DDSBM（CTMC + IMF）、GSBoG 等都直接建立在 IMF 之上，已把该生态位占得很满。GFlowNet 若进场，必须回答"balance 训练比 IMF 交替回归好在哪"。
- **结构对照**：IMF 的两步（马尔可夫投影 = 学一个匹配边缘的策略；reciprocal 投影 = 保留端点耦合、重刷中间路径）与 GFlowNet 的 TB（一步约束整条轨迹流量守恒）是两种截然不同的守恒实施方式。IMF 每步是**无仿真回归**（桥有闭式），只有缓存耦合时需要采样学到的 SDE；GFlowNet 训练每步都要 rollout 策略。这是 DSBM 系在可扩展性上的真实优势。
- **GFlowNet 的独特能力**：(i) 不需要目标样本、只要 reward——DSBM 完全做不了这个设定；(ii) 天然的组合状态空间与共享子结构 credit assignment；(iii) \(Z\) 估计。反之 GFlowNet 的劣势：没有 IMF 这种"每步保两端边缘 + KL 单调下降"的干净投影几何，TB 残差与最终误差的关系仍是开放问题。
- **人员交叉**：SB-CFM/OT-CFM 的 Tong et al. (2023) 作者列表含 Malkin 与 Bengio（GFlowNet 核心圈），两个社区在 simulation-free 训练这条线上早已互相渗透。

## 6. 局限与批判

- **误差累积并未根除，只是换了位置**：马尔可夫投影的回归误差仍会让 \(M^{n+1}_T \ne \pi_T\)（作者用前向/反向交替缓解而非消除）；Theorem 8 是理想投影下的收敛，有限容量网络 + 有限样本下没有误差界。
- 缓存步仍需模拟学到的 SDE 采 \((X_0,X_T)\)，不是完全 simulation-free；\(\sigma\) 小时 EOT 数值上更难（作者自认），因此"逼近确定性 OT"这个卖点在 \(\sigma\to 0\) 极限恰恰失效。
- 生成建模收益单薄（CIFAR-10 仅轻微改进），说明 SB 最优性对纯生成质量帮助有限——买 SB 的理由必须来自传输结构本身（对齐、插值、能量最优）。
- 收敛速度没有速率刻画：\(\mathrm{KL}(P^n|P^{n+1})\to 0\) 不给出迭代次数与精度的定量关系；实验里 20 轮外循环是经验选择。
- 与 Peluchetti (2023) 的 IDBM 撞车（作者已注明并发），单从 IMF 理论看本文的独立增量是 IPF 对偶视角、前向/反向交替与大规模实验。

## 7. 对后续研究的启示

- IMF 已成为 SB 数值的新默认范式：离散版（DDSBM 的 CTMC-IMF）、图版（GSBoG）、benchmark（离散 SB/EOT 测评）全部沿此线展开。任何想在这个生态位做 GFlowNet 变体的工作，DSBM 是必须对比的基线，且要在 IMF 不擅长的维度上找差异化——unbalanced 边缘、未知 \(Z\)、reward-only 设定、组合动作空间。
- "每步保持什么、逼近什么"的投影对偶（IPF vs IMF）是可移植的设计透镜：GFlowNet 的 DB/TB/SubTB 目标也可以问同样的问题——哪些量在训练全程被硬保持、哪些只在收敛时成立？设计"每步保 reward 边缘"的 GFlowNet 更新是一个直接可试的类比。
- reciprocal 投影"只存端点耦合 + 解析桥重建路径"的缓存技巧对任何轨迹式采样器（含 GFlowNet 的 replay buffer）都是内存优化模板。
- \(\sigma\) 作为"对齐度 vs 生成质量"的旋钮（CelebA 消融）给跨域迁移应用提供了明确调参指南，也提示熵正则强度应随问题维度/分辨率一起调。
