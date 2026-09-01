# N010 · 扩散采样器的改进离策略训练

> **Improved off-policy training of diffusion samplers**
> 作者：Marcin Sendera, Minsu Kim, Sarthak Mittal, Pablo Lemos, Luca Scimeca, Jarrid Rector-Brooks, Alexandre Adam, Yoshua Bengio, Esmeralda S. Whitammer · NeurIPS 2024 · [arXiv](https://arxiv.org/abs/2402.05098) · [代码](https://github.com/GFNOrg/gfn-diffusion)

## 一句话

把 PIS/DDS/DIS 与连续 GFlowNet 各种训练目标放进同一个基准库逐一对照，证伪了部分前人声称（FL-SubTB 的优势不可复现），证实了另一部分（离策略探索与 Langevin 参数化确实有效），并提出目标空间并行 MALA 局部搜索 + 回放缓冲区这一新探索技术，在多数任务上取得最优或持平的采样质量。

## 1. 要解决的问题

给定可微能量函数 \(\mathcal E:\mathbb R^d\to\mathbb R\)，令奖励 \(R(\mathbf x)=\exp(-\mathcal E(\mathbf x))\)、配分函数 \(Z=\int_{\mathbb R^d}R(\mathbf x)\,d\mathbf x\)，目标是只靠查询 \(\mathcal E\)（及其梯度）训练一个扩散模型，从 Boltzmann 密度 \(p_{\text{target}}(\mathbf x)=R(\mathbf x)/Z\) 采样并估计 \(Z\)。没有目标分布的样本可用，经典 MCMC（MALA/HMC）在多峰高维分布上混合缓慢，这促使人们转向摊销变分推断（amortized variational inference）。

论文指出该领域存在两个具体痛点：

1. **基准混乱**：既有工作（PIS、DDS、DIS、DGFS 等）架构不一、超参数未公开、甚至对同一个目标密度的定义都不一致（附录 B.1）；作者明确指出 DGFS（Zhang et al., ICLR 2024）在对比实验中修改了关键实验变量、报告了不可复现的结果。
2. **训练难题未解**：离策略采样模型的两大瓶颈是探索效率（发现高奖励区域）与信用分配（credit assignment，把终端奖励信号传播回早期采样步的参数）。

## 2. 核心方法

**设定：Euler–Maruyama 分层采样即 GFlowNet。** 生成过程为神经 SDE

\[
d\mathbf x_t = u(\mathbf x_t,t;\theta)\,dt + g(\mathbf x_t,t;\theta)\,d\mathbf w_t, \tag{1}
\]

其中 \(u\) 为漂移、\(g\) 为扩散率、\(\mathbf w_t\) 为 Wiener 过程。取 \(T\) 步离散化（\(\Delta t=1/T\)），每步转移核为高斯

\[
p_F(\mathbf x_{t+\Delta t}\mid \mathbf x_t)=\mathcal N\big(\mathbf x_{t+\Delta t};\,\mathbf x_t+u(\mathbf x_t,t;\theta)\Delta t,\; g(\mathbf x_t,t;\theta)^2\Delta t\,\mathbf I_d\big), \tag{3}
\]

这恰好是连续 GFlowNet（Lahlou et al., ICML 2023）里状态空间 \(\mathcal S=\{(\mathbf x,t)\}\) 上的前向策略；学 SDE 等价于学带高斯策略的 GFlowNet。学习目标是使终止分布匹配目标：\(p_F^\top(\mathbf x_1;\theta)=R(\mathbf x_1)/Z\)（式 9）。

**训练目标族。** 核心是轨迹平衡（Trajectory Balance, TB）：\(p_F^\top\) 采样自目标分布当且仅当存在反向策略 \(p_B\) 与标量 \(Z_\theta\) 使得对每条完整轨迹 \(\tau=(\mathbf x_0\to\cdots\to\mathbf x_1)\)

\[
Z_\theta\, p_F(\tau;\theta) = R(\mathbf x_1)\, p_B(\tau\mid \mathbf x_1;\psi), \tag{10}
\]

对应损失为两边 log 比值的平方：

\[
\mathcal L_{\mathrm{TB}}(\tau;\theta,\psi)=\left(\log\frac{Z_\theta\,p_F(\tau;\theta)}{R(\mathbf x_1)\,p_B(\tau\mid\mathbf x_1;\psi)}\right)^2. \tag{11}
\]

与 KL 目标（式 7，PIS 所用）不同，\(\mathcal L_{\mathrm{TB}}\) 可以用任意全支撑分布 \(\pi\) 采的轨迹做**离策略**优化，且不需要对前向过程的模拟反向传播。主实验沿用前人设定：\(T=100\)，\(p_B\) 固定为离散化 Brownian bridge（式 15），前向方差固定为 \(\sigma^2\)。另外对比 VarGrad 目标（等价于先对 batch 内 \(\log Z_\theta\) 解析最优、再优化策略参数）与子轨迹平衡（SubTB，式 12，需额外学习状态流 \(f(\mathbf x_t;\theta)\)）。

**被检验的两种信用分配归纳偏置：**

- **前视（forward-looking, FL）状态流**（DGFS 所用，式 13）：\(\log f(\mathbf x_t;\theta)=(1-t)\log p^{\mathrm{ref}}_t(\mathbf x_t)+t\log R(\mathbf x_t)+\mathrm{NN}(\mathbf x_t,t;\theta)\)，把中间态能量注入状态流参数化，\(p^{\mathrm{ref}}_t\) 为速率 \(\sigma\) 的 Brownian 运动边缘密度。
- **Langevin 参数化（Langevin parametrization, LP）**（源自 PIS，式 14）：\(u(\mathbf x_t,t;\theta)=\mathrm{NN}_1(\mathbf x_t,t;\theta)+\mathrm{NN}_2(t;\theta)\nabla\mathcal E(\mathbf x_t)\)，即漂移 = 学习修正项 + 缩放的能量梯度（Langevin 漂移），把奖励信号直接喂给策略。

**新提出：目标空间局部搜索 + 回放缓冲区（LS）。** 先从当前采样器抽 \(M\) 个终端候选 \(\{\mathbf x^{(1)},\dots,\mathbf x^{(M)}\}\sim p_F^\top\)，以其为初始状态并行跑 \(M\) 条 MALA 链各 \(K\) 步，弃掉 \(K_{\text{burn-in}}\) 步后把接受样本存入缓冲区 \(\mathcal D_{\mathrm{LS}}\)。训练时交替两步：Step A 用在策略/探索性前向轨迹训练，Step B 从 \(\mathcal D_{\mathrm{LS}}\) 抽终端样本 \(\mathbf x\)、用 \(p_B\) 回溯出轨迹 \(\tau\) 再做 TB 梯度更新。MALA 高度可并行且只需偶尔执行（缓冲区远大于 batch），故每条训练轨迹不增加额外计算开销——这是相对 FL（要学状态流）和 LP（每步要算 \(\nabla\mathcal E\)）的直接卖点。

## 3. 理论结果

本文以实证为主，理论内容为引述与一个猜想：

- TB 条件（式 10）与终止分布正确（式 9）的等价性来自 Lahlou et al. (2023)，非本文新证。
- 在策略采样时 TB 梯度是 KL 梯度的无偏估计（引自 Malkin et al. 2023 等）：\(\mathbb E_{\tau\sim p_F}[\nabla_{\theta'}\mathcal L_{\mathrm{TB}}(\tau;\theta,\psi)]=2\nabla_{\theta'} D_{\mathrm{KL}}(p_F(\tau;\theta)\,\|\,p_{\text{target}}(\mathbf x_1)p_B(\tau\mid\mathbf x_1;\psi))\)，其中 \(\nabla_{\theta'}\) 不含 \(Z_\theta\)。该估计方差高于 PIS 的重参数化估计，但免去穿越模拟的反传。
- **猜想（未证明）**：当 \(\mathrm{NN}_1\) 不依赖 \(\mathbf x_t\) 时，LP（式 14）与 FL 状态流（式 13）在 \(\Delta t\to 0\) 极限下等价（诱导相同的短子轨迹 SubTB 残差渐近）。作者把严格分析留给后续工作；终版附注指出 Berner et al.（arXiv:2501.06148）随后建立了这些算法族的连续时间极限与渐近等价。

## 4. 实验与证据

**任务**：无条件——25GMM（\(d=2\)，25 峰高斯混合）、Funnel（\(d=10\)）、Manywell（\(d=32\)）、LGCP（\(d=1600\)，log-Gaussian Cox 过程）；条件——预训练 MNIST VAE 的 20 维隐变量后验采样。基线三类：MCMC 系（SMC、GGNS 嵌套采样）、模拟驱动变分系（DIS/DDS/PIS）、GFlowNet 系（TB/VarGrad/FL-SubTB 及其增强）。所有神经方法统一架构。

**指标**：变分下界估计 \(\log\hat Z\)（\(K=2000\) 样本）与重要性加权变体 \(\log\hat Z_{\mathrm{RW}}\)（后者强调模式覆盖，\(K\to\infty\) 时趋于真值），另有 2-Wasserstein 距离（附录 C.1）。显著性判定用单侧 Welch t 检验（\(p<0.05\)）。

**主要发现（Table 1、Table 2、Fig. 1–3）**：

1. 裸 TB 表现平平，加简单探索（策略方差加常数）后在 25GMM 上明显改善（如 \(\Delta\log Z\) 从 1.176±0.109 降到 0.560±0.302）；Fig. 2 显示**随训练衰减的探索方差**优于常数探索。
2. **FL-SubTB 相对 TB 没有一致且显著的优势**，且训练成本更高——直接质疑 DGFS 的核心声称；换成 VarGrad 结果与 TB 相近。
3. LP 加到 TB 或 FL-SubTB 上都带来显著提升（尽管每迭代慢 2–3 倍），说明 PIS 的观察迁移到离策略算法。
4. **LS 增益最大**：TB+Expl.+LS 在 25GMM 上 \(\Delta\log Z=0.171\pm0.013\)、\(\Delta\log Z_{\mathrm{RW}}=0.004\pm0.011\)；TB+Expl.+LP+LS 在 Manywell 上 \(\Delta\log Z_{\mathrm{RW}}=0.07\pm0.17\)、在 LGCP 上 \(\log\hat Z_{\mathrm{RW}}=489.03\pm1.38\)（接近 DDS 的 489.30±0.62），多数任务/指标追平或超过全部基线；Fig. 1 展示其防止 Manywell 上的模式坍缩。
5. **条件任务（VAE）**：TB 系明显不如 PIS（拟合条件配分函数困难），但 VarGrad+Expl.+LP+LS 达到 \(\log\hat Z_{\mathrm{RW}}=-46.245\pm0.543\)，与 PIS+LP（−47.326±0.777）持平或更优——VarGrad 只学策略、不需拟合条件 \(\log Z\) 是作者给出的解释。
6. **扩展实验（§5.3）**：学习前向策略方差可在少步数（\(T\) 小至 10）时避免最后一步高斯噪声"糊掉"尖峰，固定方差与学习方差模型的 \(\log Z\) 估计分别为 −1.67 与 −0.62（Fig. 3）；库还支持学习 backward 过程与 VP 噪声调度（附录 C.2 初步结果）。

**维度扩展性（附录 C.3，Manywell \(d\in\{8,32,128,512\}\)，Table C.3）**：

- \(d=128\) 时裸 TB 崩坏（\(\Delta\log Z=205.6\)），TB+LP 降到 46.4，TB+LP+LS 为 66.6 但 \(\Delta\log Z_{\mathrm{RW}}=14.9\) 与 TB+LP 的 14.0 相当；\(d=512\) 时所有方法都严重退化（最好也在 \(\Delta\log Z\approx 200\) 量级）。
- LP 的每迭代开销随维度增长但"对性能至关重要"；FL-SubTB 开销更高；单独 LS 开销最低，但在高维（\(d=128\) 时 \(\Delta\log Z=458.7\)）不加 LP 时探索不动——LS 与 LP 是互补而非替代。

**关键训练细节（附录 D）**：基础扩散率 \(\sigma^2\)：25GMM 取 5、Funnel/Manywell 取 1、LGCP 扫描 \(\{1,3,5\}\) 后取 5；学习率 \(10^{-3}\)（流参数 TB 下 \(10^{-1}\)、SubTB 下 \(10^{-2}\)）；探索因子 0.2、前半程线性衰减；batch 300；普通模型训 25,000 迭代、带 Langevin 的训 10,000 迭代以对齐总计算量；对 LP 的能量得分做 \(\pm10^2\)、对策略网络输出做 \(\pm10^4\) 裁剪（式 17）。附录 C.2 用 VP（Ornstein–Uhlenbeck）噪声过程（式 16，\(\beta_{\min}=0.01,\beta_{\max}=4.0\)）替换 Brownian bridge，Manywell 上结果与 Brownian 相近。

## 5. 在 GFlowNet 版图中的位置

- 这是"连续 GFlowNet = 扩散采样器"这条桥的**实证支柱**：理论骨架由 Lahlou et al. (ICML 2023) 给出，本文把 TB/SubTB/VarGrad 与 PIS/DDS/DIS 放到同一实现下对拍，直接催生了 Berner et al. (2025) 的连续时间极限定理。
- 它把离散 GFlowNet 社区的探索工具（局部搜索 GFlowNet、回放缓冲区、GGNS 的 backward-trace 训练）系统性地搬到连续域，其中 LS 的设计明确借鉴 Kim et al. (ICLR 2024) 的 Local Search GFlowNets，但把 MCMC 核从"策略诱导"换成目标空间 MALA。
- `gfn-diffusion` 库成为后续 diffusion sampler 工作的公共基准（含 25GMM/Funnel/Manywell/LGCP/VAE 任务与统一指标），此后论文普遍以它的 TB+Expl.+LS / +LP 作为对照组。
- 对社区内部，它扮演"复现警察"角色：附录 B.1 逐条指出 DGFS 与 PIS 等论文在目标密度定义、超参数、评测口径上的不一致。

## 6. 局限与批判

- 主实验继承前人设定：\(p_B\) 固定为 Brownian bridge、前向方差固定、\(T=100\)——学习 backward/方差/噪声调度只给了初步证据（§5.3、附录 C.2），未进主表。
- LP 与 LS 都依赖 \(\nabla\mathcal E\) 可得且便宜：LP 每步采样都要算梯度（慢 2–3 倍），LS 的 MALA 同样需要梯度；对黑盒（不可微）能量两者都失效，此时本文的结论退化为"TB+探索方差"。
- LS 在 VAE 条件任务上反而伤害 TB（\(\log\hat Z\) 从 −148 恶化到 −245.78±13.80）；作者未给出机制解释，只能靠换 VarGrad 补救——说明 LS 并非普适。
- LGCP 无解析 \(\log Z\)，只能报告下界估计值比大小，"更高"不严格等于"更好"。
- FL≈LP 的连续时间等价只是猜想；论文自己的显著性判定（Welch t 检验，5 次运行）在若干列上把多个方法同时标为"最优不可区分"，区分度有限。
- 基准全部是合成能量或小规模 VAE，最高维 LGCP（\(d=1600\)）结构规则，与真实科学应用（如分子 Boltzmann 分布）尚有距离；且 \(d=512\) 的 Manywell 上没有任何方法真正可用，高维仍是开放问题。

## 7. 对后续研究的启示

- **基准先行**：该库统一了任务/指标/架构，使后续方法（如 Berner et al. 2025 的加速训练）可以做 apples-to-apples 比较；做新采样器先接入该库已成惯例。
- **探索与信用分配解耦**：LS 证明"把探索外包给目标空间 MCMC + 回放"比改损失函数（SubTB）更划算；这一"采样器管摊销、MCMC 管探索"的分工被后续 Boltzmann 采样与 LLM 推理微调工作反复复用。
- **VarGrad 在条件问题上的优势**（免学条件 \(\log Z\)）预示了后来 LLM 场景下 in-batch \(\log Z\) 估计的流行（如 TBA、GFlowRL 一系）。
- 少步数下学习方差的实验（Fig. 3）指向"粗离散化 + 非高斯转移"的后续方向，作者也点名图像扩散模型 GFlowNet 微调是库的直接应用场景。
- 遗留的连续时间极限问题由 Berner et al. (arXiv:2501.06148) 接棒解决，说明"离散 GFlowNet 目标的 \(\Delta t\to0\) 行为"是值得形式化的富矿。
