# T09 · GFlowNet 的变分视角与 αKL 目标族

> **A Variational Perspective on Generative Flow Networks**
> 作者：Heiko Zimmermann, Fredrik Lindsten, Jan-Willem van de Meent, Christian A. Naesseth（Amsterdam Machine Learning Lab, UvA / Linköping University） · TMLR 2023 · [arXiv](https://arxiv.org/abs/2210.07992) · 代码：未开源（arXiv 版与公开渠道均未给出仓库链接）

## 一句话

从变分推断（variational inference, VI）一侧独立建立「Trajectory Balance（TB）≈ 轨迹空间 KL 散度的 score-function 优化」这一等价关系，并据此提出两件 T08 没有的东西：反向/前向 KL 的凸组合目标族 αKL（对照 αTB），以及把 VI 的控制变量（control variate）工具箱（含 leave-one-out 估计）系统移植进 GFlowNet 训练。

## 问题与动机

GFlowNet（T01/T02）用 flow matching、detailed balance 或 TB（T03）条件训练一个沿 DAG 顺序采样的策略，使终态分布正比于奖励 $R$。把 $R$ 看成非归一化目标密度，这就是概率推断里的老问题——从非归一化分布采样，而 MCMC、重要性采样（importance sampling）、VI 三大家族已有大量工具。本文追问：

1. TB 目标在什么意义上就是一个变分目标？用前向模型采样、用后向模型采样、或按比例混合两者采样时，TB 分别对应什么散度？
2. 前向/后向转移模型共享参数与不共享参数时，上述对应关系是否仍成立？
3. 一旦识别出等价性，VI 文献里的方差缩减技术（控制变量、LOO baseline）能否直接搬来训 GFlowNet？

与 T08 是同期独立工作（论文明确写「concurrent and independent work」），两文推出的核心等价式相同；本文的差异化贡献是 αKL/αTB 凸组合目标、能量模型（energy-based model, EBM）联合训练语境下的比较、以及共享参数情形的分析。

## 方法核心

**记号。** 轨迹 $\tau=(s_0,s_1,\dots,s_T,s_f)$ 沿 DAG $G=(\mathcal{S},\mathcal{E})$ 从根 $s_0$ 到终止态 $s_T$ 再进入唯一叶节点 $s_f$。轨迹流（trajectory flow）$F:\mathcal{T}\to\mathbb{R}^+$ 诱导概率测度 $P(A)=F(A)/Z$，$Z=\sum_{\tau} F(\tau)$ 为总流量；状态流 $F(s)$、边流 $F(s\to s')$ 由事件流定义，前后向转移概率为 $P_F(s'\mid s)=F(s\to s')/F(s)$、$P_B(s\mid s')=F(s\to s')/F(s')$。马尔可夫流满足 $P(\tau)=\prod_t P_F(s_{t+1}\mid s_t)=\prod_t P_B(s_t\mid s_{t+1})$。定义两个轨迹级模型：

$$Q(\tau;\phi) := \prod_{t=0}^{T} P_F(s_{t+1}\mid s_t;\phi) \quad\text{（前向模型）}, \qquad P(\tau;\theta) := \frac{R(s_T)}{Z}\prod_{t=0}^{T-1} P_B(s_t\mid s_{t+1};\theta) \quad\text{（后向模型）}.$$

TB 目标（式 (1)，$\lambda=(\phi,\theta,\psi)$，$Z_\psi$ 为学习的总流量估计）：

$$\mathcal{L}_{\mathrm{TB}}(\tau,\lambda) = \left( \log \frac{Z_\psi \prod_t P_F(s_{t+1}\mid s_t;\phi)}{R(s_T)\prod_t P_B(s_t\mid s_{t+1};\theta)} \right)^2 = \left( \log \frac{Z_\psi\, Q(\tau;\phi)}{Z\, P(\tau;\theta)} \right)^2.$$

**变分目标。** 记重要性权重 $w := P(\tau;\theta)/Q(\tau;\phi)$，则反向 KL（RKL，式 (2)）与前向 KL（FKL，式 (3)）为

$$\mathcal{L}_{\mathrm{RKL}} = \mathrm{KL}(Q \Vert P) = \mathbb{E}_{\tau\sim Q}[-\log w], \qquad \mathcal{L}_{\mathrm{FKL}} = \mathrm{KL}(P \Vert Q) = \mathbb{E}_{\tau\sim P}[\log w].$$

对「自己一侧」参数求导需要 score-function 梯度：例如 $\frac{d}{d\phi}\mathcal{L}_{\mathrm{RKL}} = \mathbb{E}_{\tau\sim Q}[-\log w \cdot \frac{d}{d\phi}\log Q(\tau;\phi)]$（利用 $\mathbb{E}_Q[a\,\nabla_\phi \log Q]=0$ 消掉常数项）；这类估计器方差高。

**控制变量工具箱（§3.1）。** 修正估计器 $g' = g + c(h-\mathbb{E}[h])$ 保持无偏，方差 $\mathrm{Var}[g'] = \mathrm{Var}[g] + c^2\mathrm{Var}[h] - 2c\,\mathrm{Cov}[g,h]$（式 (4)），最优标度 $c^* = \mathrm{Cov}[g,h]/\mathrm{Var}[h]$。取 $h=\frac{d}{d\phi}\log Q$（score function 本身）时，$c$ 直接加进负对数权重：$g' = \frac1S\sum_s(-\log w_s + c)\frac{d}{d\phi}\log Q(\tau_s;\phi)$。为保证无偏，用批内样本估 $c$ 时须用 leave-one-out（LOO）估计 $\hat c_s$（Mnih & Rezende 2016）；两个免梯度信息的次优选择是 $c^{\log w}=\mathbb{E}[\log w]$ 与 $c^{\log Z}=\log\mathbb{E}[w]$，各配 LOO 版本。

**αTB 与 αKL。** 受 T04（EB-GFN，Zhang et al. 2022b）「先抽 Bernoulli 变量 $u\sim\mathcal{B}(\alpha)$ 决定从前向还是后向模型采轨迹」的做法启发，把这种混合采样下的期望 TB 梯度写成凸组合（式 (5)(6)），等价于优化

$$\mathcal{L}_{\alpha\mathrm{TB}}(\tau_F,\tau_B,\lambda) := \alpha\,\mathcal{L}_{\mathrm{TB}}(\tau_B,\lambda) + (1-\alpha)\,\mathcal{L}_{\mathrm{TB}}(\tau_F,\lambda), \quad \tau_F\sim Q,\ \tau_B\sim P;$$

对应的变分版本是 $\mathcal{L}_{\alpha\mathrm{KL}} = (1-\alpha)\mathcal{L}_{\mathrm{RKL}} + \alpha\mathcal{L}_{\mathrm{FKL}}$——非负、当且仅当 $P=Q$ 时为零，因此本身就是合法散度。$\alpha=0$ 与 $\alpha=1$ 两个端点分析清楚后，中间情形按线性组合即得。

## 理论结果

论文未以编号定理形式陈述，以下是 §4 逐情形推导出的结论（成立条件：$P_F$、$P_B$、$Z_\psi$ 按上述参数化，期望梯度意义）：

- **$\alpha=0$（从前向模型采样）、参数不共享**：TB 对 $\phi$ 的期望梯度不依赖 $\log Z_\psi$，且 $\frac{d}{d\phi}\mathcal{L}_{\mathrm{RKL}} = \frac12\mathbb{E}_{\tau\sim Q}[\frac{d}{d\phi}\mathcal{L}_{\mathrm{TB}}]$——on-policy TB 训前向模型 = 用带学习标度 $c_\psi = \log(Z/Z_\psi)$ 的 score-function 估计器优化 RKL。但对 $\theta$（后向模型）两者不等价：TB 梯度的被积函数比 RKL 多乘一个因子 $\log w + c_\psi$，即 TB 只在样本于 $P$ 下的似然「超出 $-c_\psi$ 预测的程度」时才推高/压低 $P$，而 RKL 一律推高 $P$ 对采到样本的似然；两者全局极小点相同（$P=Q$），优化动力学不同。
- **$\alpha=1$（从后向模型采样）、参数不共享**：对称结论。$\frac{d}{d\theta}\mathcal{L}_{\mathrm{FKL}} = \frac12\mathbb{E}_{\tau\sim P}[\frac{d}{d\theta}\mathcal{L}_{\mathrm{TB}}]$，即 TB 训后向模型 = 带学习标度的 FKL；对 $\phi$ 则差同样的乘子。
- **共享参数 $\eta=\phi=\theta$（§4.1）**：TB 期望梯度中前后向的 score 项以差的形式同时出现（$\frac{d}{d\eta}\log Q - \frac{d}{d\eta}\log P$ 一并乘上 $\log w + \log\frac{Z}{Z_\psi}$），与 RKL/FKL 的梯度结构不再重合，等价性失效。
- **$Z_\psi$ 的角色**：$\log(Z/Z_\psi)$ 在所有情形下都可解读为一个「学出来的控制变量标度」，其自身更新规则由 TB 对 $\psi$ 的梯度给出；这提示可以用 §3.1 的 LOO 控制变量替代学习的 baseline。

与 T08 Prop. 1 对照：$\phi$ 侧等价式两文相同；T08 进一步给出 on-policy TB 对 $P_B$ 参数的梯度等于 $D_{\log^2}$ 伪散度 + 修正项的显式表达，本文则停在「差一个乘子」的定性刻画，但补上了 $\alpha=1$ 端点与共享参数两种 T08 未覆盖的情形。

## 实验与证据

两个合成任务，公共设定：目标为 $\{0,1\}^D$ 上的离散分布，逐位构造状态（状态空间 $\{\emptyset,0,1\}^D\cup\{s_f\}$，每步把一个 $\emptyset$ 位填成 0/1）；$P_B$ 固定为「均匀选一个已置位、抹回 $\emptyset$」，$P_F$ 由 MLP 输出各位置的 Bernoulli logits。

**合成密度（2spirals、8gaussians）**：沿 Dai et al. 2020 与 T04 的协议，把二维连续密度离散成 $2^{16}$ 格并用 Gray code 编码成 32 位向量。两种模式：(1) 联合学习能量函数 $\xi$（contrastive divergence 风格，MH 链的 proposal 由 GFN 构造，即 T04 的 EB-GFN 框架）与 GFN；(2) 固定预训练能量函数只训 GFN。指标为测试数据的负对数似然（NLL，用式 (7) 的重要性采样估计边缘似然）。Table 1 结果（LRN = 学习 baseline）：

- $\alpha=0$：αTB 与 αKL 在一个标准差内持平（2spirals 20.163±0.013 vs 20.171±0.015），符合等价性预测。
- $0<\alpha<1$：两者相近，αTB 略优（2spirals $\alpha=0.5$：20.118±0.006 vs 20.145±0.008；8gaussians $\alpha=0.5$：19.995±0.008 vs 20.003±0.014）。
- $\alpha=1$：αTB 显著劣化（2spirals 20.994±0.037；fixed-$\xi$ 2spirals 21.230±0.029），αKL 保持稳定（对应 20.174±0.009 与 20.172±0.008）。此处两目标不等价（$\phi$ 侧差乘子），实验站在 αKL 一边。

**Ising 模型**：$\pi_T(s_T)\propto\exp(-\beta H(s_T))$，$H(s_T)=-\frac12 s_T^\top A_N s_T$（式 (8)），$A_N$ 为 $N\times N$ 周期边界格点邻接阵，$D=N^2$，$\beta\in\{\pm0.2,\dots,\pm1\}$ 共 10 档。无真实样本故只能 $\alpha=0$（此时 αTB 与 αKL 等价），于是专门比较「学习 baseline（LRN，即 $\log Z_\psi$）」与「LOO 控制变量 $\hat c^{\log Z}_s$」：以期望对数权重 $\mathbb{E}_{\tau\sim Q}[\log w] \le \log Z$ 为指标，Table 2 显示所有 $\beta$ 下两者差异不显著（如 $\beta=1$：LRN 174.262±23.712 vs LOO 190.531±19.436，重叠于误差带内）。定性上 GFN 样本与 MH 链样本相当（Fig. 2）。

## 与谁对话

- 与 T08 构成「同一等价性、两个社区视角」的对偶：T08 从 GFlowNet 出发向 VI 翻译（并覆盖 SubTB/DB↔NVI），本文从 VI 出发向 GFlowNet 翻译（并覆盖 α 谱系与共享参数）。两文互引，合起来构成 GFlowNet-VI 等价性的完整первичная文献。本文还引用 Zhang et al. 2022a（Unifying Generative Models with GFlowNets）的 HVAE↔GFN 对应作为第三条相关线。
- 上游：T02（流理论与马尔可夫流记号，本文 §2 的定义直接取自它）、T03（TB 目标）、T04（EB-GFN：α 混合采样的来源、合成密度实验协议、GFN 构造 MH proposal 的联合训练框架）、T05（partial-episode 目标，被引为 credit assignment 改进）。
- 下游：T17（用一般 $f$-散度与控制变量训练 GFlowNet 的系统研究）是本文「用 VI 方差缩减工具改造 GFN 目标」议程的直接展开；T31（用轨迹似然最大化优化后向策略）呼应本文识别出的「TB 训 $P_B$ ≠ FKL 训 $P_B$」缺口；T16（policy gradient 训练）与 T14/T15（熵正则 RL 等价）延续「GFN 目标 = 某种已知框架 + 特定估计器」的还原论路线。结论中点名的 variational SMC 与 NVI 扩展方向后来在扩散采样器一线（T20 及 GFLOWNET_DIFFUSION_SAMPLER_CN.md 覆盖的工作）被兑现。
- 本文的 $\alpha=1$ 负结果（αTB 从后向模型采样时性能崩）是后来「backward policy 设计」子领域（T25、T31）反复引用的动机证据之一。

## 局限与批判

- 等价性刻画停留在期望梯度层面，且「差一个乘子 $\log w + c_\psi$」的两种情形只给了定性讨论，没有量化这个乘子对收敛速度/方差的影响；T08 对同一对象给出了更完整的显式散度表达。
- 实验规模小：两个 32 维合成密度 + 最大 $D=N^2$ 的 Ising 格点，没有分子、序列或结构学习等 GFlowNet 主战场任务；「αKL 在 $\alpha=1$ 更稳」的结论未在真实任务上验证。
- $P_B$ 全程固定为均匀，θ 侧的不等价分析（本文理论上最有新意的部分之一）在实验里实际上没有被检验；共享参数情形同样只有推导没有实验。
- LOO 控制变量与学习 baseline 的比较结论是「无显著差异」，且 LOO 的最优标度版本需要逐样本梯度（自动微分框架下要 $S$ 次前反向传播），实践收益存疑——论文自己也只测了免梯度的 $c^{\log Z}$ 变体。
- Ising 实验用 $\mathbb{E}_Q[\log w]$（ELBO 型下界）作指标，对 mode coverage 不敏感，恰好回避了 RKL 系目标最受质疑的 mode collapse 维度。
- arXiv 版无代码，实验协议（网络宽度、学习率、MH 步数 $K$）在正文中披露不全，复现依赖 T04 的公开实现。

## 对后续研究的启示

- αKL/αTB 谱系说明「从哪个模型采样」与「优化哪个散度」是两个可独立调节的旋钮，$\alpha$ 本身可以退火或自适应——后续探索/利用权衡研究（T25、T35、T45）里的混合采样策略都可以放进这个框架描述。
- 「$\log Z_\psi$ = 学习的控制变量标度」与 T08 的 baseline 识别相互印证，且本文额外指出它可被 LOO 估计替换而不掉点：训练 GFlowNet 时 $\log Z$ 参数并非必需品，这为无 $Z$ 参数的目标（如后来的 VarGrad 型损失、T17 的散度族）铺路。
- $\alpha=1$ 端点的失败模式提示：给定后向轨迹时该优化的是 FKL 型目标而非原始 TB——这正是 T31「以轨迹似然最大化训 $P_B$」的出发点，也解释了为什么 sleep 阶段式训练在 GFlowNet 里要换损失。
- 共享参数分析提醒：GFN 实现里常见的「$P_F$、$P_B$ 共享主干」工程选择会破坏与 VI 的干净对应，理论比较时必须声明参数化方式。
- 把 GFlowNet 摆进 VI–SMC–MCMC 的既有分类学里，是后来综述（GFLOWNET_VI_MCMC_SMC_CN.md 所覆盖的谱系）组织材料的模板。
