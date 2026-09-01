# GFlowNet 与 Diffusion Sampler：从离散构造到连续路径测度

> 本文扩展理论指南 §6（连续/非无环）。
> 来源：GFlowNet 调研 2026-08 审查扩充（E04）。核心索引见 [README](README.md)。

---

> 并入说明：本节作为独立一级小节插入 §6《理论扩展》与 §7《经典文献脉络》之间，后续章节编号顺延；文内交叉引用按现行 guide 编号书写。
> 论文核实：2026-08-14 逐篇核对 arXiv 元数据、OpenReview/TMLR 记录与论文原文。

## GFlowNet 与 diffusion sampler：从离散构造到连续路径测度

前面各节把 GFlowNet 讲成“离散图上的流”。但同一套 \(P_F/P_B/Z/R(x)\) 记号，在把状态换成 \((x,t)\in\mathbb R^d\times[0,1]\)、把转移换成高斯核之后，就变成另一个社区所说的 **diffusion sampler**：训练一个神经 SDE，从简单先验分布把样本输运到只知道未归一化密度的目标分布。这条线在 2024–2026 年间从“两个孤立交叉点”长成了完整的理论桥，核心结论是：

1. 每一步为高斯核的离散 GFlowNet，就是生成 SDE 的 Euler–Maruyama 离散化对应的策略。
2. TB 残差恰是前向与反向**路径测度** Radon–Nikodym 导数的 log-ratio（外加 \(\log Z\) 偏移）。
3. 在步长趋于零的极限下，TB/SubTB/DB 这些离散目标分别收敛到路径测度散度与偏微分方程约束——这是[已发表的定理](https://arxiv.org/abs/2501.06148)，不再只是类比。

### 问题设定：两个社区在解同一个问题

目标与 §2 相同，只是空间换成连续的：给定能量 \(\mathcal E:\mathbb R^d\to\mathbb R\)，从

\[
p_{\mathrm{target}}(x)=\frac{R(x)}{Z},\qquad
R(x)=e^{-\mathcal E(x)},\qquad
Z=\int_{\mathbb R^d}R(x)\,\mathrm dx
\]

采样，只能查询 \(\mathcal E\)，没有目标样本。diffusion sampler 社区的做法是学习生成 SDE

\[
\mathrm dX_t=\mu_F(X_t,t)\,\mathrm dt+\sigma(t)\,\mathrm dW_t,\qquad X_0\sim p_0,
\]

使 \(X_1\sim p_{\mathrm{target}}\)。代表方法是 [PIS](https://arxiv.org/abs/2111.15141)（ICLR 2022，把采样写成 Schrödinger bridge / 随机最优控制问题）、[DDS](https://arxiv.org/abs/2302.13834)（ICLR 2023，固定 noising 参考过程再学其反转）与 [DIS](https://arxiv.org/abs/2211.01364)（TMLR 2024，统一的最优控制视角）。

GFlowNet 视角下，取时间格点 \(0=t_0<\cdots<t_N=1\)，状态为 \((x,t_n)\)，源状态 \(s_0\) 后第一步抽 \(x_0\sim p_0\)，Euler–Maruyama 给出前向策略

\[
P_F(x_{n+1}\mid x_n)=\mathcal N\!\big(x_{n+1};\,x_n+\mu_F(x_n,t_n)\,\Delta t_n,\ \sigma(t_n)^2\Delta t_n\,I\big),
\]

终止状态 \(x_N\) 以边流 \(F(x_N\to s_f)=R(x_N)\) 进汇点，\(P_B\) 是 noising 方向的转移核。这正是 §6.3 中 [Lahlou et al. 2023](https://proceedings.mlr.press/v202/lahlou23a.html) 连续 GFlowNet 框架的一个实例。注意方向约定：GFlowNet 的 \(P_F\) 指**生成方向**（噪声到样本），与 diffusion 建模文献把“加噪”称作 forward 恰好相反。

两套语言的对照：

| GFlowNet 概念 | diffusion sampler 对应物 |
|---|---|
| 前向策略 \(P_F\) | 生成 SDE 的 Euler–Maruyama 高斯核 |
| 反向策略 \(P_B\) | noising 过程转移核（DDS/DIS 中固定） |
| 状态流 \(F(s)\)（DB 中的中间量） | 时间边缘密度 \(p_t\) 的未归一化估计 |
| TB 残差 | 路径测度 log Radon–Nikodym 导数 |
| DB 约束的极限 | Fokker–Planck 方程 + Nelson 恒等式 |
| \(Z\)、\(R(x)\) | 配分函数、\(e^{-\mathcal E(x)}\) |

### 数学核心：TB 是路径测度 log-ratio 的二阶矩

记 \(\widehat{\mathbb P}\) 为前向过程的轨迹分布（含 \(x_0\sim p_0\)），\(\widehat{\mathbb Q}\) 为“从 \(p_{\mathrm{target}}\) 出发沿 \(P_B\) 倒走”的轨迹分布。§3.3 的 TB 约束在此写成

\[
Z_\theta\,p_0(x_0)\prod_{n=0}^{N-1}P_F(x_{n+1}\mid x_n)
=
R(x_N)\prod_{n=0}^{N-1}P_B(x_n\mid x_{n+1}),
\]

其残差正是两个轨迹分布的 Radon–Nikodym 导数加一个常数偏移：

\[
\delta_\theta(\tau)
=\log\frac{Z_\theta\,p_0(x_0)\prod_n P_F(x_{n+1}\mid x_n)}{R(x_N)\prod_n P_B(x_n\mid x_{n+1})}
=\log\frac{\mathrm d\widehat{\mathbb P}}{\mathrm d\widehat{\mathbb Q}}(\tau)+\log\frac{Z_\theta}{Z}.
\]

于是 TB loss \(\mathbb E_{\tau\sim\widehat{\mathbb W}}[\delta_\theta(\tau)^2]\) 是 log-RN 导数在任意全支持参考测度 \(\widehat{\mathbb W}\) 下的二阶矩散度；把 \(\log Z_\theta\) 换成批内均值就得到 log-variance（VarGrad）散度。这就是 §5.2 “log-ratio 回归在 off-policy 下保零点”性质的连续空间版本，也是 off-policy 训练不需要 importance weighting 的根源。

连续时间里同样的对象存在：设 \(\mathbb P,\mathbb Q\) 分别是生成 SDE 与“目标分布 + noising SDE”诱导的路径测度（\(C([0,1],\mathbb R^d)\) 上的测度），Girsanov 定理给出

\[
\log\frac{\mathrm d\mathbb P}{\mathrm d\mathbb Q}(X)
=\log\frac{p_0(X_0)}{p_{\mathrm{target}}(X_1)}
+\int_0^1\frac{\|\mu_B\|^2-\|\mu_F\|^2}{2\sigma^2}\,\mathrm dt
+\int_0^1\frac{\mu_F}{\sigma^2}\cdot\mathrm dX_t
-\int_0^1\frac{\mu_B}{\sigma^2}\cdot\mathrm d\overleftarrow X_t,
\]

连续 TB 目标即 \(D^{\mathbb W}_{\mathrm{TB}}(\mathbb P,\mathbb Q)=\mathbb E_{X\sim\mathbb W}\big[(\log\tfrac{\mathrm d\mathbb P}{\mathrm d\mathbb Q}(X))^2\big]\)。而“两个路径测度相等”的局部刻画是 Nelson 恒等式：

\[
\mu_B=\mu_F-\sigma^2\nabla\log p_t
\quad\text{且}\quad
\mathbb Q_1=\mathbb P_1 .
\]

**离散到连续的极限**由 [From Discrete-Time Policies to Continuous-Time Diffusion Samplers](https://arxiv.org/abs/2501.06148)（Berner, Richter, Sendera, Rector-Brooks, Malkin；TMLR 2026）严格化，两个关键命题：

- **全局**（Prop. 3.3）：离散 TB/log-variance 二阶矩散度在 \(\max_n\Delta t_n\to0\) 时收敛到连续时间路径测度散度。证明直觉很干净：Euler–Maruyama 下的离散 log-RN 恰是上述 Itô 积分的 Riemann 和（论文 Lemma B.3）。
- **局部**（Prop. 3.4）：DB 约束（带学习的中间密度 \(\widehat p_n\)，即 GFlowNet 的状态流）的极限是时间边缘所满足的 Fokker–Planck 方程与 Nelson 恒等式；单步 DB 的 ratio 本身就是子区间上 RN 导数的一步 Euler–Maruyama 近似。SubTB 介于两者之间，两端分别退化为 DB 与 TB。

成立条件要写清：噪声为加性、各向同性且只依赖时间的 \(\sigma(t)\)（此时 Euler–Maruyama 有强收敛阶 1）；drift 满足论文附录 B.1 的正则性假设；策略为条件高斯；相关测度绝对连续。还有一个容易踩的细节：**时间反转与离散化不交换**——高斯增量链的反转一般不再是高斯增量链，先离散再反转与先反转再离散只在极限下由 Nelson 恒等式重合。这正是“离散 GFlowNet 与连续 sampler 等价”必须写成渐近定理、而非换元恒等式的原因。

### 承重论文线：基准统一、渐近等价、潜空间摊销

- **[Diffusion Generative Flow Samplers（DGFS）](https://arxiv.org/abs/2310.02679)**（Zhang, Chen, Liu, Courville, Bengio；ICLR 2024）：最早把 GFlowNet 训练信号系统引入 diffusion sampler。在 PIS/DDS 只有终端信号、必须整条轨迹计算 loss 的地方，DGFS 额外参数化一个 “flow function”（即中间时间边缘的未归一化估计，对应状态流 \(F(s)\)），从而使用 §3.5 的 SubTB 式部分轨迹目标：中途即可获得学习信号、梯度方差下降、\(\log Z\) 估计更准。
- **[Improved Off-Policy Training of Diffusion Samplers](https://arxiv.org/abs/2402.05098)**（Sendera et al.；NeurIPS 2024）：统一代码库与评测协议，在同一框架下公平比较 simulation-based 变分方法（PIS/DDS 的 on-policy KL）与 off-policy 目标（TB/SubTB/DB/log-variance），指出既往论文间评测不可比、部分结论存疑；并提出目标空间 local search + replay buffer 的探索策略。方法论意义：这条线的经验主张从此有了共同基准。
- **[From Discrete-Time Policies to Continuous-Time Diffusion Samplers](https://arxiv.org/abs/2501.06148)**（TMLR 2026）：即上一小节的渐近等价定理；推论是训练时可用远比推断粗、且非均匀的时间离散化，以少量能量求值达到相近性能——把“GFlowNet 目标与连续路径测度目标是同一件事”从口号变成定理，并直接产生训练加速。
- **[Amortizing Intractable Inference in Diffusion Models（RTB）](https://arxiv.org/abs/2405.20971)**（Venkatraman et al.；NeurIPS 2024）：把这套机制用于 **diffusion 先验微调**。给定先验 \(p_\theta\) 与约束 \(r\)，relative TB 约束

\[
Z_\phi\,p^{\mathrm{post}}_\phi(x_0\to\cdots\to x_1)=r(x_1)\,p_\theta(x_0\to\cdots\to x_1)\ \ \forall\tau
\ \Longrightarrow\
p^{\mathrm{post}}_\phi(x_1)\propto p_\theta(x_1)\,r(x_1),
\]

  即后验 sampler 的正确性（论文证明其渐近无偏）。与 TB 不同，RTB 比较的是**两个生成方向过程**的比值，但可严格解释为“以先验路径测度为参考测度的 TB 特例”（论文 §2.4）；loss 不需要对采样过程反传，因此天然支持 off-policy（从高密度样本倒走 noising 轨迹、replay buffer），这是它相对 on-policy RL 微调在模式覆盖上的优势来源。视觉、语言、控制上均有实验；另有预印本（[arXiv:2509.01632](https://arxiv.org/abs/2509.01632)，2025）声称 RTB 等价于 Trust-PCL，读时应核对两边路径分布与边界条件。
- **[Outsourced Diffusion Sampling](https://arxiv.org/abs/2502.06999)**（Venkatraman, Hasan et al.；ICML 2025）：把“可采样的先验 + 黑箱约束”推广到**任意**能写成外生噪声确定性变换 \(x=f_\theta(z)\) 的生成模型（VAE、GAN、normalizing flow、latent diffusion 等）：在噪声空间 \(z\) 训练 diffusion sampler 采 \(p(z\mid y)\propto p_z(z)\,r(f_\theta(z),y)\)，因为噪声空间后验往往比数据空间更光滑、维度更低。应用覆盖条件图像生成、RLHF 与蛋白质结构生成。理论上仍是同一个未归一化密度采样问题——目标能量换成了 \(-\log p_z(z)-\log r(f_\theta(z),y)\)。

一句话总结这条线的分工：DGFS 给出**信用分配**方案，NeurIPS 2024 基准给出**可比的经验学**，TMLR 2026 给出**离散—连续等价定理**，RTB 与 Outsourced 给出**规模化应用形态**（先验微调与潜空间后验）。

### 与 Schrödinger bridge（O04）的衔接

§2.6 说过：边界条件（源分布与 \(R\)）不唯一决定内部流。连续设定下同样如此——同时学 \(\mu_F\) 与 \(\mu_B\) 时，满足“前后向路径测度一致 + 终端边缘正确”的解构成一个连续族。[TMLR 2026 论文](https://arxiv.org/abs/2501.06148)明确列出三种恢复唯一性的方式，恰好对应三个方法家族：

1. **固定 \(P_B\)**（noising 过程给定）：DDS、DIS、RTB 与 diffusion 微调的默认选择——对应 guide 中“固定 \(P_B\) 则 flow 唯一”的离散结论；
2. **加参考过程正则**：即 Schrödinger (half-)bridge——在所有满足边界约束的路径测度中，选取相对参考过程 KL 最小的那一个；PIS 本身就建立在 Schrödinger bridge 问题上；
3. **固定中间边缘**：annealing 型方法（如 controlled Monte Carlo diffusions，ICLR 2024）。

Schrödinger bridge 是熵正则最优传输的动态形式，因此这一衔接与 §5.5 的 OT 结论同属一个模式：**reward matching 只规定“到哪里”，附加原则（最小总流 / 最小 KL / 固定退火路径）负责规定“怎么去”**。资源目录 O04 的 [Schrödinger Bridge Flow for Unpaired Data Translation](https://arxiv.org/abs/2409.09347)（NeurIPS 2024 Spotlight）给出无需反复重训的 SB 求解流程，但注意问题设定不同：O04 两端都是数据分布（翻译问题），本节是“一端简单先验、一端未归一化密度”（采样问题）。二者的衔接是结构性的（同一“消除内部自由度”家族），目前**没有**定理说 GFlowNet 训练等价于求解某个 SB；把 TB 换成带参考动力学与路径熵的 balance 目标，仍是开放方向（见 OT 分析文档 §3.4）。

### 声明纪律：哪些已证明，哪些是直觉

**论文证明**（引用时可作定理使用，注意各自条件）：

- 离散 TB 残差 = 路径测度 log-RN 导数 + \(\log(Z_\theta/Z)\)：定义层面的恒等式（Foundations 与 Lahlou et al. 2023 的连续框架；TMLR 2026 论文式 (4)(7)）。
- 离散目标 → 连续目标的收敛（Prop. 3.3/3.4 与 Lemma B.3，TMLR 2026）：条件为加性时变噪声、正则 drift、条件高斯策略。
- RTB 约束成立 ⟹ 后验边缘正确，且 RTB 是参考测度意义下的 TB（NeurIPS 2024）。
- DGFS 的正确性继承连续 GFlowNet 框架的测度论条件（ICLR 2024 + ICML 2023）。

**直觉与经验规律**（不要当定理引用）：

- “GFlowNet 就是 diffusion sampler”：仅在特定 MDP 构造、高斯核与步长趋零极限下严格；有限步长时先反转再离散与先离散再反转并不重合。
- “粗离散化训练不掉点”：TMLR 2026 的实验规律，有理论动机但无逐点保证。
- “off-policy + local search 优于 on-policy KL”：NeurIPS 2024 基准的经验结论，依赖目标分布与调参。
- “与 SB/OT 的衔接”：结构对应而非等价定理（§5.5 的 OT 等价另有精确条件）。

| 论文 | arXiv | venue |
|---|---|---|
| Path Integral Sampler（PIS） | 2111.15141 | ICLR 2022 |
| Denoising Diffusion Samplers（DDS） | 2302.13834 | ICLR 2023 |
| Optimal control perspective（DIS） | 2211.01364 | TMLR 2024 |
| Diffusion Generative Flow Samplers（DGFS） | 2310.02679 | ICLR 2024 |
| Improved Off-Policy Training of Diffusion Samplers | 2402.05098 | NeurIPS 2024 |
| Amortizing Intractable Inference in Diffusion Models（RTB） | 2405.20971 | NeurIPS 2024 |
| From Discrete-Time Policies to Continuous-Time Diffusion Samplers | 2501.06148 | TMLR 2026 |
| Outsourced Diffusion Sampling | 2502.06999 | ICML 2025 |
| Schrödinger Bridge Flow for Unpaired Data Translation（O04） | 2409.09347 | NeurIPS 2024 Spotlight |

