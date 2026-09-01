# N029 · 用学习式扩散改进采样：路径空间统一框架与 log-variance 损失

> **Improved sampling via learned diffusions**
> 作者：Lorenz Richter (Zuse Institute Berlin / dida), Julius Berner (Caltech)，两人同等贡献 · ICLR 2024 · [arXiv](https://arxiv.org/abs/2307.01198) · [代码](https://github.com/juliusberner/sde_sampler)

## 一句话

把 PIS、DIS、DDS、Schrödinger bridge 四类扩散采样器统一成"路径空间测度 (path space measure) 上时间反转扩散之间的散度最小化"这一个广义桥问题，并用 log-variance 散度替换 reverse KL，一举解决 mode collapse、需要对 SDE 求解器求导、梯度方差大三个痛点。

## 1. 要解决的问题

从未归一化密度采样：给定 \(\rho:\mathbb{R}^d\to[0,\infty)\)，目标是采样

\[
p_{\text{target}} := \frac{\rho}{Z},\qquad Z := \int_{\mathbb{R}^d}\rho(x)\,dx,
\]

其中 \(Z\) 不可解、且**没有** \(p_{\text{target}}\) 的样本（这是与生成建模的本质区别）。此前的扩散式采样器——Path Integral Sampler（PIS，Zhang & Chen 2022）、Time-Reversed Diffusion Sampler（DIS，Berner et al. 2024）、Denoising Diffusion Sampler（DDS，Vargas et al. 2023a）——各自独立推导、彼此关系不清，且全部依赖 reverse KL 目标，因此共有三个缺陷：(i) reverse KL 的 mode collapse；(ii) 训练需对 SDE 求解器反向传播，代价高；(iii) Monte Carlo 梯度方差大、在最优解附近不消失。

## 2. 核心方法

**双向受控 SDE。** 引入两个受控过程（原文式 (2)(3)）：

\[
dX^u_s = (\mu + \sigma u)(X^u_s, s)\,ds + \sigma(s)\,dW_s,\qquad X^u_0 \sim p_{\text{prior}},
\]
\[
dY^v_s = (-\overleftarrow{\mu} + \overleftarrow{\sigma}\overleftarrow{v})(Y^v_s, s)\,ds + \overleftarrow{\sigma}(s)\,dW_s,\qquad Y^v_0 \sim p_{\text{target}},
\]

其中 \(u,v\in\mathcal{U}\) 是待学的控制函数，\(\overleftarrow{\mu}(t):=\mu(T-t)\) 表示时间反转。目标是让 \(Y^v\) 恰为 \(X^u\) 的时间反转（Problem 2.2）：

\[
u^*, v^* \in \arg\min_{u,v\in\mathcal{U}\times\mathcal{U}} D\big(\mathbb{P}_{X^u}\,\big|\,\mathbb{P}_{\overleftarrow{Y}^v}\big),
\]

\(D\) 是路径空间测度之间的任意散度——这是本文最大的自由度：此前方法全部锁死在 \(D=D_{\mathrm{KL}}\)。

**路径测度似然（Proposition 2.3）。** 沿任意第三条过程 \(X^w\) 评估 Radon-Nikodym 导数：

\[
\frac{d\mathbb{P}_{X^u}}{d\mathbb{P}_{\overleftarrow{Y}^v}}(X^w) = Z\,\exp\Big(\mathcal{R}_{f^{\text{Bridge}}_{u,v,w}} + \mathcal{S}_{u+v} + B\Big)(X^w),
\]

其中 \(\mathcal{R}_f(X):=\int_0^T f(X_s,s)ds\) 是路径积分，\(\mathcal{S}_w(X):=\int_0^T w(X_s,s)\cdot dW_s\) 是随机积分，终端项 \(B(X^w):=\log\frac{p_{\text{prior}}(X^w_0)}{\rho(X^w_T)}\)，运行代价

\[
f^{\text{Bridge}}_{u,v,w} := (u+v)\cdot\Big(w + \frac{v-u}{2}\Big) + \nabla\cdot(\sigma v - \mu).
\]

证明用 Girsanov 定理 + Itô 引理 + \(\log \overleftarrow{p}_{Y^v}\) 满足的 HJB 方程。取 \(D=D_{\mathrm{KL}}\)（此时 \(w=u\)，随机积分期望为零）得原文式 (7) 的 KL 损失，恰是 Chen et al. (2021a) 桥问题目标的采样版。

**log-variance 散度（Definition 2.4）。** 核心提案：

\[
D^{\widetilde{\mathbb{P}}}_{\mathrm{LV}}(\mathbb{P},\mathbb{Q}) := \mathbb{V}_{\widetilde{\mathbb{P}}}\Big[\log\frac{d\mathbb{P}}{d\mathbb{Q}}\Big],
\]

即在参考测度 \(\widetilde{\mathbb{P}}\)（实践中取 \(\mathbb{P}_{X^w}\)）下取对数似然比的方差。由方差的平移不变性，\(\log Z\) 自动消去——不需要知道归一化常数。得到原文式 (8)：

\[
\mathcal{L}^w_{\mathrm{LV}}(u,v) = \mathbb{V}\Big[\mathcal{R}_{f^{\text{Bridge}}_{u,v,w}} + \mathcal{S}_{u+v} + B(X^w)\Big].
\]

三个实际好处：(1) 参考控制 \(w\) 与被优化的 \(u\) 解耦，可自由选 \(w\) 或初始分布做探索（off-policy），对抗 mode collapse；(2) 不需要对 \(X^w\) 关于 \(w\) 求导——SDE 模拟可以 detach，不用穿过求解器反传，也不需要 \(\nabla\log\rho\)（目标可以是黑箱）；(3) sticking-the-landing 性质（见 §3）。

**统一已有方法（Table 1）。** 由 Girsanov 定理，式 (7)(8) 的解只被约束到 Nelson 恒等式（原文式 (9)）：

\[
u^* + v^* = \sigma^\top \nabla\log p_{X^{u^*}},
\]

因此有无穷多解；各已有方法即通过固定 \(v\) 选出唯一解：
- **SB**：在所有满足 (9) 的解中选最小化 \(D_{\mathrm{KL}}(\mathbb{P}_{X^u}|\mathbb{P}_{X^r})\)（对参考过程 \(X^r\) 的熵约束，原文式 (11)(12)）者，即熵正则 OT；
- **DIS**：取 \(\bar v = 0\)、\(p_{\text{prior}} \approx p_{Y^0_T}\)（VP-SDE 噪声化的时间反转，即 denoising diffusion 采样版）；
- **PIS**：取 \(r=0\)、\(p_{\text{prior}}=\delta_{x_0}\)，Doob h-transform 保证半桥 (half-bridge) 精确可解；
- **DDS**：取 \(r=\sigma^\top\nabla\log\overleftarrow{p}_{Y^{0,\text{ref}}}\)、\(\bar v=0\)，VP-SDE 的不变分布做参考。

每种方法都立即获得对应的 log-variance 版本。

## 3. 理论结果

- **Proposition 2.3**（路径测度似然）：如上，给出统一 RND 表达式；附录 Remark A.1 给出用 backward 随机积分消去散度项 \(\nabla\cdot(\sigma v-\mu)\) 的散度自由 (divergence-free) 版本，高维时避免 Hutchinson 估计的额外方差。
- **Proposition 2.5**（最优解处的鲁棒性 / sticking-the-landing）：log-variance 损失的 Monte Carlo 梯度估计在最优解 \((u^*,v^*)\) 处方差为零，对 \(\theta\)（参数化 \(u\)）和 \(\gamma\)（参数化 \(v\)）都成立、对任意参考 \(w\) 成立；KL 损失不具备该性质，梯度下降会在最优解附近震荡。
- **Remark A.2**（控制变元解释）：\(w=u\) 时 log-variance 梯度 = KL 梯度加 batch 内局部 baseline 控制变元，解释其方差缩减；把方差换成二阶矩并额外学一个 \(\log Z\) 估计，即得 (second) moment loss——**原文明确指出这就是 trajectory balance 目标（Malkin et al., 2022a）**。
- 唯一性分析：一般桥目标 (7)(8) 无唯一解（只约束到 Nelson 恒等式），这解释了 SB 类双控制训练的不稳定；固定 \(v=\bar v\) 后近似误差为不可约损失（原文式 (13)）。

## 4. 实验与证据

三个基准：GMM（\(d=2\)，9 个分离模式）、Funnel（\(d=10\)，\(\eta=3\)）、double well（DW，\(d=5, m=5, \delta=4\) 与 \(d=50, m=5, \delta=2\)）。指标：\(\Delta\log Z\)、Sinkhorn 距离 \(\mathcal{W}^2_\gamma\)、归一化 ESS、边缘标准差误差 \(\Delta\)std；5 个独立 run 取中位数，KL 与 LV 用完全相同的超参和目标评估次数。

Table 2 关键数字（KL → LV）：
- GMM PIS：\(\Delta\log Z\) 1.094 → 0.046，ESS 0.0051 → 0.9093；GMM DIS：1.551 → 0.056。KL 版本只覆盖 \(p_{\text{prior}}\) 附近一个模式（Figure 2），LV 恢复全部 9 个模式。
- DW(\(d=5\)) PIS：\(\Delta\log Z\) 3.567 → 0.214；DIS：1.462 → 0.375。
- Funnel 上两者接近（0.288 vs 0.277），说明改进主要来自多模式场景。

附录扩展：\(d=1000\) 的 shifted DW 仍保持模式覆盖（Figure 7）；batch size 512 的小批量场景 LV 优势更明显（Table 4，方差缩减的直接推论）；40 模式 GMM（Midgley et al. 2023 基准）上，利用 LV 独有的子轨迹训练（partial trajectory optimization，受 DGFS 即 Zhang et al. 2023 启发）在**不用 \(\nabla\log\rho\)** 的条件下恢复全部 40 个模式，而 KL-DIS 和 LV-DIS 全轨迹版都塌缩（Figure 4）；与 CRAFT 对比（Table 6）各有胜负、总体可比。LV 还降低了每步梯度耗时（不穿求解器反传，Figure 9）。

## 5. 与 GFlowNet 生态位的关系

这篇论文是"扩散采样器 ↔ GFlowNet"翻译词典的官方版本之一，关系是**同一问题的连续时间镜像，方法论互相输血**：

- **损失对应关系是精确的**：log-variance loss = VarGrad（Richter et al. 2020）= TB 的局部 baseline 变体；原文 Remark A.2 直说 second moment loss + 学习 \(\log Z\) 就是 trajectory balance（并引 Malkin et al. 2022a/b）。GFlowNet 社区的 DGFS（Zhang et al. 2023）反过来启发了本文的子轨迹训练。也就是说 TB/SubTB 与 LV/moment loss 在连续 SDE 语境下是同一个家族，谁引用谁已经是双向的。
- **共享的核心卖点**：两边都做"只给未归一化 \(R(x)\) 或 \(\rho(x)\)、无样本"的 amortized 采样，都靠平移不变（TB 里学 \(Z\)、LV 里方差消去 \(Z\)）绕开归一化常数，都支持 off-policy 探索。本文的探索机制（自由选 \(w\) 和子轨迹起点）就是 GFlowNet off-policy 训练的连续版。
- **GFlowNet 的独特能力**：(i) 原生处理**离散/组合空间**（分子图、序列），本文框架完全绑定在 \(\mathbb{R}^d\) 上的 SDE；(ii) DAG 结构允许多条路径到同一终态并做 flow 级 credit assignment（DB/SubTB），扩散采样器的轨迹空间没有这种共享子结构；(iii) 显式学出的 \(Z_\theta\) 可直接用于下游（如 Bayesian 模型比较）。
- **GFlowNet 的劣势也被照出来**：连续 GFlowNet 理论（Lahlou et al. 2023）在实践上不如本文的 SDE 框架成熟；本文对散度选择、梯度方差、sticking-the-landing 的分析比 GFlowNet 文献里对 TB 方差的处理更系统——GFlowNet 一侧的梯度方差研究（如 VarGrad 视角）大量直接借用这条线的结果。
- **定位判断**：竞争面在"连续空间从能量函数采样"这一 benchmark 生态（GMM/Funnel/DW 也是 GFlowNet 扩散采样论文的标准靶子）；互补面在于它提供了 GFlowNet 损失设计可以直接吸收的散度工具箱（任意路径散度、参考测度选择、divergence-free 技巧）。

## 6. 局限与批判

- 一般桥（双控制 \(u,v\) 同时学）仍不稳定，作者自己承认并归因于解的非唯一性，主实验退回到固定 \(v\) 的 PIS/DIS——"广义 SB 框架"的完全体没有被真正驯服。
- log-variance 损失的参考测度 \(w\) 的选择只试了 \(w=u\)（detach），探索潜力（噪声化 \(u\)、别的初始分布）留给未来；子轨迹训练里 \(X^w_t\sim\mathrm{Unif}([-a,a]^d)\) 需要知道支撑集大小 \(a\)，高维下这种均匀探索会失效。
- Funnel 上 LV 无优势，说明改进集中在"分离多模式"这一特定失败模式；对高度病态几何（如 Neal funnel 的尺度分层）没有新办法。
- 基准全部是合成低维目标（最高 \(d=1000\) 但结构是可分解的 DW），没有真实 Bayesian 后验或分子体系；与 MCMC/SMC 的系统对比被明确推给引文。
- 方差型损失对 batch 内样本相关性敏感，batch 小、似然比重尾时方差估计本身高方差；论文用 2048 的大 batch，小 batch 表（Table 4）里 GMM-PIS 两种损失都失败（ESS 0.0002），说明 LV 不是万灵药。

## 7. 对后续研究的启示

- 散度选择是被低估的设计维度：把"选散度"从"选方法"里解耦后，任何路径空间采样器（包括离散 GFlowNet）都可以问一句"换成 log-variance/moment/其他 f-divergence 会怎样"。GFlowNet 一侧对应的问题是：TB 的 batch-Z 版本（相当于局部 baseline）vs 学习 Z 版本的系统比较。
- 子轨迹 + 任意起点的训练范式（只在 LV 类损失下合法）为 GFlowNet 的 SubTB/DGFS 提供了连续版理论背书，也提示反向移植：离散 GFlowNet 里用"从高维均匀分布或缓冲区重放采起点"的探索策略有同样的合法性基础。
- 无导数（derivative-free）采样场景（黑箱 \(\rho\)、不可微 reward）是扩散采样器与 GFlowNet 共同的下一战场，本文 Table 3 的初步结果给出了可行性证据。
- 后续工作序列（Sendera et al. 2024 的 benchmark、Vargas et al. 2024 的 controlled Monte Carlo diffusions）都以本文为基线；读 GFlowNet 扩散采样文献时应把本文当作"KL vs 方差型损失"这条对照轴的原点。
