# GFN ≡ 熵正则 RL：完整推导

> 本文深化理论指南 §5.1。
> 来源：GFlowNet 调研 2026-08 审查扩充（E08）。核心索引见 [README](README.md)。

---

> 本小节把 §5.1 的定性结论展开为可自学的完整推导。记号沿用 guide §2：\(P_F,P_B,F,Z,R,s_0,s_f\)，\(\mathcal X=\operatorname{Par}(s_f)\)，轨迹 \(\tau=(s_0\to s_1\to\cdots\to x\to s_f)\)；除 D 节 modified DB 处（全状态可终止）外，沿用 §3 约定 \(F(x)=R(x)\)、\(P_F(s_f|x)=1\)。
> 定理编号已于 2026-08-14 对照 PMLR 正式版核实：Tiapkin et al.（AISTATS 2024, PMLR 238:4213–4221）的 Theorem 1、Proposition 1、Remarks 1–4；Deleu et al.（UAI 2024, PMLR 244:997–1021）的 Theorem 3.1、Proposition 3.2、Corollary B.1、Proposition 3.3。Deleu 原文用能量 \(E\) 与温度 \(\alpha\)：本节统一取 \(\alpha=1\)、\(E(x)=-\log R(x)\) 译回 guide 记号。

## A. 从 GFN 的 DAG 构造 soft MDP

状态取 DAG 节点加吸收态：\(\mathcal S'=\mathcal S\cup\{s_f\}\)。动作就是出边：在 \(s\in\mathcal S\setminus\mathcal X\) 处 \(\mathcal A_s=\operatorname{Child}(s)\)；每个 \(x\in\mathcal X\) 只有唯一动作 \(s_f\)（对应约定 \(P_F(s_f|x)=1\)）；\(s_f\) 上是零奖励自环。转移确定：选哪条边就到达该边终点。DAG 有限无环，回合必然终止，取折扣 \(\gamma=1\) 不会发散。一句话：**状态 = GFN 的 DAG 节点，动作 = 边，\(\gamma=1\)，熵系数 \(\lambda=1\)**。

熵正则（soft）值函数定义为

\[
V_\lambda^\pi(s)=\mathbb E_\pi\Big[\sum_{t\ge 0}\big(r(s_t,s_{t+1})+\lambda\,\mathcal H(\pi(\cdot|s_t))\big)\,\Big|\,s_0=s\Big],
\]

其最优解满足 soft Bellman 方程（Haarnoja et al. 2017, Theorems 1–2；确定性转移下写成状态-后继形式）：

\[
Q_\lambda^\star(s,s')=r(s,s')+V_\lambda^\star(s'),\qquad
V_\lambda^\star(s)=\lambda\log\!\!\sum_{s'\in\operatorname{Child}(s)}\!\!\exp\big(Q_\lambda^\star(s,s')/\lambda\big),\qquad
\pi_\lambda^\star(s'|s)=\exp\big(\tfrac1\lambda(Q_\lambda^\star(s,s')-V_\lambda^\star(s))\big).
\]

\(\lambda\to0\) 时 LogSumExp 退化为 max，回到普通 Bellman 方程。下面固定 \(\lambda=1\)。

## B. 核心定理与逐步推导（Tiapkin et al. 2024, Theorem 1）

给定固定反向策略 \(P_B\) 与 GFN 奖励 \(R\)，把 MDP 奖励设计为

\[
r(s\to s')=\log P_B(s|s')\ \ (s\notin\mathcal X\cup\{s_f\}),\qquad
r(x\to s_f)=\log R(x)\ \ (x\in\mathcal X),\qquad
r(s_f\to s_f)=0.
\]

**定理（Tiapkin et al., Theorem 1）**：令 \(F\) 为由 \((P_B,R)\) 唯一确定的 Markovian flow（guide §2.6；设 \(R>0\)、\(P_B\) 在所有 DAG 边上为正，保证各 \(\log\) 有定义）。则上述 soft MDP 在 \(\lambda=1\) 下的最优策略满足 \(\pi_1^\star(s'|s)=P_F(s'|s)\)（\(s\in\mathcal S\setminus\mathcal X\)；\(x\in\mathcal X\) 处两侧都只有 \(s_f\) 一个动作，平凡成立），且对一切 \(s\ne s_f,\ s'\ne s_f\)：\(V_1^\star(s)=\log F(s)\)、\(Q_1^\star(s,s')=\log F(s\to s')\)。

推导按 DAG 拓扑序**反向归纳**，四步：

**第 0 步（吸收态边界）**：\(s_f\) 只有零奖励自环，故 \(V_1^\star(s_f)=0\)。

**第 1 步（终端基例）**：\(x\in\mathcal X\) 只有动作 \(s_f\)，单动作时 LogSumExp 退化为恒等：

\[
Q_1^\star(x,s_f)=\log R(x)+V_1^\star(s_f)=\log R(x),\qquad
V_1^\star(x)=\log\exp Q_1^\star(x,s_f)=\log R(x)=\log F(x).
\]

**第 2 步（Bellman 备份 = \(P_B\) 的定义式）**：设 \(s\) 的所有孩子 \(s'\) 已满足 \(V_1^\star(s')=\log F(s')\)。代入奖励设计与 guide §2.4 的 \(P_B(s|s')=F(s\to s')/F(s')\)：

\[
Q_1^\star(s,s')=\log P_B(s|s')+V_1^\star(s')
=\log\frac{F(s\to s')}{F(s')}+\log F(s')=\log F(s\to s').
\]

**第 3 步（LogSumExp = 出边流守恒）**：

\[
\exp V_1^\star(s)=\sum_{s'\in\operatorname{Child}(s)}\exp Q_1^\star(s,s')
=\sum_{s'\in\operatorname{Child}(s)}F(s\to s')=F(s),
\]

最后一个等号正是 guide §2.3 的流守恒。归纳完成；特别地 \(V_1^\star(s_0)=\log F(s_0)=\log Z\)。

**第 4 步（softmax 策略 = 归一化出边流）**：

\[
\pi_1^\star(s'|s)=\exp\big(Q_1^\star(s,s')-V_1^\star(s)\big)=\frac{F(s\to s')}{F(s)}=P_F(s'|s).\qquad\blacksquare
\]

逐行对照可见，**soft Bellman 方程恰是 flow 递推取了对数**：

| soft RL 侧 | GFN 侧 |
|---|---|
| \(Q_1^\star(s,s')\)、\(V_1^\star(s)\)、\(V_1^\star(s_0)\) | \(\log F(s\to s')\)、\(\log F(s)\)、\(\log Z\) |
| 备份 \(Q=r+V'\)，其中 \(r=\log P_B\) | 边流分解 \(F(s\to s')=F(s')P_B(s|s')\) |
| \(V=\operatorname{LogSumExp}(Q)\) | 出边守恒 \(F(s)=\sum_{s'}F(s\to s')\) |
| softmax 策略 \(\exp(Q-V)\) | \(P_F(s'|s)=F(s\to s')/F(s)\) |

值函数还有全局解释（Tiapkin Proposition 1）：任意策略 \(\pi\) 的轨迹分布 \(q^\pi\) 满足 \(V_1^\pi(s_0)=\log Z-\operatorname{KL}\big(q^\pi\,\|\,R(x)P_B(\tau|x)/Z\big)\)，即 soft RL 的策略提升就是在缩小与 guide §5.2 那个反向目标轨迹分布的 KL；再由数据处理不等式，\(V_1^\star(s_0)-V_1^\pi(s_0)=\operatorname{KL}(q^\pi\|\cdot)\ge\operatorname{KL}(P_T^\pi\|R/Z)\)：策略误差直接控制终止分布误差。

## C. 树的退化与多路径 DAG 的 \(n(x)\) 偏置

先推一个引擎恒等式（取自 Deleu Theorem 3.1 的证明，式 (19)–(25) 的 telescoping）：确定性 soft MDP、\(\lambda=1\) 时，

\[
\pi^\star(\tau)=\prod_t \exp\big(Q_1^\star(s_t,s_{t+1})-V_1^\star(s_t)\big)
=\exp\Big(\sum_t r(s_t,s_{t+1})-V_1^\star(s_0)\Big),
\]

因为代入 \(Q_1^\star=r+V_1^\star(\text{后继})\) 后相邻的 \(V_1^\star\) 望远镜相消，只剩 \(V_1^\star(s_f)=0\) 与 \(-V_1^\star(s_0)\)。**最优轨迹概率只由沿途奖励和决定。**

**朴素 MaxEnt RL 的偏置（本节补充推导；结论由 Bengio et al. 2021 提出，Deleu Fig. 1 给出实例）**：若不做修正、只给稀疏终端奖励 \(r(x\to s_f)=\log R(x)\)、中间奖励取 0，则每条到 \(x\) 的轨迹奖励和都等于 \(\log R(x)\)，与走哪条路径无关。记 \(n(x)\) 为 \(s_0\) 到 \(x\) 的完整轨迹条数，对引擎恒等式求边际：

\[
\pi^\star(x)=\sum_{\tau\to x}\pi^\star(\tau)
=\frac{n(x)\,R(x)}{\sum_{x'}n(x')\,R(x')}\;\propto\;n(x)\,R(x),
\]

且 \(\exp V_1^\star(s_0)=\sum_{x}n(x)R(x)\ne Z\)。偏置把质量推向"能以更多顺序生成"的对象；这正是 §5.1 所说路径数偏置在 \(\lambda=1\)、稀疏奖励下的精确形式。

**树的退化**：树中每个非根节点只有一个父节点，\(P_B(s|s')\equiv1\Rightarrow\log P_B=0\)，修正奖励与稀疏奖励重合；同时 \(n(x)\equiv1\)，朴素形式本来就无偏。两个视角一致（Tiapkin Remark 1，由此重现 Bengio et al. 2021 Proposition 1(a,b)）。

**修正奖励的一般充分条件（Deleu Theorem 3.1，把 Tiapkin 的逐边修正推广到轨迹级）**：只要奖励沿每条完整轨迹满足

\[
\sum_{t}r(s_t,s_{t+1})=\log R(x)+\sum_{t=1}^{n_\tau}\log P_B(s_{t-1}|s_t)
\]

（第二个和取遍 DAG 内部边、\(s_{n_\tau}=x\)），代回引擎恒等式得 \(\pi^\star(\tau)=R(x)P_B(\tau|x)\big/\exp V_1^\star(s_0)\)；对 \(\tau\to x\) 求和并用 \(\sum_{\tau\to x}P_B(\tau|x)=1\)（guide §3.3 用过的归一化）即得 \(\pi^\star(x)=R(x)/Z\)、\(\exp V_1^\star(s_0)=Z\)。逐边分配 \(\log P_B\) 只是满足该约束的方案之一，轨迹级约束甚至兼容非 Markov 的 \(P_B\)。还有一个值得记住的特例：若所有对象的路径数相同（如 \(n\) 个变量按任意顺序赋值的因子图，\(n(x)\equiv n!\)），常数偏置在归一化中消去，朴素 MaxEnt RL 也无偏——"多路径必偏"须以 \(n(x)\) 非常数为前提（Deleu §6 的讨论）。

## D. 算法层对应：哪个 GFN 目标是哪个 MaxEnt RL 算法

| GFN 目标 | MaxEnt RL 算法 | 精确关系（\(\alpha=1\)；一般差常数 \(\alpha^2\)） |
|---|---|---|
| TB | PCL（Nachum et al. 2017） | Deleu Prop 3.2 取完整轨迹的特例：逐样本残差互为相反数 |
| SubTB | PCL（子轨迹 rollout） | Deleu Prop 3.2：\(\mathcal L_{\mathrm{PCL}}=\mathcal L_{\mathrm{SubTB}}\)，对应 \(V_{\mathrm{soft}}(s)=\log F(s)\)、\(\pi_\theta=P_{F,\theta}\) |
| DB | SQL / SoftDQN | Deleu Corollary B.1；Tiapkin §3.4：DB 即 dueling 架构 SoftDQN（值流+优势流，LogSumExp 归一） |
| modified DB | \(\pi\)-SQL（策略参数化 SQL） | Deleu Prop 3.3：要求全状态可终止，奖励整形 \(r(s\to s')=\log\tfrac{R(s')}{R(s)}+\log P_B(s|s')\)、终止奖励 0 |
| TB 的 on-policy 期望梯度 | policy gradient（\(\log Z_\theta\) 作基线） | Tiapkin §3.4 + Malkin et al. 2023：\(-\nabla_\theta V_1^{\pi_\theta}(s_0)=\tfrac12\,\mathbb E_{\tau\sim\pi_\theta}[\nabla_\theta\mathcal L_{\mathrm{TB}}(\tau)]\) |

TB↔PCL 只需一次代入（本节补充推导；一般子轨迹情形见 Deleu App. B.1）。PCL 残差在完整轨迹上、以 \(V_{\mathrm{soft}}(s_f)=0\)、\(V_{\mathrm{soft}}(s_0)\leftrightarrow\log Z_\theta\)：

\[
\Delta_{\mathrm{PCL}}(\tau)=-\log Z_\theta+\sum_t\big(r_t-\log\pi_\theta(s_{t+1}|s_t)\big)
=\log\frac{R(x)\prod_tP_B(s_{t-1}|s_t)}{Z_\theta\prod_tP_{F,\theta}(s_t|s_{t-1})}=-\Delta_{\mathrm{TB}}(\tau),
\]

平方后两个 loss 逐样本相同。DB↔SQL 同法：把 \(Q_\theta(s,s')=\log F_\theta(s\to s')\)、\(V_\theta(s')=\log\sum_{s''}\exp Q_\theta(s',s'')\) 代入 SQL 残差 \(Q_\theta-(r+V_\theta')\) 即得 DB 残差。modified DB 的要点是"处处可终止"的边界条件 \(F(s)P_F(s_f|s)=R(s)\)（guide §3 约定段的 HyperGrid 情形）可反解 \(F(s)=R(s)/P_F(s_f|s)\)：终止概率顶替了流网络，整个算法只剩一个策略网络。

**等价为什么不等于相同的训练动力学**：

1. **采样分布不同**。上表的等价是"逐样本残差相等"或"on-policy 期望梯度相等"，对训练分布保持沉默。RL 实现默认 replay buffer + \(\varepsilon\)-greedy 的 off-policy 采样，GFN 实现默认 on-policy / tempered \(P_F\)；残差相同、样本权重不同，期望梯度与整条优化轨迹就不同。TB↔policy gradient 一条更脆弱：只在 on-policy 期望意义下成立，off-policy 时 TB 回归仍保正确零点，policy gradient 则需重要性修正（与 guide §5.2 对 VI 的辨析同款）。
2. **bootstrap 与否**。SQL/SoftDQN 用冻结 target 网络做半梯度 bootstrap（Tiapkin 的 loss 中是 \(F_{\bar\theta}\)），DB 惯例对残差两侧同时求导；TB/PCL 不 bootstrap、终端信号直达整条轨迹但梯度方差更高。零点集合相同，梯度场不同。
3. **\(P_B\) 是否可学**。等价定理要求 \(P_B\) 固定，因为 \(\log P_B\) 就是 MDP 的奖励；学 \(P_B\) 等于边训练边改奖励（非平稳 MDP），soft RL 的收敛结论不再直接套用。Tiapkin 建议按双人博弈理解，截至核实日仍是开放方向。

## E. 最容易误解的三点

1. **\(\lambda=1\) 是硬约束（Tiapkin Remarks 3–4）**：系数 \(\lambda\) 的 soft RL 等价于系数 1、奖励整体除以 \(\lambda\)。终端项变成 \(\log R^{1/\lambda}\) 尚可读作退火，但中间项变成 \(\tfrac1\lambda\log P_B\)——不再是任何归一化反向策略的对数，终止分布既不是 \(\propto R\) 也不是 \(\propto R^{1/\lambda}\)。想采样 tempered 目标，应只让终端奖励承担温度（等价地把中间奖励预乘 \(\lambda\) 抵消缩放）。Bengio et al. 2021 早期"PPO/SAC 不行"的实验同时用了 \(\lambda\ne1\) 与零中间奖励，两条都违反，故不构成对本等价的反例。
2. **等价有方向**：定理说"GFN 解 = 带 \(\log P_B\) 修正奖励的那个 soft MDP 的最优解"，不是"任意 MaxEnt RL 都在采样 \(R/Z\)"；不修正就得 \(\propto n(x)R(x)\)（C 节）。
3. **"值 = log 流"依赖所选 \(P_B\)**：换 \(P_B\) 就换了奖励，进而换了 \(F\)、\(V_1^\star\) 与最优策略的内部信用分配，但终止分布不变。guide §2.6 的内部流自由度，在 RL 侧的化身是"不同奖励整形共享同一最优边际"。

### 参考（定理编号核实日：2026-08-14）

- Tiapkin, Morozov, Naumov, Vetrov. [Generative Flow Networks as Entropy-Regularized RL](https://proceedings.mlr.press/v238/tiapkin24a.html). AISTATS 2024, PMLR 238:4213–4221（Theorem 1；Proposition 1；Remarks 1–4；§3.3–3.4 的 SoftDQN/dueling/policy-gradient 对应）。
- Deleu, Nouri, Malkin, Precup, Bengio. [Discrete Probabilistic Inference as Control in Multi-path Environments](https://proceedings.mlr.press/v244/deleu24a.html). UAI 2024, PMLR 244:997–1021（Theorem 3.1；Proposition 3.2；Corollary B.1；Proposition 3.3；Propositions B.2–B.3 覆盖 \(\pi\)-SQL 与 FL-DB）。
- 背景：Haarnoja et al. 2017（soft Bellman 方程）；Nachum et al. 2017（PCL）；Malkin et al. 2023（TB 与策略梯度的期望梯度等价）。

