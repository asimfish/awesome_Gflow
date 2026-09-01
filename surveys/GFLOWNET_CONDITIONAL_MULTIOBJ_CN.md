# 条件、多目标与摊销 GFlowNet

> 本文深化理论指南 §6.1–6.2。
> 来源：GFlowNet 调研 2026-08 审查扩充（E17）。核心索引见 [README](README.md)。

---

## 1. 本节定位

guide §6.1 已用一个例子概述条件 GFlowNet：奖励依赖条件 \(y\) 时目标变为 \(P_T(x\mid y)=R(x\mid y)/Z(y)\)，同一网络接收 \(y\) 即可跨相关目标摊销采样与配分函数估计，并给出三条告诫（\(Z(y)\) 跨数量级、外推依赖表示与训练分布、状态流的概率解释不可脱离定义）。本节把这条线系统展开为四块：**条件 GFN 的形式化**、**温度条件与 \(R^\beta\)**、**多目标（标量化 vs 偏序）**、**跨 (reward, 约束) 的摊销视角**，并与 guide §5.5 / OT 分析 §3.2–3.3 的 “amortized transport operator” 呼应。所有代表作 venue 均已联网核实；凡论文明确证明者标 “（已证）”，凡属本调研的推断/拟议方向者标 “（推断/拟议）”。

## 2. 条件 GFlowNet 的形式化

设条件 \(c\in\mathcal C\)（可为温度 \(\beta\)、约束/掩码、偏好向量 \(\omega\)、或一整个目标分布/奖励规格）。给定条件奖励 \(R(x\mid c)\ge 0\)，终止目标与配分函数为

\[
P_T(x\mid c)=\frac{R(x\mid c)}{Z(c)},\qquad Z(c)=\sum_{x}R(x\mid c).
\]

条件 GFlowNet 用**一个共享网络**同时表达随 \(c\) 变化的前向/反向策略、状态（或边）流与对数配分：

\[
P_F(s'\mid s,c),\quad P_B(s\mid s',c),\quad F(s\mid c),\quad \log Z_\theta(c).
\]

局部/全局目标逐条件成立即可，例如条件 Trajectory Balance：对完整轨迹 \(\tau=(s_0\to\cdots\to s_n=x)\)，

\[
Z_\theta(c)\prod_{t=0}^{n-1}P_F(s_{t+1}\mid s_t,c)=R(x\mid c)\prod_{t=0}^{n-1}P_B(s_t\mid s_{t+1},c).
\]

条件 Detailed Balance 则要求逐边成立 \(F(s\mid c)\,P_F(s'\mid s,c)=F(s'\mid c)\,P_B(s\mid s',c)\)，好处是回避了全局标量 \(Z(c)\)（代价见 §6 第 3 点）。

需要区分两类语义不同的条件，二者常被混谈：

- **改变奖励**：\(c\) 只改 \(R(x\mid c)\) 与 \(Z(c)\)（温度 \(\beta\)、偏好 \(\omega\)、目标区域），状态图与可行转移不变；
- **改变可行集**：\(c\) 是约束/掩码，直接改 \(P_F(\cdot\mid s,c)\) 的**支撑**（禁用某些动作），此时不同 \(c\) 下的可达终止集合 \(\mathcal X(c)\) 都可能不同。

两类的正确性含义不同：前者沿用 guide §4.1 的 “全支撑 + 逐条件 balance” 条件；后者还须保证掩码不破坏可达性（否则某些 \(x\) 在 \(c\) 下无合法轨迹，条件 balance 无从谈起）。

条件 GFlowNet 的形式化源头是 [GFlowNet Foundations](https://www.jmlr.org/papers/v24/22-0364.html)（Bengio, Lahlou, Deleu, Hu, Tiwari, E. Bengio；**JMLR 2023**, 24(210):1–55；arXiv 2111.09266）：该文把 “条件 GFlowNet” 列为核心贡献之一，用它把跨多步构造的难解求和**摊销**为一次训练好的前向传播，从而估计配分函数与**自由能**（未归一化边缘概率的对数），并可估计熵、互信息、以及集合/图上超集给定子集的条件与边缘分布（这些估计性质在 balance 达到最优时成立——**已证**；有限训练下的近似误差是另一回事，见 guide §4）。

承接 guide §6.1 的告诫，条件状态流的 “未归一化求和” 解释

\[
F(s\mid c)\approx\sum_{x\succeq s}R(x\mid c)\times(\text{与后向路径分配有关的权重})
\]

只在给定 \(F\) 与 \(P_B\) 的定义下成立，不能脱离定义任意解释（**沿用 guide 定性结论**）。

## 3. 温度条件与 \(R^\beta\) 采样

guide 练习 7 已给出温度的两条事实：乘正常数 \(c\) 不改变目标（\(cR/\sum cR=R/Z\)），但取幂

\[
P_\beta(x)=\frac{R(x)^\beta}{\sum_{x'}R(x')^\beta}
\]

会改变温度、熵与模式质量。**温度条件 GFN** 就是把 \(c=\beta\) 作为条件，令 \(R(x\mid\beta)=R(x)^\beta\)、\(Z(\beta)=\sum_x R(x)^\beta\)，用**一个模型覆盖整族** \(\{P_\beta\}\)，在推断期任选 \(\beta\) 调节探索/利用。

其代表作是 [Learning to Scale Logits for Temperature-Conditional GFlowNets](https://proceedings.mlr.press/v235/kim24s.html)（Kim et al.，**ICML 2024**, PMLR 235:24248–24270；arXiv 2310.02823）。该文的关键观察是：不同 \(\beta\) 会诱导差异极大的 logit 幅度与梯度分布，直接把 \(\beta\) 喂进网络会造成数值/优化困难；其修复（Logit-GFN）是用一个**关于 \(\beta\) 的学习函数直接缩放策略 logit**，作为适配目标温度的归纳偏置，从而稳定训练并改善**离线跨 \(\beta\) 泛化**与**在线模式发现**（实证结论）。这正是下文 “条件 \(Z\) 估计” 难点在温度轴上的具体化。

## 4. 多目标：标量化 vs 偏序

多目标下奖励是向量 \(\mathbf R(x)=(R_1,\dots,R_K)\)，GFlowNet 有两条哲学不同的路线。

### 4.1 标量化 + 偏好/目标条件（把向量压成标量再条件）

- **MOGFN-PC**：[Multi-Objective GFlowNets](https://proceedings.mlr.press/v202/jain23a.html)（Jain, Raparthy, Hernández-García, Rector-Brooks, Y. Bengio, Miret, E. Bengio；**ICML 2023**, PMLR 202:14631–14653；arXiv 2210.12765）把 MOO 化为一族由偏好 \(\omega\)（单纯形上，训练时 \(\omega\sim\mathrm{Dirichlet}\)）定义的标量子问题，用 **reward-conditional GFlowNet** 条件在 \(\omega\) 上，标量化用加权和或加权 Tchebycheff。论文称这是 reward-conditional GFlowNet 的**首个成功实证**，产出覆盖 Pareto 前沿且**同一偏好内也保持多样**的候选（该文另有 MOGFN-AL，把 GFN 嵌入主动学习循环，属采集函数子问题序列，与条件化正交）。
- **Goal-conditioned GFN**：[Goal-conditioned GFlowNets for Controllable Multi-Objective Molecular Design](https://arxiv.org/abs/2306.04620)（Roy, Bacon, Pal, E. Bengio；**ICML 2023 Workshop**（Challenges in Deployable Generative AI）；arXiv 2306.04620——**workshop 论文，非主会**）指出：标量化在**凹 Pareto 前沿**上会把解推向极端点；改为条件在目标空间的**目标区域**上，可让用户更均匀地遍历整条前沿（实证动机，非定理）。

### 4.2 偏序（不标量化，直接用序）

[Order-Preserving GFlowNets](https://arxiv.org/abs/2310.00386)（Yihang Chen, Lukas Mauch；**ICLR 2024**；arXiv 2310.00386, 2023-09）放弃预设标量奖励，转而学习一个**与给定（偏）序一致**的奖励并按其采样：单目标时是全序，MOO 时偏序取 **Pareto 支配**。**已证**：在单目标最大化（全序）中训练过程会逐步**稀疏化**学到的奖励地形，集中到序中更高层的候选，从而 “前期探索、后期利用”；**MOO 的 Pareto 前沿逼近是实证 SOTA，不是该定理的直接推论**（区分证明与实证）。这条线正是 “reward = 标量化 vs 偏序” 对照的另一极：偏序绕开了 “如何选 \(\beta\)/\(\omega\)/标量化函数” 的难题，但把难度转移到 “序信号从何而来、是否可靠” 上。

## 5. 摊销视角：一个模型跨多个 (reward, 约束)

条件 GFN 的价值不止 “可控采样”，更在**摊销**：Foundations 的观点是把逐条件的边缘化/自由能计算摊销成一次前向传播，替代逐条件重跑的 MCMC。把这一视角推到 OT，就是 guide §5.5 与 OT 分析 §3.2–3.3 的主张：

- 当前 “secretly learns an OT plan” 一类结论基本是**每对源/目标边缘 \((L,R)\) 训练一个模型**；
- OT 分析 §3.3 拟议的下一步是训练 \(P_F(a\mid s,L,R,c)\)，让同一模型对不同源分布、目标分布与代价 \(c\) 泛化，把 GFlowNet 从**单次 OT solver** 升级为 **amortized transport operator**（**本调研拟议方向，非既有定理**）；
- OT 分析 §3.2 的 “隐式/组合图上 OT” 进一步说明：当 \(|U||X|\) 巨大到无法构造 cost matrix 时，条件 GFN 只需局部访问 \(P_F(s'\mid s,c),P_B(s\mid s',c)\)，其独特卖点是**隐式图 + 局部合法路径 + 组合泛化**。

路径空间上更自然的一步是**熵正则 OT / Schrödinger bridge**（OT 分析 §3.4）：把条件设为参考动力学 \(P_0\) 加上固定的起终点边缘，求 \(\min_{P(\tau)}\mathbb E_P[c(\tau)]+\varepsilon\,\mathrm{KL}(P(\tau)\Vert P_0(\tau))\)。GFlowNet 天生持有前向/反向轨迹分布（\(P_F\) 为传输策略、\(P_B\) 为时间反演、TB 约束路径概率比），因此这一结合可能比静态 coupling 更自然；但**标准 TB 并不自动等价于 Schrödinger bridge**，需重新推导参考过程、熵项与边缘约束（OT 分析 §3.4 的告诫，非定理）。

一个易被忽略但贯穿所有条件方法的设计量是**训练时的条件分布** \(p(c)\)：模型只在 \(p(c)\) 覆盖到的条件上被优化，落在其外的 \(c\)（如更大的 \(\beta\)、更偏的 \(\omega\)、更紧的约束）纯属外推——这与 guide §6.1 “条件外推仍依赖表示与训练分布” 的告诫直接对应，也是 §6 第 2 点评估要点的来源。

这条摊销线有强劲的神经 OT 竞争者，例如 [Universal Neural Optimal Transport](https://proceedings.mlr.press/v267/geuter25a.html)（Geuter et al.，**ICML 2025**），因此 “条件 GFN 做摊销 OT” 要成立，必须在上述独特维度上胜出，而非重复静态 coupling 的已解问题（判断，非定理）。统一地看：**温度 \(\beta\)、偏好 \(\omega\)、目标区域、乃至 \((L,R,\text{cost})\)，都是 “对目标测度规格的条件” 的不同实例**——这是本节最重要的洞见，也解释了为何这些看似分散的工作可以共用同一套条件流/条件 \(Z\) 机制与同一批技术难点。

## 6. 技术难点

1. **条件表示**：如何编码 \(c\)（连续 \(\beta\)、单纯形 \(\omega\)、离散约束掩码、或整个奖励/目标测度规格）使共享网络在条件间插值而非互相干扰；表示差 \(\Rightarrow\) 跨条件负迁移。Logit-GFN 的 logit 缩放正是 \(\beta\) 轴上的一个表示修复。
2. **跨条件泛化的评估**：必须在**留出条件**（未见 \(\beta\)/\(\omega\)/约束）上测分布保真——训练条件上的拟合 \(\neq\) 泛化；应报**逐条件**指标（per-\(c\) 的 TV、模式覆盖、hypervolume），而非只看聚合均值，与 guide §4.3 的评估边界纪律一致。
3. **条件 \(Z\) 的估计**：\(Z(c)\) 可跨多个数量级（guide §6.1），TB 里原本的标量 \(Z\) 变成需回归的函数 \(\log Z_\theta(c)\)，其误差直接偏置 \(P_T(\cdot\mid c)\)；DB/SubTB 虽不需全局 \(Z\)，仍需逐条件的局部一致性。温度轴上的数值不稳（§3）就是该难点的典型症状。
4. **与 off-policy/replay 的交互**：回放缓冲里的轨迹各自带着采样时的 \(c\)，跨 \(c\) 复用需正确记账（哪条轨迹以哪个 \(c\) 采得、用于哪个 \(c\) 的 loss），否则重蹈 guide §8.3 “off-policy 不做 importance weights 未必无偏” 的覆辙，只是现在错误沿条件轴额外扩散一维。

## 7. 对比表（venue 均已联网核实）

| 条件类型 | 参数化（网络如何条件化） | 评估指标 | 代表作（venue） |
|---|---|---|---|
| 通用条件 \(y\) / 自由能 | 条件流 \(F(s\mid c)\)、\(\log Z_\theta(c)\)；摊销边缘化 | 配分函数/自由能估计误差 | GFlowNet Foundations（JMLR 2023） |
| 温度 \(\beta\) | \(R(x\mid\beta)=R(x)^\beta\)；学习函数缩放策略 logit | 跨 \(\beta\) 离线泛化、在线模式发现 | Logit-GFN（ICML 2024） |
| 偏好 \(\omega\)（标量化） | \(\omega\sim\mathrm{Dirichlet}\) 输入编码；加权和/Tchebycheff | Pareto/hypervolume + 条件内多样性 | MOGFN-PC（ICML 2023） |
| 目标区域（goal） | 目标空间区域嵌入为条件 | Pareto 均匀覆盖（尤其凹前沿） | Goal-conditioned GFN（ICML 2023 **Workshop**） |
| 偏序（Pareto 支配） | 学习序一致奖励，无显式标量 \(R\) | 单目标收敛（已证稀疏化）+ MOO Pareto（实证） | OP-GFN（ICLR 2024） |
| 目标边缘/代价 \((L,R,\text{cost})\) | \(P_F(a\mid s,L,R,c)\) 编码源/目标/代价 | 与精确 OT solver 距离；跨 \((L,R)\) 泛化 | OT 分析 §3.3 **拟议**（对标 Universal Neural OT, ICML 2025） |

## 8. 声明纪律小结

- **已证**：Foundations 的条件 GFN/自由能估计性质（在 balance 最优时）；OP-GFN 单目标全序下的奖励稀疏化。
- **实证（非定理）**：MOGFN-PC 的 Pareto 覆盖 + 条件内多样性、Goal-conditioned 的凹前沿均匀覆盖、Logit-GFN 的跨 \(\beta\) 泛化与 OP-GFN 的 MOO SOTA。
- **本调研推断/拟议**：把温度/偏好/目标/\((L,R,\text{cost})\) 统一为 “目标测度规格的条件”，以及 amortized transport operator 方向（OT 分析 §3.3）——尚无既有定理保证，需另立参考过程/边缘约束/泛化界。
- **venue 提醒**：Goal-conditioned GFN 是 **ICML 2023 Workshop** 论文，不应记为主会；其余为 JMLR / ICML / ICLR 主会或正式 proceedings。

## 9. 与 guide 的衔接锚点

- guide §6.1（条件 GFN 与摊销边缘化）→ 本节 §2、§5：把 \(P_T(x\mid y)=R(x\mid y)/Z(y)\) 补上共享参数化 \(P_F/P_B/F/Z_\theta(c)\)、条件 TB/DB 与 Foundations 的自由能估计出处。
- guide 练习 7（reward scale 与温度）→ 本节 §3：把 \(R^\beta\) 从 “一次性事实” 升级为 “可条件化的整族 \(\{P_\beta\}\)”，并给出 Logit-GFN 的落地与失败模式。
- guide §5.5 / OT 分析 §3.2–3.4 → 本节 §5：把 “单次 OT solver” 推广为 amortized transport operator 与 Schrödinger bridge 式条件路径分布（拟议，非定理）。
- guide §4.1 / §4.3 / §8.3（正确性、评估边界、replay）→ 本节 §6：条件化使这三条纪律各自多出一维 “跨 \(c\)” 的负担。

