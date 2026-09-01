# 非无环 GFlowNet：两个理论框架的深度对比

> 本文扩展理论指南 §6.4。
> 来源：GFlowNet 调研 2026-08 审查扩充（E11）。核心索引见 [README](README.md)。

---

> 联网核验日期：2026-08-14。两篇论文以官方 proceedings 为准：Brunswic, Li, Xu, Feng, Jui, Ma, [A Theory of Non-acyclic Generative Flow Networks](https://ojs.aaai.org/index.php/AAAI/article/view/28989)（AAAI 2024, 38(10):11124–11131，下称 **B24**；附录级结果引其 [arXiv:2312.15246](https://arxiv.org/abs/2312.15246) 扩展版）；Morozov, Maksimov, Tiapkin, Samsonov, [Revisiting Non-Acyclic GFlowNets in Discrete Environments](https://proceedings.mlr.press/v267/morozov25a.html)（ICML 2025, PMLR 267:44887–44910，下称 **M25**）。
> 定位：guide §6.4 已给五条概述（吸收性、重复访问、expected visit counts、发散风险、FM/DB/TB 困环），本节不重复，而是把两框架逐项对齐、给出定理级细节与一个可手算的小例子。记号沿用 guide §2：\(s_0,s_f,\mathcal X,F(s),F(s\to s'),P_F,P_B,R,Z\)。

## 1. 两框架逐项对比

| 维度 | B24（AAAI 2024） | M25（ICML 2025） |
|---|---|---|
| 状态空间假设 | 一般可测空间（含连续），edgeflow 是 \(S\times S\) 上的**有限非负测度**，边约束用支配测度 \(F\ll\mu\) 表达 | 有限离散图 + Assumption 3.1：\(s_0\) 无入边、\(s_f\) 无出边、每个状态从 \(s_0\) 可达且可达 \(s_f\) |
| 第一性对象 | 流（测度）为原语，\(P_F,P_B\) 经 Radon–Nikodym 导数从流导出 | 策略为原语：先取 \(P_B>0\) 定义轨迹分布 \(P(\tau)=\prod_t P_B(s_t\mid s_{t+1})\)（其 Eq. 7），流是导出量 |
| flow 的定义 | 满足守恒与奖励约束的测度，可与采样实现脱钩：sampler flow \(\bar F\le F\)（arXiv 版附录 Definition 5、Proposition 2），\(\bar F=F\) 才叫 exactly sampled | \(F(s)\triangleq F(s_f)\cdot\mathbb E_{\tau\sim P}[\text{访问 }s\text{ 的次数}]\)（Definition 3.5），构造上就是**期望访问次数**；且与 \((P_B,F(s_f))\) 一一对应（Proposition 3.7） |
| 吸收性条件 | Theorem 2：\(R\neq 0\) 的 R-flow 之采样时间 \(\tau\) a.s. 有限，且 \(s_\tau\sim R/R(S^*)\)（Bengio 采样定理的测度版） | Lemma 3.4：\(P_B>0\)（全边）+ Assumption 3.1 ⇒ 反向链是有限吸收 Markov 链，\(P\) 归一且 \(\mathbb E[n_\tau]<\infty\) |
| 总流与期望长度 | 不等式：\(\mathbb E[n_\tau]\le\frac{1}{F(s_0)}\sum_{s\notin\{s_0,s_f\}}F(s)\)（Theorem 2 的上界，此处按 M25 Eq. 6 的离散转述；arXiv 版附录 Corollary 1 用 sampler flow \(\bar F\) 把它收紧为等式） | 等式：\(\mathbb E[n_\tau]=\frac{1}{F(s_f)}\sum_{s\notin\{s_0,s_f\}}F(s)\)（Proposition 3.12，明确自称是 B24 Theorem 2 的加强） |
| 环的代数结构 | 0-flow（Definition 2：\(R=0\) 的流）推广环；图上所有 0-flow 都是环的非负组合（arXiv 版 Corollary 2/3），R-流集合是仿射空间 \(\mathcal F_R=F^\ast+H_1(G)\)（arXiv 版 Theorem 5） | 不需要独立的 0-flow 概念：沿环加流不产生"新自由度"，只等价于**换一个 \(P_B\)**（Proposition 3.7 的双射） |
| 损失稳定性 | Definition 3（加 0-flow 不减小 loss 即稳定）；Theorem 3：divergence 型 FM/DB/TB **不稳定**；Theorem 4：差值型稳定族；Theorem 1：稳定 loss + 流正则的极小化序列若收敛则趋于**无环** R-flow | Corollary 3.11：**固定 \(P_B\) 时稳定性不起作用**（解唯一、\(\mathbb E[n_\tau]\) 有限）；学 \(P_B\) 时提出 scaling 假说：起作用的是误差按 \(\Delta F\) 还是 \(\Delta\log F\) 尺度计算 |
| 适用训练目标 | 稳定化的 FM/DB（见 §3）；TB 无已知稳定版本（其原文明言 \(F\mapsto F^\otimes\) 非线性导致分析失败） | 固定 \(P_B\)：无环文献中任何 loss 照用；学 \(P_B\)：标准 loss + 状态流正则 \(\lambda F_\theta(s)\)（其 Eq. 12） |
| 与 RL 的联系 | 未涉及 | Theorem 3.13：把 Tiapkin et al.（AISTATS 2024，guide §5.1）的熵正则 RL 等价推广到非无环：\(V^\star=\log F(s)\)、\(Q^\star=\log F(s\to s')\)，证明绕开 DAG 拓扑序归纳，改用 occupancy measure 凸优化 |

一句话总结取向差异：B24 用测度论换最大普适性（连续、带环一并覆盖），流可以"存在但采不出来"；M25 用有限性与正性假设换唯一性与等式刻画，流被定义为采样过程的期望访问次数，**不存在脱离策略的流**。

两框架在有限图上可以互译，词典如下：

- B24 的"给 R-flow 加一个 0-flow"\(\;\Longleftrightarrow\;\) M25 的"换一个 \(P_B\)"。因为 \(F+t\mathbf 1_\gamma\) 仍守恒且正，由 Proposition 3.7 它必是**另一对** \((P_B',Z)\) 的期望访问次数流；仿射空间 \(\mathcal F_R\) 与合法 \(P_B\) 的集合一一对应。
- B24 的"exactly sampled 流"在 M25 假设（有限、连通、全边正）下是**全部**流：\(\bar F<F\) 的僵尸分量只在测度框架或违反 Assumption 3.1 时出现。
- B24 的"寻找无环 R-flow"（Theorem 1 的正则化极限）\(\;\Longleftrightarrow\;\) M25 的"在 \(P_B\) 空间里最小化 \(\sum_s F(s)\)"（其 Eq. 11 约束优化）。

## 2. 关键技术点展开

### 2.1 为什么标准 FM/DB/TB 会"鼓励流困在环中"（B24 Theorem 3）

B24 把三个经典 loss 统一写成广义 divergence \(\operatorname{div}_{g,\nu}(\alpha,\beta)=\int g\!\big(\tfrac{d\alpha}{d\beta}\big)\,d\nu\)：FM 取 \((\alpha,\beta)=(F_{\mathrm{in}},F_{\mathrm{out}})\)、DB 取 \((F_f,F_b)\)、TB 取轨迹流 \((F_f^\otimes,F_b^\otimes)\)，\(g=\log^2\) 恰好还原 guide §3 的三个平方 log-ratio loss。**Theorem 3（Instability of divergence-based losses）**：若 \(\operatorname{div}_f\) 是 proper f-divergence，则 \(\mathcal L_{\mathrm{FM},f},\mathcal L_{\mathrm{DB},f}\) 不稳定（唯一例外是 total variation）；\(\mathcal L_{\mathrm{FM},g,\nu},\mathcal L_{\mathrm{DB},g,\nu},\mathcal L_{\mathrm{TB},g,\nu}\) 均不稳定。

机制是**比值的尺度不变性**：对任意 0-flow \(F_0\)，当 \(t\to\infty\) 时 \(\frac{d(F_1+tF_0)}{d(F_2+tF_0)}\to 1\)。也就是说，在环上同时给"入流"和"出流"堆一份共同质量 \(tF_0\)，任何只看比值（log-ratio）的残差都会被稀释趋零——梯度下降因此获得一个永远下坡的方向：不修复真实失配，而是把流无限堆进环里，\(\mathbb E(\tau)\to\infty\)。证明骨架（B24 附录 B.1，Lemma 4/5 + Lebesgue 收敛定理）：沿 0-子流方向计算 loss 的方向导数并证明其可为负；由 Lemma 1（稳定的充分条件是该方向导数对一切 0-子流非负）即得不稳定。TV 例外恰恰因为 TV 看的是**差值** \(|\alpha-\beta|\)，加共同质量不改变差值；这正是 M25 scaling 假说（\(\Delta F\) 尺度自带稳定性、\(\Delta\log F\) 尺度不稳定）的源头。

on-policy 训练还有第二重恶化：divergence 中的训练测度 \(\nu\)（\(\nu_{\mathrm{state}},\nu_{\mathrm{edge}},\nu_{\mathrm{path}}\)）本身来自当前策略采样。流越困在环里，采到的转移就越集中在环上，环上"比值趋 1"的伪零残差在 loss 中权重越大，而携带真实失配的终止边样本占比趋零——不稳定方向被数据分布自我强化。B24 Figure 1 的网格轨迹对比（标准 FM 满屏打转、稳定 loss 直奔目标）是这一机制的直接可视化。

### 2.2 期望访问次数流的良定义条件（M25）

有环时"轨迹经过 \(s\) 的未归一化概率"不再满足守恒。M25 §3.3 的显式反例（含 2-环 \(b\to c\to b\)）：

```text
s0 → a → b ⇄ c → sf      P_B 使得 c→b 的边访问概率为 0.5
访问概率:   in(b)=1+0.5 ≠ out(b)=1     守恒破坏
期望访问次数: in(b)=1+1 = out(b)=2      守恒恢复（b→c 期望走 2 次）
```

DAG 上两种定义重合（每条边至多走一次），这就是 guide §2.2 的流定义在无环时"看不出差别"的原因。良定义需要三件事：

1. **吸收性**：\(P_B>0\)（全边）+ Assumption 3.1 ⇒ 从 \(s_f\) 出发的反向随机游走是吸收链，每个中间状态的返回概率 \(p_s<1\)（Lemma 3.4 证明核心）；
2. **有限期望访问**：访问次数尾概率几何衰减 \(\Pr[N_s'>k]=p_s^k\)，故 \(\mathbb E[N_s']=\frac{1}{1-p_s}<\infty\)，进而 \(\mathbb E[n_\tau]=\sum_s\mathbb E[N_s]<\infty\)；
3. **正性**：Proposition 3.7 的唯一性证明是一个最大值原理论证，要求 \(F>0\) 于全部边；由此每个满足守恒的正边流恰是唯一一对 \((P_B,F(s_f))\) 的期望访问次数流。

对照 B24：其框架允许 \(\bar F<F\)（例如守恒但采样到达不了、或到达强度低于所载流的环分量），这类"僵尸流"在 M25 的假设下被排除——这是两框架最实质的语义分歧。

另有一个 M25 有限框架**原则上不会遇到**、只属于 B24 测度框架的现象：连续空间中存在**无环的 0-flow**。B24 §3.2 的例子是单位圆上无理角旋转 \(\pi_f(z)=e^{2i\pi\theta}z\)：轨迹永不闭合（无环）却永不终止，构成 0-flow。因此在一般可测空间上，"控制环"必须升级为"控制一切 0-flow"，仅仅杀死字面意义的环不够；这也是 B24 用 0-flow（而非环）作为稳定性定义基元的原因。离散有限 + 全边正时无此问题：图上所有 0-flow 都是环的非负组合（arXiv 版 Corollary 2/3，引 Kalpazidou 2007 的环表示理论）。

### 2.3 小例子：3 节点环上的 flow explosion（本文自拟，构造思路仿 B24 §3.1）

取内部状态 \(\{a,b,c\}\)，边 \(s_0\to a\to b\to c\to s_f\)，再加一条回边 \(c\to a\) 构成 3-环 \(\gamma=(a\to b\to c\to a)\)；唯一终止状态 \(c\)，\(R(c)=1\)。参数化边流（终止边 \(F(c\to s_f)=1\) 由实现强制）：

\[
F(s_0\to a)=f,\quad F(a\to b)=F(b\to c)=f+t,\quad F(c\to a)=t .
\]

守恒在 \(a,b\) 处对任意 \(f,t\ge 0\) 成立；在 \(c\) 处残差为 \(f-1\)。故 R-flow 当且仅当 \(f=1\)，环流 \(t\) 是自由参数——\(t\,\mathbf 1_\gamma\) 正是 B24 意义的 0-flow，加多少都不破坏正确性，但诱导策略 \(P_F(s_f\mid c)=\frac{1}{1+t}\)，期望绕环次数为 \(t\)，\(\mathbb E[n_\tau]=3(1+t)\)（与 M25 Proposition 3.12 验算一致：\(F(a)=F(b)=F(c)=1+t\)，和为 \(3+3t\)，除以 \(F(s_f)=1\)）。

现在看训练不稳定性。设网络暂时学错了投递流，\(f=\tfrac12\) 固定，只对 \(t\) 做梯度下降。\(c\) 处标准 FM 残差

\[
\mathcal L_{\mathrm{FM}}(c)=\Big[\log\frac{f+t}{1+t}\Big]^2:\qquad
t=0:\ 0.480,\quad t=4:\ 0.011,\quad t\to\infty:\ 0 .
\]

loss 沿 \(t\) 单调下降到 0，但失配 \(f\neq 1\) 从未被修复，终止概率每圈衰减、\(\mathbb E[n_\tau]=3(1+t)\to\infty\)：这就是"流困在环中"。对照 B24 Theorem 4 的差值型稳定 loss，同一状态的残差为 \(\log\!\big(1+\varepsilon\,(f+t-(1+t))^2\big)\cdot g=\log\!\big(1+\varepsilon(f-1)^2\big)\cdot g\)，其中 \(f\) 项与 \(t\) 无关、权重 \(g\ge 1\) 随总流上升：沿环堆流不再降 loss，唯一下降方向是把 \(f\) 修回 1。

同一例子的 M25 读法：\(f=1\) 时每个 \(t\) 对应一个合法解，其反向策略 \(P_B(c\mid a)=\frac{t}{1+t}\) 随 \(t\) 变化——"0-flow 自由度"翻译过来就是 \(P_B\) 的自由度。若固定 \(P_B(c\mid a)=\tfrac12\)（即钉死 \(t=1\)），解唯一、\(\mathbb E[n_\tau]=6\)，任何标准 loss 都不会爆炸（Corollary 3.11）；不稳定性只有在放开 \(P_B\)（等价地放开 \(t\)）时才被激活。而最小流解是 \(t=0\)：砍掉回边流量，轨迹退化为最短路 \(s_0\to a\to b\to c\to s_f\)，\(\mathbb E[n_\tau]=3(1+t)\) 在 \(t=0\) 取下确界 3。注意 \(t=0\) 使 \(P_B(c\mid a)=0\)，落在 M25 正性假设的边界上（同 guide §2.5 端点 \(q\in\{0,1\}\) 产生零流边的情形）；最小流线性规划在闭多面体上取到它没有障碍——§4 的最短路/OT 联系在这个五状态图上已经现形。

## 3. 非无环框架下各训练目标的修正形式

- **FM → 稳定 FM**（B24 Example 1，Theorem 4 保证稳定）：\(\mathbb E_\tau\sum_t\log\!\big[1+\varepsilon\,|F_{\mathrm{in}}(s_t)-F_{\mathrm{out}}(s_t)|^\alpha\big]\cdot\big(1+\eta\,(F_{\mathrm{in}}(s_t)+F_{\mathrm{out}}(s_t))\big)^\beta\)。要点：Theorem 4 记号中的外层函数作用于**流差**而非流比，权重函数 \(g\ge1\) 随流量上升惩罚大流（勿与 §2.3 的参数 \(f\) 混淆）。
- **DB → SDB**（B24 提出，M25 Eq. 5 记法）：\(\log\!\big(1+\varepsilon\Delta^2\big)\cdot\big(1+\eta F_\theta(s)\big)\)，其中 \(\Delta=F_\theta(s)P_F(s'\mid s)-F_\theta(s')P_B(s\mid s')\)。
- **DB → DB + 状态流正则**（M25 Eq. 11–12）：保留 \(\Delta\log F\) 尺度的标准 DB（拟合精度更好），加 \(\lambda F_\theta(s)\) 近似求解"守恒约束下最小化 \(\sum_s F(s)\)"。\(\lambda\) 权衡期望长度与终止分布偏差；其附录 B.1 指出 on-policy 下该正则实际最小化 \(\sum_s F(s)^2\)。B24 Theorem 1 是同思想的极限版本：稳定 loss + 沿 0-flow 方向导数为正的正则 \(\alpha_n\to0^+\)，极小化序列（若收敛）趋于无环 R-flow。
- **TB**：无已知稳定修正（B24 原文明言）。两条可用路线：(i) **固定 \(P_B\)** 后 TB/SubTB/DB 等一切标准 loss 直接合法（M25 Corollary 3.11），解唯一且 \(\mathbb E[n_\tau]\) 有限，代价是手选 \(P_B\) 的期望长度可能巨大（M25 在 \(20^4\) hypergrid 上量级不可用）；(ii) 学 \(P_B\) 时用 TB + 流正则，即 guide §5.5 所述 2026 OT 论文神经实验采用的"正则化 TB"。
- **RL 化目标**（M25 Theorem 3.13）：设 \(r(s,s')=\log P_B(s\mid s')\)、\(r(x,s_f)=\log R(x)\)，则 \(\lambda=1\) 的熵正则最优策略即 \(P_F\)，SoftDQN/MaxEnt 类算法可整体迁入非无环环境。

## 4. 为什么非无环是最短路 / OT 结果的必要基石（衔接 guide §5.5）

guide §5.5 的核心恒等式 \(\sum_{e\in E^\circ}F(e)=\mathbb E[|\tau|]\) 正是 M25 Proposition 3.12 的等式（边/状态计数相差一个不影响最优解的加性常数），而它只在**期望访问次数**语义下成立——没有非无环流理论，"最小总流"这个目标函数本身写不出来。必要性有三层：

1. **自由度**：最小流原理有意义，前提是 R-流集合有可优化的方向。B24 arXiv 版 Theorem 5 说 \(\mathcal F_R=F^\ast+H_1(G)\)：优化空间恰是图的环空间。DAG 上非负 0-flow 只有 0；分层 DAG（如单向 hypergrid）中所有到 \(x\) 的路径等长，总流被 reward matching 完全锁死——最小流原理在其上退化为常数。
2. **可达性**：OT 定理要求任意源—终点对 \((u,x)\) 可达，否则耦合集 \(\Gamma(L,R)\) 中的质量无路可走。对称移动（网格 ± 步、群生成元）天然造环，这类环境恰是全对可达的典型来源。
3. **代价结构**：ground cost \(d(u,x)\) 取图最短路，只有在允许双向/回退移动的有环图上才是真正的度量（如 Manhattan 距离、Cayley 距离）；同时"最短路被选中"这一现象的理论种子就是 B24 Theorem 1（正则极限流无环、剪掉一切绕行），经 2026 最短路论文（arXiv 2603.01786）到 OT 等价（guide §7.3 表）一脉贯通。

因此 guide §5.5 结论中"允许有环（non-acyclic 框架）"不是背景板，而是让"最小总流 = 最小期望长度 = 最短路路由 = Kantorovich OT"这条链的第一环成立的先决条件；M25 的 \(\lambda\) 权衡也预演了 guide §5.5 的实验观察——更强流正则缩短路径、却增大终止分布偏差。做 guide §11 练习 9/10（最小流即最短路、从最短路到 OT）之前，建议先在本节 §2.3 的五状态图上手算一遍 \(t\mapsto\mathbb E[n_\tau]\)：最小流解砍环、选最短路的逻辑在最小规模上完全同构。

## 5. 记号与归属声明

- 本节记号完全沿用 guide §2/§3：\(F_{\mathrm{in}},F_{\mathrm{out}}\) 即 guide §3.1 FM 中的父边/子边流和；\(n_\tau\)（M25 记法）为内部状态计数的轨迹长度，B24 的采样时间 \(\tau\) 与之等价（差固定常数）。
- 定理编号对照（正文引用均按此）：B24 主文 Definition 1–3、Lemma 1、Theorem 1（稳定正则化极限）、Theorem 2（吸收 + 采样定理 + 总流上界）、Theorem 3（divergence 型 loss 不稳定）、Theorem 4（稳定族充分条件）；B24 arXiv 扩展版附录 Definition 5 / Proposition 2 / Corollary 1（sampler flow）、Corollary 2/3（0-flow 皆环型）、Theorem 5（仿射结构），AAAI 8 页版不含附录。M25：Assumption 3.1、Lemma 3.4、Definition 3.5、Proposition 3.6–3.10、Corollary 3.11、Proposition 3.12、Theorem 3.13。
- §2.3 的 3 节点例子为本文自拟并逐一验算（守恒、\(\mathbb E[n_\tau]\)、FM 残差数值），构造思路仿 B24 §3.1 的带自由环参数图与其 Figure 2 的 \(A\to B\to C\to A\) 拓扑，数值不出自任何一篇论文。
- M25 对 B24 的两处修正应转述准确：(i) "无环情形一切定义直接搬到有环"并不成立（访问概率流不守恒，见 §2.2）；(ii) 固定 \(P_B\) 时 loss 稳定性与优化结果无关（Corollary 3.11），稳定性只在学 \(P_B\) 时才是真问题。

