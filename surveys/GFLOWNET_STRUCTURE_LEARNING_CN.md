# GFlowNet 用于贝叶斯结构学习与因果发现

> 本文为应用专题，配合资料清单结构学习论文阅读。
> 来源：GFlowNet 调研 2026-08 审查扩充（E13）。核心索引见 [README](README.md)。

---

> 定位：本专题展开 catalog §3.2（A17–A20 附近）的结构学习线，记号沿用 guide §2（\(P_F,P_B,R,Z,P_T\)）。已收录论文只引编号，不复制卡片；新核实论文给出建议编号（A36 起）。所有标题/venue/链接均已于 2026-08-14 联网核实。

## 1. 核心定位：为什么这个问题是 GFlowNet 的"原生"落点

贝叶斯结构学习的目标不是找一个最优图，而是刻画后验 \(P(G\mid\mathcal D)\propto P(\mathcal D\mid G)P(G)\)。它与 GFlowNet 的匹配来自三个结构性事实：

1. **组合空间超指数**：\(d\) 个节点的 DAG 数量随 \(d\) 超指数增长（\(d=20\) 时邻接矩阵空间已是 \(2^{400}\) 量级），穷举与精确归一化都不可行，恰是 guide §1.1 "组合对象 + 未归一化 reward" 的典型情形。
2. **有限数据下本质不可识别**：仅有观测数据时似然只能识别到 Markov 等价类（MEC）；加上观测噪声，与数据相容的高分图构成庞大等价类。单点估计（如 GES、NOTEARS 类连续优化）丢弃了这一不确定性，而下游因果查询——干预效应估计、实验设计/主动学习——需要对结构不确定性积分。
3. **多条构造路径天然存在**：同一个图可按不同边序构造，这正是 guide §2.1 "图不是树"的情形；GFlowNet 的多路径信用分配（T03 及后续）在此直接适用。

训练后 \(P_T(G)=R(G)/Z\)，即一次摊销训练换来任意多的独立后验样本，且 \(Z=\sum_G R(G)\) 就是模型证据的未归一化估计。这与逐查询 MCMC 的成本结构互补（详见 §3.3 与 guide §5.3）。

## 2. 论文主线（按"结构 → 结构+机制 → 含环 → 规模化/等价类"展开）

### 2.1 起点：DAG-GFlowNet（A17，UAI 2022）

A17 把 DAG 生成写成序贯决策：从 \(d\) 节点空图出发，每步加一条边或停止。它有两个后续反复继承的设计：

- **所有状态皆可终止**（每个中间图都是合法 DAG），因此用一个避免显式学 flow 的修正 detailed balance 条件（对任意加边转移 \(G\to G'\)）：

\[
R(G)\,P_\phi(G'\mid G)\,P_\phi(s_f\mid G')
=R(G')\,P_B(G\mid G')\,P_\phi(s_f\mid G),
\]

  其中 \(P_B\) 取父图上的均匀分布固定不学。这是 guide §3.2 DB 在 "全状态终止" 情形下的特化，后被 A18 推广。
- **reward 取闭式边际似然**：\(R(G)=P(\mathcal D\mid G)P(G)\)，线性高斯用 BGe、离散用 BDe 分数；分数按节点模块化分解，加一条边只需增量计算局部项。

评估上它做了两件此后成为标配的事：小图（\(d\le 5\)）与精确后验逐边比对边际概率；真实数据用 Sachs 流式细胞术数据（\(d=11\)、\(n=853\)、共识图 17 条边），并扩展到含干预的 BDe 分数（精确后验 AUROC 0.816，DAG-GFlowNet 0.700，MC\(^3\) 0.665）。

### 2.2 从结构到"结构 + 机制"

- **VBG**（[Bayesian learning of Causal Structure and Mechanisms with GFlowNets and Variational Bayes](https://arxiv.org/abs/2211.02763)，Nishikawa-Toomey, Deleu, Subramanian, Charlin, Bengio，预印本 2022，arXiv 2211.02763；**建议编号 A36**）：保持 A17 的结构 GFlowNet，外层用变分 Bayes 交替更新线性高斯机制参数，思路与 T10 的 "GFN 内嵌 EM 循环" 同族。仍限线性高斯。
- **JSP-GFN**（A18，NeurIPS 2023）：单一 GFlowNet 两阶段采样——先逐边生成 \(G\)，停止后再采参数 \(\theta\)，目标为联合后验 \(P(G,\theta\mid\mathcal D)\propto P(\mathcal D\mid G,\theta)P(\theta\mid G)P(G)\)。关键技术是对形如 \((G,\theta)\leftarrow(G,\cdot)\to(G',\cdot)\to(G',\theta')\) 的长度-3 无向路径写 SubTB 条件（guide §3.5），并证明满足全部该类条件即得 \(P_T(G,\theta)\propto R(G,\theta)\)；连续参数部分依赖 T12 的测度论框架。**联合建模的实质收益**：不再需要闭式边际似然 \(P(\mathcal D\mid G)\)，只需逐点可算的 \(P(\mathcal D\mid G,\theta)\)，因此机制可以是神经网络参数化的非线性模型——这是从"玩具分数"走向真实机制的门槛一步。

### 2.3 走出 DAG：DynGFN 与"环"的领域动机

**DynGFN**（[DynGFN: Towards Bayesian Inference of Gene Regulatory Networks with GFlowNets](https://arxiv.org/abs/2302.04178)，Atanackovic, Tong, Wang, Lee, Bengio, Hartford，NeurIPS 2023，arXiv 2302.04178；**建议编号 A37**）面对基因调控网络（GRN）的两个非标准挑战：调控回路**固有含环**（反馈调节），且测量噪声使等价类极大。解法是利用 RNA velocity 把结构学习改写为动力系统稀疏辨识：状态转为 \(\mathrm dx=f(x)\) 的依赖稀疏模式，环通过时间展开被合法化，故 reward 基于 velocity 拟合而非 DAG 分数。两个值得单独记住的设计：

- **per-node 分解**：\(Q(G\mid\mathcal D)=\prod_i Q(G_i\mid\mathcal D)\)（每节点独立选父集），把搜索空间从 \(2^{d^2}\) 降到 \(d\cdot 2^d\)（\(d=20\)：\(2^{400}\to\approx 2^{24.3}\)），代价是无法表达跨节点耦合的后验相关；
- **HyperNetwork 机制**：\(P(\theta\mid G)\) 由超网络在给定结构下输出参数（h-DynGFN），或线性系统解析求解（ℓ-DynGFN）。

作者明言当前只能处理 5–20 变量的小系统，但这类小规模、强不确定的问题（如细胞周期调控）恰是后验建模价值最高处。

### 2.4 2024–2026：规模化、等价类与优化取向

- **EP-GFlowNets**（[Embarrassingly Parallel GFlowNets](https://proceedings.mlr.press/v235/silva24a.html)，da Silva, Carvalho, Souza, Kaski, Mesquita，ICML 2024；**建议编号 A38**）：面向数据分片/联邦场景，各客户端对局部后验 \(R_n\) 训练局部 GFlowNet，服务器用新提出的 aggregating balance 条件一步聚合出全局乘积后验 \(R\propto R_1\cdots R_N\) 的 sampler；实验含联邦贝叶斯结构学习与并行系统发育（与 A07 呼应）。
- **SB-GFlowNets**（[Streaming Bayes GFlowNets](https://arxiv.org/abs/2411.05899)，da Silva, de Souza, Mesquita，NeurIPS 2024；**建议编号 A39**）：流式到达的数据下增量更新 GFlowNet 后验、免于从头重训；案例为线性偏好学习与系统发育推断（非 DAG 结构学习，但同属"GFN 作贝叶斯后验基础设施"线，结构任务可直接受益）。
- **CPDAG-GFN**（[Learning Equivalence Classes of Bayesian Network Structures with GFlowNet](https://mlanthology.org/tmlr/2025/liu2025tmlr-learning/)，Liu, Zhu, Bilaniuk, E. Bengio，TMLR 2025；**建议编号 A40**）：观测数据本就只能识别到 MEC，索性把状态空间改为 CPDAG、直接学等价类后验，并配稀疏偏好过滤；这消解了"同一 MEC 内成员重复计数"对后验的稀释。
- **GFlowOpt**（[GFlowNet with Gradient-based Optimization for Bayesian Network Structure Learning](https://dl.acm.org/doi/10.1145/3746252.3761333)，CIKM 2025；**建议编号 A41**）：BIC 作 reward，GFlowNet 生成 + 连续代理模型梯度上升 + 爬山精化，目标是**高分结构搜索**而非后验拟合——读它要带着 A19 的同一提醒：优化指标提升 ≠ reward-proportional 分布正确。
- **GFlowCausal**（[GFlowCausal: Generative Flow Networks for Causal Discovery](https://arxiv.org/abs/2210.08185)，预印本 2022，arXiv 2210.08185；**建议编号 A42**）：同为优化取向的早期工作，贡献在用传递闭包的矩阵化维护把每步无环性检查摊销到 \(O(1)\)。
- 综述入口：**GFlowNets for Causal Discovery: an Overview**（[Manta, Hu, Bengio](https://openreview.net/forum?id=atgDufs209)，ICML 2023 SPIGM Workshop，非主会；**建议编号 A43**）系统对比了本线各方法的后验对象（仅结构 / 结构+机制）、训练目标与机制假设，可与 A01 的应用综述互补。

### 2.5 新论文建议收录一览（供主控整合进 catalog §3.2）

| 建议编号 | 论文 | 状态 | 一句话定位 |
|---|---|---|---|
| A36 | VBG（arXiv 2211.02763） | 预印本 2022 | GFN 结构 + 变分 Bayes 机制的交替方案，A17→A18 的中间站 |
| A37 | DynGFN（arXiv 2302.04178） | NeurIPS 2023 | RNA velocity 驱动的含环 GRN 后验，per-node 分解 |
| A38 | EP-GFlowNets | ICML 2024 | aggregating balance 聚合分片局部后验，含联邦结构学习实验 |
| A39 | SB-GFlowNets | NeurIPS 2024 | 流式 Bayes 更新 GFN 后验（案例非 DAG，基础设施相关） |
| A40 | CPDAG-GFN | TMLR 2025 | 直接在 CPDAG/MEC 空间学等价类后验 |
| A41 | GFlowOpt | CIKM 2025 | 优化取向：GFN + 代理梯度 + 爬山搜高分结构 |
| A42 | GFlowCausal（arXiv 2210.08185） | 预印本 2022 | 矩阵化传递闭包的 \(O(1)\) 无环检查，优化取向 |
| A43 | GFlowNets for Causal Discovery: an Overview | ICML 2023 SPIGM Workshop（非主会） | 本线专题综述 |

## 3. 技术要点拆解

### 3.1 状态、动作与无环约束

状态 = 部分构建的图（邻接矩阵），动作 = 加一条有向边（+ 显式 stop）。构造是**单调的**（只加不删），因此 GFN 状态图本身仍是 DAG，即使被采样的对象（DynGFN 的 GRN）含环。无环约束的执行是**硬 mask**：维护可达性/传递闭包，屏蔽一切"若 \(j\) 已可达 \(i\) 则禁止 \(i\to j\)"的动作（A17；GFlowCausal 将其矩阵化到摊销 \(O(1)\)）。对照组是连续松弛路线（NOTEARS 式 \(h(A)=0\) 惩罚、DiBS/VCN 的无环先验项）：那是软约束，A17 实测 DiBS 在 \(d=11\) 时约 1.5% 的样本含环；GFN 按构造保证支撑集落在 DAG 空间内。这是"约束进动作空间 mask"相对"约束进目标函数"的一次干净胜利，与 A06 的有效性 mask 同型。

策略参数化上，本线通行**层级分解**：先以 \(P_\phi(\text{stop}\mid G)\) 决定是否终止，再在合法边对上按 \(P_\phi(G'\mid G,\lnot\text{stop})\propto m_{ij}\exp(u_i^\top v_j)\) 选边（\(m_{ij}\) 即无环 mask，\(u,v\) 为节点嵌入）。骨干网络从 A17 的线性 Transformer 演进到 A18 的图网络 + 自注意力组合，DynGFN 则因 per-node 分解只需轻量 MLP——图规模一旦上去，骨干的置换等变性与嵌入表达力会成为新瓶颈。

### 3.2 reward 设计

- **边际似然型**（A17、CPDAG-GFN）：\(R(G)=P(\mathcal D\mid G)P(G)\)，要求共轭闭式（BGe/BDe），好处是分数模块化、可增量计算，坏处是机制假设强；
- **联合型**（A18）：\(R(G,\theta)=P(\mathcal D\mid G,\theta)P(\theta\mid G)P(G)\)，免边际化、容纳非线性机制，代价是把连续变量带进状态空间（依赖 T12）；
- **动力系统型**（DynGFN）：reward 来自 velocity 重构误差 + 稀疏先验（\(L_0\)），结构分数不再要求无环。

三者共同的实践敏感点：reward 的温度/尺度直接决定后验峰度（guide 练习 7 的问题在结构学习里同样成立——分数差几个 log 单位，后验就近乎确定性）。

### 3.3 与 MCMC-over-DAGs 的对比

| 维度 | 结构 MCMC（MC\(^3\)、order/partition MCMC、Gadget） | GFlowNet 路线 |
|---|---|---|
| 成本位置 | 推断时：每条链 burn-in + 混合 | 训练时：摊销；训练后每样本一条 \(O(\text{边数})\) 轨迹 |
| 多峰行为 | 局部 move（加/删/反转边）跨峰需长混合；order 空间缓解但引入 order 先验偏置 | 从 \(s_0\) 独立重采，天然跨峰；但训练期探索失败会**静默漏峰**（guide §5.3） |
| 渐近保证 | 固定目标下有平稳分布保证 | 神经近似，零 loss 才有正确性（guide §4，训练误差不可忽略） |
| 增量数据 | 需重跑链 | SB-GFlowNets 型流式更新 |
| 分片数据 | 难以直接并行化后验 | EP-GFlowNets 的 aggregating balance |

本节判断：二者不是替代关系——小图上 MCMC（尤其 Gadget 级别的现代实现）仍是校准基准，GFN 的优势区在"同一后验要反复大量采样、数据分片/流式、或需要条件化摊销"的场景。

## 4. 与非无环理论（guide §6.4）的呼应：把"环"分成两层

DynGFN 常被当作"非无环 GFlowNet"的应用证据，这里必须做一个精确区分：

- **对象层的环**：DynGFN 采样的 GRN 含环，但其 GFN 状态空间（单调加边序列）仍无环——环被动力系统的时间展开吸收进 reward，完全不触碰 T19 的理论困难。
- **状态空间层的环**：只有当构造过程本身允许往返（如"加边 + 删边"的编辑式动作空间、可撤销的局部搜索），才真正进入 guide §6.4 的领地——expected visit flow、吸收性条件、flow explosion（T19；离散情形的简化见 §6.4 所引 Morozov et al. 2025）。

这个区分给出一条未被占领的研究缝隙：**编辑式结构后验采样**。当前所有结构 GFN 都是单调构造，意味着 \(P_B\) 的支撑受限、且无法表达"先加错再修正"的轨迹；允许删边的非无环结构 GFN 可以把 MCMC 的局部 move 优势（细粒度修正）嫁接到摊销框架里，其正确性恰需 §6.4 的机器。DynGFN 的价值在于证明了**领域需要环**（对象层），从而为状态空间层的扩展提供了动机而非解法。

## 5. 评估陷阱：点估计指标会系统性奖励后验坍缩

结构学习沿用因果发现的点估计指标（SHD、AUROC），但对**后验**而言它们可能与目标背道而驰。已核实的三组证据：

1. **E-SHD 低可以是坍缩的症状**：Sachs 数据上 BCD-Nets 的 E-SHD 最低（18.14 vs DAG-GFlowNet 22.88），但 A17 补充材料显示其 1000 个样本只含 **2 个唯一 DAG**（各自独占一个 MEC）——后验几乎退化为点质量，E-SHD 反而受益于保守的稀疏预测。若只看 E-SHD 会把最差的后验判成最好。
2. **MEC 间多样 ≠ MEC 内多样**：BGe 分数下真后验对同一 MEC 内所有 DAG 等概率；DiBS 覆盖了多个 MEC 却几乎不采同一 MEC 的不同成员（且混入大量低分图），说明"多样性"必须分层测量：MEC 覆盖数、MEC 内唯一 DAG 数、按分数分层的覆盖率。
3. **对真图的 SHD 在不可识别问题上定义就有偏**：观测数据只能识别到等价类，DynGFN 因此用 **Bayes-SHD**（到可采集合中最近成员的 Hamming 距离）并配 KL\((Q\Vert P^*)\)；其实验里 ℓ-DynBCD 的 Bayes-SHD 很低而 KL 极高——又一次"点指标好、后验错"的实例。

可执行的协议建议（本节判断）：小图（\(d\le 5\)）必须与精确枚举后验比边际相关性与 KL；中等图报 E-SHD/AUROC 的同时必须报唯一 DAG 数、MEC 覆盖与分层多样性；不可识别设定用 Bayes-SHD 类指标替代对单一真图的 SHD；有条件时加下游校准检验——[Benchmarking Bayesian Causal Discovery Methods for Downstream Treatment Effect Estimation](https://arxiv.org/abs/2307.04988)（预印本 2023）把"后验用于 ATE 估计"作为端到端评估，DAG-GFlowNet 在 ATE 分布上 recall 最高，正说明后验多样性会传导为下游收益。

## 6. 开放问题

1. **规模化**：per-node 分解（DynGFN）牺牲跨节点后验相关，全图空间又限于几十节点；层级动作（先骨架后定向）、CPDAG 空间（CPDAG-GFN）与条件摊销（guide §6.1）的组合尚未系统探索。
2. **编辑式/非无环结构采样**：§4 所述缝隙——允许删边的结构 GFN 需要 §6.4 理论落地，尚无工作实现。
3. **评估标准化**：§5 的分层多样性 + 校准协议没有公认基准套件；社区仍普遍只报 E-SHD/AUROC。
4. **主动因果发现闭环**：结构后验的最大下游价值是干预选择（VBG 的动机之一），但"GFN 后验 → 信息增益最大的干预 → 更新后验"的完整闭环（结合 SB-GFlowNets 的流式更新）仍是空白。

