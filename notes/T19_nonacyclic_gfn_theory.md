# T19 · 非无环生成流网络理论：环导致流爆炸，稳定损失是解药

> **A Theory of Non-Acyclic Generative Flow Networks**
> 作者：Leo Maxime Brunswic, Yinchuan Li, Yushun Xu, Yijun Feng, Shangling Jui, Lizhuang Ma（华为上海研究中心 / 华为诺亚方舟实验室 / 上海交通大学）· AAAI 2024（vol. 38, pp. 11124–11131）· [arXiv](https://arxiv.org/abs/2312.15246) · 代码：未开源

## 一句话

第一篇把 GFlowNet 从 DAG 解放到允许环的一般可测空间的理论工作：证明 Flow Matching（FM）、Detailed Balance（DB）、Trajectory Balance（TB）这些既有损失在有环环境中会把流推向沿环无限累积（采样时间爆炸），并提出以「0-flow 不变性」替代「尺度不变性」的一族稳定损失来消除这一病理。

## 1. 要解决的问题

GFlowNet 的原始理论（T01/T02）建立在有向无环图（DAG）上：对象由构造性动作序列生成，天然无环。但三类场景打破这一假设：

1. **连续状态空间**：想把 GFlowNet 当作 MCMC 的可学习替代（如 ray tracing 这类工业级 MCMC 应用），状态空间是连续的，「无环」概念本身需要重新定义。
2. **内在对称性**：Rubik's cube 的 Cayley 图这类环境，环是结构的一部分。
3. **不可避免的回退**：对手可以把你打回之前状态的博弈、允许撤销动作的规划。

Lahlou et al.（T12）把理论推广到连续空间但保留了「有限吸收」这一变相无环假设；CFlowNets（N015）做了连续控制的尝试但无理论保障。有环时会发生什么、既有损失为何失效，此前没有系统回答。

**核心病理（第 3.1 节的四状态反例）。** 图 \(s_0 \to A \to B \to C \to s_f\)，其中 \(B \to C\) 之外还有 \(C \to B\) 构成环。参数化边流 \(F_\lambda\)（自由参数 \(\lambda \geq 0\) 为环上的流量），所有满足奖励约束的流对应 \(\lambda \geq 0\)。计算表明 \(\partial_\lambda \mathcal{L}_{FM}(F_\lambda) < 0\)：梯度下降会把 \(\lambda\) 推向 \(+\infty\)，即沿环的边流无限增大。后果是采样 Markov 链在环里打转，期望采样时间 \(\mathbb{E}(\tau) \to +\infty\)。DB、TB 有同样问题。

## 2. 核心方法

**测度论框架（第 3.2 节）。** 相比 Lahlou et al. 用「拓扑空间 + 有限吸收核」，本文用更朴素的有限非负测度语言，且不引入任何无环假设：

- **边流（edgeflow）**：\(S \times S\) 上的有限测度 \(F\)；入流/出流为 \(F_{in}(A) = F(S \to A)\)、\(F_{out}(A) = F(A \to S)\)。
- **流匹配约束**：测度等式 \(\mathbb{1}_{S^\circ} F_{in} = \mathbb{1}_{S^\circ} F_{out}\)（源汇之外入流等于出流）。
- **奖励约束**：\(F(\cdot \to s_f) = R\)，\(R\) 是 \(S\) 上的有限奖励测度。
- **边约束的推广**：DAG 的边集被「支配测度」取代——要求 \(F \ll \mu\)，其中 \(\mu\) 满足 \(\mu(s_f \to S) = \mu(S \to s_0) = 0\)。
- **策略与流的互换**：由 Radon–Nikodym 导数，边流 \(F\) 诱导前向核 \(\pi_f(x, A) = \frac{dF(\cdot \to A)}{dF(\cdot \to S)}\) 与后向核 \(\pi_b\)；反之从 \((\pi_f, F_{out})\) 可重建 \(F\)。

**0-flow：环的正确推广。** 0-flow 定义为对零奖励 \(R = 0\) 同时满足流匹配与奖励约束的流（Definition 2）。图上 0-flow 空间由环生成（每个环 \(\gamma\) 给出指示流 \(\mathbb{1}_\gamma\)）；连续空间上还存在**无环的 0-flow**——单位圆上无理旋转 \(\pi_f(z) = e^{2i\pi\theta} z\)（\(\theta\) 无理）诱导永久漫游但从不闭合的轨迹。这说明连续情形的病理源不只有环，「控制 0-flow」才是正确的问题表述。

**稳定性（Definition 3）。** 损失 \(\mathcal{L}\) 稳定，若对任意 0-flow \(F_0\)：

\[\mathcal{L}(F_1 + F_0, \dots, F_p + F_0) \geq \mathcal{L}(F_1, \dots, F_p).\]

即「往解里灌环流」不能降低损失。Lemma 1 给出充分条件：对所有为各 \(F_i\) 子流的 0-flow \(F_0\)，方向导数 \(\partial_{F_0} \mathcal{L} \geq 0\)。

**损失的散度形式（第 4.1 节）。** 定义 \(\mathrm{div}_{g,\zeta}(\rho, \eta) = \int g\!\left(\frac{d\rho}{d\eta}(x)\right) d\zeta(x)\)，其中 \(\zeta\) 为训练分布。取 \(g = \log^2\) 即恢复 FM/DB/TB：

- \(\mathcal{L}_{FM}(F) = \mathrm{div}_{g,\zeta_{state}}(F_{in}, F_{out})\)
- \(\mathcal{L}_{DB}(F_f, F_b) = \mathrm{div}_{g,\zeta_{edge}}(F_f, F_b)\)
- \(\mathcal{L}_{TB}(F_f, F_b) = \mathrm{div}_{g,\zeta_{path}}(\widetilde F_f, \widetilde F_b)\)（\(\widetilde F\) 为轨迹流）

**稳定损失家族（第 4.2 节）。** 病根是散度型损失的尺度不变性：\(t \mapsto \frac{d(F_1 + tF_0)}{d(F_2 + tF_0)}\) 随 \(t\) 增大趋于 1，加环流反而「看起来更平衡」。解法是把作用在**比值**上的 \(f\) 换成作用在**差值**上，即 0-flow 不变性：

\[\Delta_{f,g,\zeta}(\varphi, \psi) = \int f(\varphi(x) - \psi(x))\, g(\varphi(x), \psi(x))\, d\zeta(x),\]

\(\varphi, \psi\) 是相对背景测度的密度。据此定义 \(\mathcal{L}_{FM,\Delta,f,g,\zeta}\) 与 \(\mathcal{L}_{DB,\Delta,f,g,\zeta}\)。

**Example 1（稳定 FM 损失，图上形式，论文式 (19)）**：

\[\mathcal{L} = \mathbb{E} \sum_{t} \log\left[1 + \beta\, |F_{in}(s_t) - F_{out}(s_t)|^\alpha\right] \cdot \left(1 + \gamma\,(F_{in}(s_t) + F_{out}(s_t))\right)^{-\delta}\]

推荐参数 \((\alpha, \beta, \gamma, \delta) = (2, 1, 0.001, 1)\)。第一因子惩罚流匹配违反（差值型，对 0-flow 不变），第二因子抑制总流量过大。Example 2、3 给出同构的稳定 DB 损失与稳定 CFlowNet 损失。

## 3. 理论结果

- **Theorem 1（稳定化正则）**：设 \(\mathcal{L}_\lambda(F) = \mathcal{L}(F) + \lambda \mathcal{R}(F)\)，\(\mathcal{L}\) 稳定、\(\lambda > 0\)、正则项 \(\mathcal{R}\) 对一切 0-子流满足 \(\partial_{F_0}\mathcal{R} > 0\)。取 \(\lambda_n \to 0^+\)，若 \(\mathcal{L}_{\lambda_n}\)-最小化的 R-edgeflow 序列收敛到一个流，则极限是**无环 R-flow**。含义：总流量正则（如边流矩阵范数）能在极限意义下杀掉环。
- **Theorem 2（非无环采样定理）**：\(R \neq 0\) 时，R-flow 的采样时间 \(\tau\) 几乎必然有限、期望有限，且采样分布 \(s_\tau \sim \frac{1}{R(S)} R\)。这把 Bengio et al. 的 DAG 采样定理推广到含环与连续空间——正确性不需要无环，需要的是流有限。
- **Theorem 3（散度型损失不稳定）**：若 \(\mathrm{div}_f\) 是 proper f-divergence，则 \(\mathcal{L}_{FM,f}\)、\(\mathcal{L}_{DB,f}\) 不稳定，唯一例外是 total variation；\(\mathcal{L}_{FM,g,\zeta}\)、\(\mathcal{L}_{DB,g,\zeta}\)、\(\mathcal{L}_{TB,g,\zeta}\)（含标准 log² 形式）全部不稳定。
- **Theorem 4（稳定性充分条件）**：若 \(f \geq 0\) 且 \(g \geq 1\)；\(f(x) = 0 \Leftrightarrow x = 0\)；\(f, g\) 连续且分段连续可微；\(f\) 在 \(\mathbb{R}_-\) 递减、\(\mathbb{R}_+\) 递增；\(\partial_{(1,1)} g \geq 0\)，则 \(\mathcal{L}_{FM,\Delta,f,g,\zeta}\) 与 \(\mathcal{L}_{DB,\Delta,f,g,\zeta}\) 对 R-edgeflow 稳定。
- **TB 的空白**：论文明确承认无法给出 TB 的稳定条件——轨迹流算子 \(F \mapsto \widetilde F\) 非线性，分析失效；实验观察到 TB 的行为「更微妙」：路径漫游但采样时间不爆炸，有时长训后能追平稳定损失。这一悬置后来由 T36 部分回答。

## 4. 实验与证据

三组实验，均为验证「稳定 vs 不稳定」的机理性小实验：

1. **Hypergrid（2D, W=20）**：转移允许双向移动（人为引入环）。不稳定 FM 损失下边流发散、路径漫游（图 1 右）；稳定损失 \(\mathcal{L}_{FM,\Delta,f,g}\)（\(f(x)=x^2\)）下路径笔直奔向奖励区（图 1 左）。对照还包括 \(f(x)=(1-x)^2\)（χ² 型）与 \(f(x)=|1-x|\)（TV 型）的散度式损失：前者路径长度爆炸，后者(TV)如 Theorem 3 预测处于稳定边界。
2. **Cayley 图（\(S_{20}\)，置换群）**：由一个对换、一个 20-循环及其逆生成；奖励 \(R_1\)（\(c=20, k=1\)）模拟部分排序任务；初始策略固定为均匀分布（从任意排列出发学通用解法）。最优策略期望奖励 20、期望路径长 5（cut-off 80）。不稳定损失下流量不受控增长；稳定损失下流保持有界、采样时间不爆炸，优于 Metropolis–Hastings 基线。
3. **Point-Robot-Sparse（连续控制）**：双目标 (10,10) 与 (0,0)、起点 (5,5)、最长 12 步，动作角度范围从 (0°, 90°) 扩到 (0°, 360°) 以制造环。Stable-CFlowNets（仅替换 N015 CFlowNets 的损失为 Example 3）探索能力不降（5000 条探索中 valid-distinctive 轨迹数持平），平均奖励显著higher于 CFlowNets 与 DDPG/TD3/SAC/PPO，且方差更小。

证据边界：全部是小规模概念验证；没有大规模分子/序列任务；Cayley 图上初始流的训练未成功（论文自陈）；连续设定未测稳定 DB 损失。

## 5. 在 GFlowNet 版图中的位置

- **上游**：T02（GFlowNet Foundations，DAG 理论被本文推广）、T12（Lahlou 连续理论——本文框架与其平行，但去掉了「有限吸收」这一隐性无环假设，且自评「less involved」）、N015（CFlowNets，本文给出其稳定化版本）、T11（Markov chain 视角，处理循环状态空间的概念性前驱）。
- **直接后继 T36（Revisiting Non-Acyclic GFlowNets, ICML 2025）**：Morozov et al. 认为本文的测度论框架对离散有限环境过重，给出更简洁的存在性/唯一性理论，并**实验性重审本文的 loss stability 概念**——发现固定后向策略 + 简单的 state flow 正则化可以让「不稳定」损失照常工作，对本文「必须换损失族」的处方构成部分挑战。
- **T39（Ergodic Generative Flows, ICML 2025）**：Brunswic 本人的续作，用遍历变换与更弱的流匹配条件覆盖环、连续变换与模仿学习，是本文「acyclic 0-flow / 遍历性」讨论的正式展开。
- **下游应用**：O07（minimum-flow 偏向最短路）与 O08（GFlowNet 边流编码 Kantorovich 最优 coupling）都以非无环流的「内部流不唯一 → 需要选择原则」为出发点，本文是这条 GFN–OT 线的第一块理论基石；N028（MCMC 自适应停止）直接在本文框架内训练停止分类器。
- **与散度训练线的交点**：本文把 FM/DB/TB 统一写成广义散度（第 4.1 节），与 T17（f-散度训练）、T49（f-TB）共享「损失 = 散度」的语言，但关心的是另一维度——T17/T49 关心梯度性质与 mode 覆盖，本文关心 0-flow 方向上的单调性。
- **仓库内上下文**：`surveys/GFLOWNET_NONACYCLIC_CN.md` 以本文为起点梳理整条非无环线。

## 6. 局限与批判

1. **稳定性假设非最优**（论文自陈）：Theorem 3/4 的条件是为证明简洁而取的技术性假设，稳定与不稳定之间的精确边界未刻画。
2. **TB 分析残缺**：最常用的 TB 损失既没有不稳定性的一般证明，也没有稳定变体；这在实践上削弱了论文处方的覆盖面（社区主流用 TB）。T36 后来证明了带固定后向策略的 TB 类目标在非无环离散环境的性质，部分填坑。
3. **「稳定性」概念的实用性存疑**：稳定性是关于「损失对 0-flow 扰动不减」的局部性质，不等于训练动力学的收敛保证；T36 的实验表明状态流正则化（\(\lambda \cdot F(s)\) 惩罚项，思想同本文 Theorem 1）在实践中比更换损失族更简单有效——本文最有生命力的部分反而是正则化定理而非稳定损失族。
4. **实验与理论体量不匹配**：理论覆盖任意可测空间，实验只有三个玩具任务；稳定 DB 损失在连续设定未测；Cayley 图上初始流训练失败削弱了「GFlowNet 替代 MCMC」的叙事。
5. **符号与写作**：测度论记号密集且与 Lahlou et al. 不一致（对照表见论文 Table 1），后续引用者（含 T36）普遍抱怨可读性；乱码级排版（AAAI 版公式压缩严重）加剧了这一点。
6. **探索问题回避**：非无环空间的探索遍历性被列为开放问题，论文只用「加常数背景流/背景奖励」的临时手段。

## 7. 对后续研究的启示

1. **「采样正确」与「采样高效」分离**：Theorem 2 表明含环时正确性廉价（任意有限 R-flow 都对），昂贵的是控制 \(\mathbb{E}(\tau)\)。这把非无环 GFlowNet 的研究焦点从平衡条件转向内部流选择——最小流量、最短路径、OT 成本（O07/O08）都是这个问题的具体化。
2. **0-flow 是分析工具**：把环从图论对象升级为流空间中的方向向量，使「损失对环的响应」可以用方向导数讨论；这一手法被 T36 的简化理论继承。
3. **正则化优先于损失重设计**：Theorem 1 的「\(\lambda \to 0^+\) 极限选出无环流」是后续 state flow regularization（T36）与 minimum-flow 准则（O07）的原型。
4. **连续空间的病理更丰富**：acyclic 0-flow（无理旋转）的例子提示连续 GFlowNet 的稳定性研究不能只盯着环；这一警告与 T39 的遍历性框架、扩散采样器的路径测度分析（N002）相通。
5. **GFlowNet 作为 MCMC 替代的议程**：本文明确提出用可训练的有限混合时间采样器替代 MCMC 的无限混合时间，这一叙事贯穿 N028（学习停止时机）、N105（SMC 接口）等后续工作。
