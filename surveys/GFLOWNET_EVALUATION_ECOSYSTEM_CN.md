# GFlowNet 评估方法论与开源生态

> 本文为方法论专题，配合理论指南 §12 实验部分阅读。
> 来源：GFlowNet 调研 2026-08 审查扩充（E19）。核心索引见 [README](README.md)。

---

> 定位：独立专题，衔接 guide §4.3–§4.5（三种误差层次）、§12（第一个严谨实验配置）与 catalog §7（代码资源），并回应 review/R07 指出的"库论文本体缺失"。
> 核实说明：2026-08-14 逐条核对 arXiv 元数据、PyPI 版本、GitHub 仓库与论文原文；venue 存疑处已显式标注，不照搬二手转述。

## 1. 评估指标体系：每个指标测什么、何时用、怎么被误用

GFlowNet 的评估问题与标准 RL 根本不同：RL 主看回报（return），GFlowNet 必须问"终止对象是否按 \(R/Z\) 采样"。gfnx 的 `metrics/` 模块把这句话工程化——它刻意提供多个指标，并在文档里点明"GFlowNet 评估不同于以原始 return 为唯一分数的 RL"。以下按"精确性 → 覆盖 → 行为"三层组织。

| 指标 | 测什么 | 何时用 | 常见误用 |
|---|---|---|---|
| 对象级 TV \(\tfrac12\sum_x\lvert P_\theta(x)-P^\star(x)\rvert\) | 终止分布与目标 \(R/Z\) 的整体偏差（唯一的"金标准"分布距离） | 终点可枚举的小环境（HyperGrid、小 Ising/DAG） | 用**有限样本经验频率**算 TV 时有正偏置，完美采样器也非零（见 §3.1）；不报采样预算与"完美采样器参照 TV"就无法解读 |
| 对象级 \(L_1\) / JS / KL | 同上的不同散度视角；KL 对零概率敏感 | 需与 TV 交叉验证或强调尾部/模式项时 | 直接对未平滑经验分布算 KL 会因零概率发散；须说明平滑/截断处理 |
| \(\lvert\log Z_\theta-\log Z\rvert\) | 配分函数（全局归一化）估计精度，间接反映整体标定 | \(Z\) 已知（可枚举或合成目标）时 | 大空间 \(Z\) 未知时用重要性/AIS 估计当"真值"，本身有偏；TB 的 \(\log Z_\theta\) 只是一个可学标量，低方差 ≠ 无偏 |
| mode recall / coverage | 对**已知**参照模式集 \(M\) 的发现比例 | 人工构造 \(M\)（如 bit-sequence 基准 \(\lvert M\rvert=60\)，学习器只见奖励不见 \(M\)） | 把"发现某模式一次"当作"按正确概率覆盖"；recall 高不代表模式间比例正确 |
| number of modes | 高奖励且互相"足够不同"的对象个数（阈值 \(R>\rho\) + 多样性截断如 Tanimoto\(<\tau\)） | 真实分子/序列等不可枚举任务的覆盖代理 | 随采样预算单调增长，不固定样本数就不可比；阈值/截断敏感，换参数结论翻转 |
| 多样性（pairwise distance） | 样本间平均距离（分子 Tanimoto、序列 Hamming/编辑距离） | 与 number of modes 配合刻画"多点开花" | 只在 top-\(k\) 上算多样性，测的是高奖励尖峰而非分布保真；高多样性可与错误比例并存 |
| 轨迹长度分布 / 轨迹熵 | 构造过程的行为特征、是否存在长度偏置或前缀塌缩 | 序列/图任务诊断 stop-action 偏置、prefix collapse | 当作正确性指标；它是行为诊断，不识别 \(P_T\) |
| log-prob vs log-reward 相关（Pearson/Spearman） | \(\log\hat P_\theta(x)\) 与 \(\log R(x)\) 的形状/秩一致性 | 不可枚举但可对采样对象做 \(\hat P_\theta\) 蒙特卡洛估计时（如 bit-sequence） | 相关高只说明"排序/形状对"，不保证比例常数（温度、\(Z\)）对；对少数等奖励对象无分辨力 |
| reward 分布（histogram / mean / **max**） | 奖励侧的一阶/尾部行为 | 作为**辅助**信号或与 Anderson–Darling 比较奖励分布（Shen et al. 2023, ICML） | **最危险的误用**：见 §3.2 |

**贯穿性告诫**：guide §4.3、§13 已证——奖励分布相等推不出对象分布相等。只要多个对象同奖励，就可能"奖励直方图与均值完全一致、对象间严重错配"。因此 max/mean reward 只能当作"能否找到高奖励解"的搜索指标，**绝不能**作为分布拟合是否正确的证据。这一点值得反复强调，因为它是文献中最常见的评估错误：一条不断上升的 mean-reward 曲线看起来很像"训练成功"，但它对模式塌缩、对等奖励对象的错误分配、以及温度/尺度偏移全都不敏感。

### 1.1 指标怎么按场景搭配报告

指标不是越多越好，而要按环境的可枚举性组合，且每个非精确指标都要声明其局限：

- **可枚举环境**（HyperGrid、小 Ising、小 DAG）：以对象级 TV 为主指标，辅以 \(\lvert\log Z_\theta-\log Z\rvert\) 做全局标定检查、mode recall 做覆盖检查；reward 均值仅作搜索侧参考。这是唯一能"证明"分布正确的场景，也是校准指标本身（如经验 TV 下界）的地方。
- **proxy 驱动的分子 / 序列**（sEH、AMP、大 QM9）：没有 TV，改报固定采样预算下的 number of modes + pairwise 多样性 + top-\(k\) reward，并**强制**附 proxy 版本与一个独立 oracle 的抽检结果（§3.3）。此时任何指标都是相对比较，不能声称绝对分布正确。
- **连续 diffusion sampler**：报 \(\log Z\) 偏差与 ELBO/EUBO 夹逼、外加目标特定的 mode 指标（见 §2 与 E04）。
- **跨场景铁律**：一份合格报告至少含一个对象级分布指标；不可枚举时须写明代理指标为何、以及它**不是**全局 TV 的无偏替代。

### 1.2 三个最容易被高估的指标（形式化）

把口头指标写成公式，能立刻暴露它们的隐藏参数与失效条件：

- **number of modes**：\(\#\{x\in\mathcal S:\ R(x)>\rho\ \wedge\ \min_{x'\in\mathcal S,\,x'\neq x} d(x,x')>\tau\}\)，其中 \(\mathcal S\) 是采样集合、\(\rho\) 是奖励阈值、\(\tau\) 是多样性截断、\(d\) 是任务距离（分子 Tanimoto、序列 Hamming）。它同时依赖 \((\rho,\tau,\lvert\mathcal S\rvert)\) 三个量，且关于 \(\lvert\mathcal S\rvert\) 单调不减——**不固定这三者就不可比**，跨论文比较尤其容易在此翻车。
- **mode recall / coverage**：\(\lvert\hat{\mathcal M}\cap M\rvert/\lvert M\rvert\)，其中 \(M\) 是**预先构造**的参照模式集（如 bit-sequence 基准 \(\lvert M\rvert=60\)，学习器只见奖励不见 \(M\)）。它只在有 \(M\) 的合成任务里有意义，且只答"找没找到"，不答"比例对不对"。
- **log-prob–log-reward 相关**：在能对采样对象做蒙特卡洛 \(\hat P_\theta(x)\) 估计时，报 \(\operatorname{corr}(\log\hat P_\theta(x),\log R(x))\)。它测的是**形状/秩**一致（mode-seeking 是否对），**不测比例常数**：温度或 \(Z_\theta\) 整体偏了，相关系数仍可接近 1。因此它是有用的可扩展代理，但不能单独当分布正确性证明。

## 2. 标准 benchmark：设定与"是否有精确 ground truth"

| Benchmark | 设定 | 精确 ground truth？ | 典型评估 |
|---|---|---|---|
| HyperGrid（2D/3D/4D，边长可调） | 网格上每格皆可终止，奖励含多个已知峰；torchgfn/gfnx 均内置，规模从 \(8^2{=}64\) 到 \(32^4{\approx}10^6\) | **有**：可枚举全部终点、\(Z\) 精确 | 全局 TV（gfnx 用末 \(2\times10^5\) 终态经验分布，并报完美采样器参照 TV）；FM/DB/TB/SubTB 对比 |
| 分子·QM9（prepend/append 序列型，11 building blocks/5 blocks） | 奖励为在 QM9 数据集上训练的 proxy（HOMO–LUMO gap 预测）| **半可控**：小设定终点可枚举算 TV，但"真值"由 **proxy 定义**（非物理真值，见 §3.3 泄漏） | reward-分布 TV（gfnx 复现 Shen et al. 2023 设定并用其 proxy 权重）|
| 分子·sEH fragment-based（Bengio et al. 2021, [2106.04399](https://arxiv.org/abs/2106.04399)） | 片段拼接生成分子，proxy 预测 sEH 结合；recursion 库的 `seh_frag`/`seh_frag_moo` 标准任务 | **无**：空间巨大、\(Z\) 未知 | number of modes（reward 阈值 + Tanimoto 截断）+ top-\(k\) + 多样性；无 TV |
| permutation / 组合（TSP 类、OT permutation、phylogenetic tree、Bayesian structure/DAG、Ising） | 排列/合并/加边等结构化构造；phylo 固定 \(n{-}1\) 步合并，DAG 边加且保无环，Ising 逐格定自旋 | **部分**：小 \(n\)/小格可枚举或算后验/配分函数（DAG 由 Erdős–Rényi 真图 + linear-Gaussian/BGe 生成）；规模一大即失去精确真值 | 小规模 TV/后验距离；大规模转覆盖代理与能量 |
| diffusion sampler benchmark（[Improved Off-Policy Training of Diffusion Samplers, 2402.05098](https://arxiv.org/abs/2402.05098), NeurIPS 2024） | 连续目标采样统一基准（详见 E04）：高斯混合、funnel、many-well、log-Gaussian Cox 等，把 PIS/DDS 与 TB/SubTB/DB/log-variance 放同一框架公平比 | **合成目标有**（解析 \(Z\) 或高精度 AIS）；**真实目标无**（AIS/重要性估计非无偏） | \(\log Z\) 偏差、目标特定的 mode 指标、ELBO/EUBO 夹逼 |

### 2.1 三档 ground truth 与它们各自的奖励构造

把上表的"精确真值"列展开成一条谱系，有助于判断一个结论能声称到什么程度：

- **精确档**（HyperGrid、小 Ising/小 DAG）：终点可枚举、\(Z\) 可算，甚至可解析给出目标分布。HyperGrid 常用末 \(2\times10^5\) 终态的经验分布算 TV 并附完美采样器参照；小 DAG 的奖励是 linear-Gaussian/BGe 下的对数后验，真图由 Erdős–Rényi 生成，可与真后验比。
- **半可控档**（QM9 序列型）：终点在小设定下可枚举、能算 TV，但被比较的"真值"是 proxy（HOMO–LUMO gap 预测）而非物理量，因此 TV 小只说明"拟合了 proxy 诱导的分布"。bit-sequence 属类似情形：奖励由到最近模式的 Hamming 距离定义 \(R(x)\propto\exp(-\beta\min_{m\in M}d(x,m))\)，\(M\) 已知故 recall 可算。
- **无真值档**（sEH fragment、AMP、大分子/组合）：空间巨大、\(Z\) 未知、奖励是 proxy，只能相对比较；phylogenetic tree 用偏好"更少突变"的 Gibbs 分布、只建模拓扑，也归此档。

选择原则（承 guide §9.2、§12.1）：**第一份严谨实验必须落在精确档**（HyperGrid 首选），只有能精确算 \(P^\star\) 时，才能干净地区分"loss 下降""平均奖励上升"与"对象分布真的正确"三件事。精确档同时是"调试指标本身"的地方——先在 HyperGrid 上确认自己的 TV 实现、采样预算和完美采样器参照都对，再拿同一套代码去跑不可枚举任务。一旦进入无真值档，所有指标都退化为代理，只能支持"方法 A 相对 B 更好"这类相对结论，不能支持"分布已学正确"这类绝对结论。

## 3. 评估陷阱专题

### 3.1 低 loss、高 TV（以及经验 TV 的正下界）
**现象**：训练 loss 已经很低，但终止分布离目标仍很远。**机制**：guide §4.4 的界是"逐轨迹一致小 loss"才有 \(\operatorname{TV}(P_T,\pi)\le 1-\exp(-2c)\)（其中每条轨迹 \(\mathcal L_{\mathrm{TB}}(\tau)\le c^2\)）；而"batch 平均 loss 很低"并不蕴含逐轨迹一致小，少数高残差轨迹足以让终止分布错，却几乎不抬高平均 loss——这也是"偶发 loss spike 与真实高 TV 可以并存"的原因（[Stable GFlowNets, 2605.01729](https://arxiv.org/abs/2605.01729)，2026 预印本）。另有一个纯统计陷阱：用有限样本经验频率估 TV 有正偏置，gfnx 明确指出"即使完美采样器，其经验 TV 也非零"——直觉上，用 \(N\) 个样本估 \(K\) 个终点的分布，经验 TV 的期望大致按 \(\mathcal O(\sqrt{K/N})\) 衰减，故在大 \(K\)、有限 \(N\) 下即使模型完美也会读出可观的 TV。**对策**：不以平均 loss 单独判断收敛，同时看 max loss 与 loss 分位数；报告 TV 时必附**完美采样器在相同样本预算下的参照值**，否则无法区分残差来自模型还是采样噪声。

### 3.2 mode collapse 被平均指标掩盖
**现象**：mean reward 平稳上升、reward histogram 甚至整体 TV 都"好看"，但模型其实漏掉了若干模式。**机制**（guide 练习 8）：给一个目标质量仅 \(\varepsilon\) 的模式，模型分配远小于 \(\varepsilon\) 的概率，则整体 TV 只变化 \(\le\varepsilon\)、mean reward 几乎不动，但该模式相关的 log-ratio 可以任意大——聚合标量对稀有模式天然不敏感。**对策**：报告 per-mode recall 与 number of modes 随训练步数的**曲线**（而非终值单点），并显式关注尾部/稀有模式而非高奖励尖峰；在可枚举环境里可直接检查每个已知峰的采样频率。

### 3.3 proxy reward 泄漏
**现象**：number of modes、top-\(k\) 都很漂亮，换个 oracle 一测就崩。**机制**：分子/肽任务的奖励几乎都来自训练好的 proxy（QM9 的 HOMO–LUMO、sEH proxy、AMP 的 DBAASP proxy），有两类泄漏——(a) **评估-奖励同源**：用训练所用的同一 proxy 去数模式，测的是"proxy 空间的峰"而非真实性质，proxy 的系统误差会被当成真模式；(b) **oracle 复用**：若 proxy 训练集与评估集重叠，高奖励可能来自记忆而非泛化。**对策**：报告 proxy 的来源、版本与权重哈希；用一个**独立**oracle 或留出集复核 top 样本；在结论里区分"高 proxy 奖励"与"真实优良"，不把前者当后者。

### 3.4 种子方差
**现象**：单种子曲线显示方法 A 明显胜过 B，多跑几个种子后差异消失甚至反转。**机制**：GFlowNet 的模式发现对随机性（初始化、探索噪声、replay 顺序）高度敏感，"某个种子恰好发现了某模式"是高方差事件，不代表方法能稳定发现。gfnx 与 torchgfn 的对比实验都以**至少 3 个随机种子**取平均正是为此。**对策**：分布指标建议 \(\ge 5\) 种子、覆盖/搜索指标 \(\ge 3\)，报告均值 ± 标准差或置信区间；覆盖类指标额外给种子间离散度，避免用"最好一次运行"作图。

## 4. 开源库对比（2026-08-14 联网核实）

| 库 | 后端 | 代表环境 | 损失 | 规模 / 性能 | 维护状态 | 论文 / venue |
|---|---|---|---|---|---|---|
| [torchgfn](https://github.com/GFNOrg/torchgfn) | PyTorch | HyperGrid、DiscreteEBM、BitSequence、Box（连续）、BayesianStructure、Ring、DiffusionSampling | TB/DB/SubTB/FM/ZVar(log-var)/RTB | 教学与方法研究基准；作者自评 9 组配置 × 4 环境族（HyperGrid \(64\to10^6\)）横评（截至 2026-02-10）| **活跃**：PyPI v2.4.1（2026-04-05），CI + 静态类型 | [arXiv:2305.14594](https://arxiv.org/abs/2305.14594)（v4, 2026；OpenReview `oPTBlkG8Sd`；NeurIPS 2025 FPI Workshop 版）。**venue 校正**：R07/任务写的"JMLR 2026"经核实无法确认，疑与 *GFlowNet Foundations*（JMLR 24(210), 2023）混淆；本篇入库前应按 arXiv/OpenReview 记载，勿标 JMLR |
| [recursionpharma/gflownet](https://github.com/recursionpharma/gflownet) | PyTorch + torch_geometric | 图/分子：QM9（温度条件、HOMO–LUMO）、`seh_frag`、`seh_frag_moo`（QED/SA/分子量多目标）| TB/SubTB/FM | 真实分子任务、online/offline 混合训练；依赖较重 | **活跃**：MIT，创建于 2022-02，约 292 stars，示例 v0.0.10 | 无独立论文，配套 Bengio et al. 2021（[2106.04399](https://arxiv.org/abs/2106.04399)）；库本身以仓库引用 |
| [alexhernandezgarcia/gflownet](https://github.com/alexhernandezgarcia/gflownet) | PyTorch | 面向科学发现（如 crystal 生成）+ 标准环境 | TB/FM 等 | 复杂科学应用工程化最深 | 活跃（社区/学术维护）| 无独立库论文；见其科学应用论文群 |
| [gfnx](https://github.com/d-tiapkin/gfnx) | **JAX**（JIT） | hypergrid、序列（多编辑机制）、QM9、AMP、phylogenetic tree、Bayesian structure、Ising | DB/TB/SubTB（单文件基线，CleanRL 风格）| 摘要报"最高 55×（CPU 序列生成）/ 80×（GPU 贝叶斯结构学习）"wall-clock 加速；HyperGrid CPU 对 torchgfn ~5× | **早期**：PyPI v0.0.1，文档齐全 | [arXiv:2511.16592](https://arxiv.org/abs/2511.16592)（2025 预印本，Tiapkin et al.）|

读法：**方法研究/教学选 torchgfn**（可枚举环境、损失最全、验证充分）；**真实分子任务选 recursion（或其 SynFlowNet 衍生，如 recursionpharma/synflownet-boltz）**；**大批量性能实验/横评选 gfnx**（JAX 加速，但版本尚早、抽象少）。一个可信的交叉验证信号：torchgfn 与 gfnx 在同一 HyperGrid 上收敛到**相同 TV**、仅速度不同，说明后者的 55–80× 加速不以牺牲采样质量为代价——这正是"先在有真值的环境上对齐指标"（§2）的价值。

生态提醒：四个库的损失/环境命名并不完全一致（例如 backward action 的抽象层级、exit action 处理各异），跨库复现须逐项对齐而非假定同名同义；社区索引 [Awesome-GFlowNets](https://github.com/zdhNarsil/Awesome-GFlowNets) 可用于发现新实现，但**发表状态与 venue 必须回原始来源核验**（本节 torchgfn 的 JMLR 误标即为一例）。

### 4.1 库论文怎么入库引用（回应 R07）

R07 指出 catalog §7 只收代码仓库、未收论文本体。建议按以下形式补全，务必 venue 准确、版本注明：

- **torchgfn**：引 [arXiv:2305.14594](https://arxiv.org/abs/2305.14594)（Viviano, Younis, Choi, Schmidt, Bengio, Lahlou；2023 首发，2026 大幅扩充 v4，OpenReview `oPTBlkG8Sd`，另有 NeurIPS 2025 FPI Workshop 版），代码引 PyPI v2.4.1；**不要**标 JMLR——JMLR 24(210), 2023 是 *GFlowNet Foundations*（Bengio et al.），与本库论文不是同一篇。
- **gfnx**：引 [arXiv:2511.16592](https://arxiv.org/abs/2511.16592)（Tiapkin, Agarkov, Morozov, Maksimov, Tsyganov, Gritsaev, Samsonov；2025 预印本），代码引 GitHub `d-tiapkin/gfnx`（PyPI v0.0.1）。
- **recursion / alexhernandezgarcia**：均无独立库论文，按"仓库 + 对应方法论文"引用（如 recursion 的 sEH 任务引 Bengio et al. 2021, [2106.04399](https://arxiv.org/abs/2106.04399)）。

## 5. 可复现清单（报告实验必须交代）

承 guide §12.4，一份可复现的 GFlowNet 实验报告应显式包含：

1. **超参数**：损失类型与其专有项（SubTB \(\lambda\)、\(P_B\) 固定/学习、replay 配置与优先级）；策略与 \(\log Z\) 的**各自**学习率与梯度尺度；batch size、优化器；温度 \(\beta\)、reward clipping 与下界 \(\epsilon\)——后三者会**改变实际目标分布**，不报即不可复现。
2. **种子**：种子数（\(\ge3\)，分布指标建议 \(\ge5\)）与聚合方式（均值 ± std / CI），并给覆盖类指标的种子间离散度。
3. **评估协议**：主指标及其**采样预算**；经验 TV 必附完美采样器参照值；不可枚举任务须说明代理指标（held-out 可枚举子空间、重要性/双向轨迹诊断）及其**非无偏**性质；number of modes 的奖励阈值与多样性截断；proxy 的版本/权重来源（防 §3.3 泄漏）。
4. **算力**：硬件（CPU/GPU 型号）、wall-clock 与 iterations/sec、总训练步数；扩散采样器额外报**能量求值次数**（该社区的公平性货币）。
5. **环境规范**：状态等价类与对象 canonicalization、action mask、stop-action 约定与长度偏置处理，确保每个终止对象的合法父路径都被表示。
6. **早停与选择**：**不得**仅以 training loss 早停（guide §12.4）；至少同时监控分布或覆盖代理指标，并报告 batch mean loss / max loss / 分位数以捕捉 §3.1 的逐轨迹一致性。

作为对照，一份**不可复现**的报告通常长这样：只给"mean reward 曲线创新高"、单种子、不写温度 \(\beta\) 与 reward clip、number of modes 不注明阈值与采样预算、用训练同源 proxy 兼作评估 oracle、以 training loss 最低点选模型。上面六条正是对这些坏味道的逐一堵漏——它们不增加实验成本，只增加"别人（和三个月后的自己）能不能重跑出同样数字"的概率。

## 6. 核心建议（一句话）

**永远至少配一个对象级分布指标，并把它锚定在可枚举环境上校准**：在能算 \(P^\star\) 的 HyperGrid 上用带完美采样器参照的 TV 站稳，再带着同一套多种子、固定预算的评估协议迁到不可枚举任务，用覆盖代理做相对比较——**任何时候都不要让 max/mean reward 单独为"分布学对了"背书**。这条纪律同时防住了本专题的四个陷阱：低 loss 高 TV、模式塌缩、proxy 泄漏与种子方差。

