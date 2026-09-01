# GFlowNet 的失效边界：什么时候它不work

> 调研日期：2026-09-01 · 方法：web 检索（arXiv / OpenReview / IEEE / 期刊）+ 本仓库 213 篇目录逐条比对 · 作者：awesome_Gflow 维护流水线

## 0. 为什么单独写这一份

本仓库目录有 213 篇论文，但用「失效/崩溃/不如/欠拟合/反例」这类词扫一遍简介，只有 **1 篇**（T07）命中。这不是目录的疏漏——而是整个领域的论文都以「我们提出 X 并改进了 Y」的框架写作，失效证据被埋在实验章节、附录和 limitations 里，从标题和摘要看不出来。

结果是：想知道「我这个任务该不该用 GFlowNet」的人，读完 200 篇也拼不出答案。这份报告把散落各篇的失效证据抽出来按失效机制归类，每条给出**触发条件、定量表现、可行对策**。

判断口径：只收有定量实验或定理支撑的失效证据，不收「作者礼貌性列出的 future work」。

## 1. 失效模式一：奖励峰值过尖 → 性能崩塌（最硬的边界）

**证据**：N081《Benchmarking GFlowNets against MCMC: The Role of Peak Sharpness and Dimensionality in Discrete Sampling》在 8D HyperGrid 上系统扫描奖励峰宽 σ：

| 奖励地形 | GFlowNet 发现模态数 | MCMC（Metropolis-Hastings）发现模态数 | 结论 |
|---|---|---|---|
| 尖峰 σ ≤ 0.5（近 Dirac） | **1-2 个（崩塌）** | 11-16 个 | MCMC 完胜 |
| 交叉点 σ ≈ 1.0 | 相当 | 相当 | 分水岭 |
| 宽峰 σ = 2.0 | GFlowNet-DB 多约 4.75 倍 | — | GFlowNet 完胜 |

**机制**：GFlowNet 依赖连续可微的学习信号。峰值趋于 Dirac 时梯度信号极度稀疏，训练无法把概率质量导向模态；MCMC 的接受-拒绝机制不依赖梯度，因此更鲁棒。

**成本对比同样残酷**：GFlowNet 训练约 1500 秒，MCMC 采样不到 3 秒——即使在 GFlowNet 占优的宽峰情形，4.75 倍的模态发现是用约 500 倍算力换来的。

**目标内部的排序**：该实验里 DB 一致优于 TB 与 FM；**FM 在多数高维实验中不收敛**，作者归因于流一致性 loss 在稀疏离散图上方差过高。这与 T40（Secrets of GFlowNets）给出的「FM 收敛率 $\mathcal{O}(1/\sqrt{T})$ 优于 DB 的 $\mathcal{O}(1/T^{1/3})$」是**表面矛盾**：前者是高维稀疏图上的实测，后者是 tabular 假设下的理论速率。我的判断：这个矛盾本身就是「理论速率不能直接指导高维选型」的证据，选目标要看任务的图结构而非引用理论速率。

**对策**：奖励尖锐（近 Dirac）、维度不高、且有可用的 MCMC 提议分布时，先试 MCMC。要用 GFlowNet 就先做奖励平滑（温度退火、log 变换）把 σ 推到 1.0 以上。

## 2. 失效模式二：轨迹截断 → 流泄漏 → mode collapse

**证据**：N113《Fixing Truncation-Induced Mode Collapse in GFlowNets via Pruning Loss》（IEEE BIBM 2025）把 mode collapse 的根因定位到一个被长期忽略的实现细节：

在巨大状态空间里训练必须对轨迹长度设上限，于是产生**强制终止态**（forced terminals）——它们不是环境定义的自然终止态。强制终止态违反流守恒的边界约束，造成「流泄漏」，把生成偏向最大长度轨迹，进而触发 mode collapse。

**对策与其适用边界**：Pruning Loss 要求强制终止处的 sink flow 与总出流都等于奖励（双约束），理论上恢复截断空间的流守恒且保证梯度不消失。实验对照很干净：

- 稀疏奖励（kinase 蛋白靶点）：大幅超过标准目标。
- 稠密奖励（drug-likeness）：所有方法表现相当。

也就是说**流泄漏只在稀疏奖励下限制性能**。这个对照让论断可检验：如果你的任务是稠密奖励，换 Pruning Loss 不会有收益。

论文的挑战性论断值得记下来：「修正强制终止处的边界约束比改进平衡方程更根本」——如果成立，那么 TB→SubTB→f-TB 这条改进平衡方程的主线在 mode collapse 问题上一直在治标。

## 3. 失效模式三：系统性欠拟合目标分布

**证据**：T07《Towards Understanding and Improving GFlowNet Training》(ICML 2023) 报告的现象——GFlowNet **系统性地以过高概率采样低奖励对象**：

- 采样平均奖励 $\mathbb{E}_{p_\theta}[R(x)]$ 从远低于目标均值处起步，然后极慢地接近、或在数万轮主动训练后**始终达不到**目标均值。
- 机制：随机初始化时 $P_F(s_{t+1}\mid s_t)$ 熵很高，因此 $p_\theta(x)$ 也高熵，训练要做的是「降熵到恰好匹配」而非「升熵」。
- 早期证据：原始 GFlowNet 论文（T01）在小分子任务上报的 $\log p_\theta(x)$ 对 $\log R(x)$ 回归斜率只有 **0.58**（理想是 1.0），这个数字当时没被当成问题。

**对策**：T07 自己给了三个（优先回放高奖励样本 PRT、相对边流参数化 SSR、引导轨迹平衡 GTB）；T25（Pessimistic Backward Policy）从另一个角度切入——在有限轨迹训练下，GFlowNet 对高奖励对象的流估计偏低，用悲观后向策略最大化观测流使其贴近真实奖励。注意 T25 作者明确承认这是**探索-利用的取舍**，在需要强探索的环境可能反而变差。

## 4. 失效模式四：LLM 场景的前缀坍缩与长度偏置

**证据**：T54（RapTB）把 LLM-GFlowNet 的失败模式命名并复现：

- **前缀坍缩**：早期 token 的熵急剧下降，不同终止态共享几乎相同的前缀。
- **长度偏置**：模型系统性偏好过短或过长的序列。

**归因**：(i) 只有终端奖励时中间步的反馈高方差且含义模糊；(ii) 回放偏置——训练被限制在搜索空间的极小一部分，对这个窄子集的反复强化使分布坍缩。

定量表现见目录 T54 与 `insights/trends_neighbors.md` 的对比表（SubTB 在该任务上有效性掉到 0.328，是「换目标可能换来新失效」的实例）。

## 5. 失效模式五：loss 与分布误差脱钩

**证据**：T51（Stable GFlowNets）在可解析设置下构造出反例——**小 TV 误差不蕴含有界训练 loss**。也就是说：

- 学到的分布已经全局准确，训练 loss 仍可能出现剧烈尖峰；
- 反过来，看着 loss 在降也不能推断分布在靠近目标（这是 T32 的主题）。

实测层面，该文在中等规模 HyperGrid 上观察到 Max-to-Rest 比值的尖峰与平滑后训练 loss 的大幅波动同时出现。

**对策**：注入参考流（reference flow）封顶 loss 比率，代价是有量化的保真度损失（其 Theorem 3.10 给出稳定性-保真度折中）；监控上改用该文的概率 TV 证书（训练时监控）与 Monte Carlo TV 估计（数据高效评估）。

## 6. 元失效：常用评测指标本身是错的

这一节可能比前五节更重要——**如果指标错了，前面所有失效都测不出来**。

T32（ICLR 2025 Spotlight，预印本题为《Analyzing GFlowNets: Limitations, Countermeasures, and Assessment》）系统性地否证了几个流行指标：

| 指标 | 为什么错 |
|---|---|
| $\log p_\theta(x)$ 与 $\log R(x)$ 的 Spearman/Pearson 相关 | 当 $\log p_\theta = c\log R$ 时相关恒为 1.0，但只有 $c = 1$ 才匹配目标；对 $p \propto \tilde\pi^\alpha\ (\alpha>1)$ 给满分 |
| 平均未归一化目标 $\mathbb{E}_{x\sim p_\top}[\tilde\pi(x)]$ | 同样对「在模态上堆积过多质量」的错误分布给高分 |
| Shen's accuracy | 对模态上质量过剩的分布给不当高分 |
| 发现的模态数 / top-k 平均奖励 | 完全不反映分布正确性，只反映搜索能力 |

**唯一被验证可用的**：FCS（Flow Consistency in Sub-graphs）——对目标支撑集的随机「切分」求 L1 误差的 Monte Carlo 估计。与（通常不可算的）L1/TV 误差的 Spearman 相关达 **0.99（集合生成）/ 0.90（序列生成）**，而计算量少约 **3 个数量级**，并有 PAC 界。

**用 FCS 复检得到的负面结果**：LED-GFlowNets 与 FL-GFlowNets（终止态无限制变体）**根本学不到目标分布**——它们在「发现高价值状态」上确实显著优于标准 TB-GFlowNet，但采样分布是错的。用模态数或平均奖励评测会把这两个方法判为优胜。

我的判断：这是整份报告里最该立刻行动的一条。任何要声称「分布匹配得更好」的实验，不报 FCS（或可枚举时的精确 TV）就没有说服力；反过来，读别人论文时看到只报模态数与 top-k 奖励，就该假定分布正确性未被验证。

## 7. 表达能力的硬限制

T32 还证明了基于 1-WL 表达力 GNN 的 GFlowNet 存在**无法表示的目标分布**——这不是训练问题，是参数化的天花板。由于图任务里 GIN/GCN/GAT 这类 1-WL 上界的架构是默认选择，这个限制影响面很广。同一工作也证明在合适的状态图上 GFlowNet 可精确表示任意树上分布，并构造出 balance 不可达的失败案例（其 workshop 前身 N006 里的反例在正式版之外仍独有）。

**对策**：图任务上换更高阶 GNN（论文明确把这列为待做方向），或改用不依赖图同构判别的状态编码。

## 8. 决策表：什么时候不要用 GFlowNet

| 你的情形 | 建议 | 依据 |
|---|---|---|
| 奖励近 Dirac / 极尖峰，维度不高 | 用 MCMC，别用 GFlowNet | N081（1-2 模态 vs 11-16 模态） |
| 有力场 / 梯度 / 能量结构可用（分子构象、Boltzmann 采样） | 用 iDEM（N031）、力引导漂移（N107）一类方法 | `trends_neighbors.md` §1 |
| 只要单个最优解，不需要多样候选 | 用 PPO/GRPO 等奖励最大化方法，更省算力 | GFlowNet 的比例采样在此是纯开销 |
| 算力紧张且任务规模小到可枚举 | 直接精确计算或 MCMC | N081 的 500 倍成本差 |
| 稠密奖励 + 只担心 mode collapse | 不必换 Pruning Loss，收益为零 | N113 的稠密/稀疏对照 |
| 需要多样高奖励候选、奖励函数难以精调 | **用 GFlowNet**，这是它的护城河 | N109 肽实验、T54 分子实验 |
| 需要离散组合空间上的 off-policy 训练与 $Z$ 估计 | **用 GFlowNet**，RL 工具箱无现成对应物 | `trends_neighbors.md` §2 |

## 9. 理论空白（本次核实后确认）

上一版报告写「函数逼近下的收敛性未检索到任何结果」，本次找到了**作者自认**的直接证据：T40（Secrets of GFlowNets' Learning Behavior）在 limitations 里明确写道，其分析假设之外，「把收敛保证扩展到更一般的函数逼近设定，特别是具有现实架构的深度神经网络，将弥合理论与实践的鸿沟，这包括分析网络深度、宽度与 GFlowNet 性能之间的相互作用」。

也就是说现有所有收敛速率（FM $\mathcal{O}(1/\sqrt{T})$、DB $\mathcal{O}(1/T^{1/3})$）与样本复杂度 $\mathcal{O}(|S|L\log(|S|/\delta)/\epsilon^2)$ 都依赖 tabular 或可实现性假设。第 1 节里 FM 在高维稀疏图上不收敛的实测，正是这个理论-实践鸿沟的具体表现。

其余待填空白：
- T51 的 TV 证书依赖后向采样，**连续状态空间下如何构造**是开放问题（其 future work）；概率证书因最坏情况依赖参考流而偏保守。
- FCS 目前只验证了集合与序列生成，图与分子任务上的表现未检索到。
- 1-WL 之外的高阶 GNN 参数化对 GFlowNet 表达能力的提升，未检索到定量结果。

## 附：本报告引用的论文

| 编号 | 论文 | 提供的失效证据 |
|---|---|---|
| **N081** | Benchmarking GFlowNets against MCMC | 峰值尖锐度与维度的定量失效边界、500 倍成本差 |
| **N113** | Fixing Truncation-Induced Mode Collapse via Pruning Loss（本次补录） | 截断→流泄漏→mode collapse 的机制与稀疏/稠密对照 |
| **T07** | Towards Understanding and Improving GFlowNet Training | 系统性欠拟合、回归斜率 0.58 |
| **T25** | Pessimistic Backward Policy for GFlowNets | 有限轨迹下高奖励对象的流被低估；探索-利用取舍 |
| **T32** | Analyzing GFlowNets / When Do GFlowNets Learn the Right Distribution | 指标否证、FCS、1-WL GNN 表达能力上限、LED/FL-GFlowNets 学不到目标 |
| **T40** | Secrets of GFlowNets' Learning Behavior | 收敛速率分层；函数逼近保证是自认空白 |
| **T51** | Stable GFlowNets with Probabilistic Guarantees | 小 TV 误差与无界 loss 可共存；稳定性-保真度折中 |
| **T54** | Rooted Absorbed Prefix Trajectory Balance | 前缀坍缩、长度偏置的命名与复现 |
| **N006** | Analyzing GFlowNets（workshop 前身） | balance 不可达的表达能力反例 |
