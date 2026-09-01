# N099 · torchgfn：模块化 PyTorch GFlowNet 库

> **torchgfn: A PyTorch GFlowNet library**
> 作者：Joseph D. Viviano*, Omar G. Younis*, Sanghyeok Choi*, Victor Schmidt, Yoshua Bengio, Salem Lahlou（*共同一作，署名顺序按"距蒙特利尔圣若瑟圣堂的地理距离"排列）· 预印本（JMLR 格式）· [arXiv](https://arxiv.org/abs/2305.14594) · [代码](https://github.com/GFNOrg/torchgfn)

## 一句话

torchgfn 把 GFlowNet 训练拆成环境（Env）、状态/动作（States/Actions）、估计器（Estimator）、采样器（Sampler）、损失（GFlowNet 类）五层可互换组件，是当前 GFlowNet 社区事实上的通用实验协议层——用它换一个训练目标只需换一个类名。

## 问题与动机

GFlowNet 研究扩张后，开源实现碎片化：多数代码是单项目定制（bespoke），已有的库各有强绑定的领域偏向——

- `recursionpharma/gflownet`：专攻分子/图生成，核心数据结构绑死 torch_geometric 与 networkx；
- `alexhernandezgarcia/gflownet`：面向复杂科学应用（如 Crystal-GFN 晶体生成）的完整实验框架，环境是有状态的（stateful）；
- `gfnx`（Tiapkin et al., 2025）：JAX 实现，JIT 编译采样循环换取速度，但只支持离散环境、抽象层少。

结果是：想在标准基准上测试一个新损失函数或新采样策略的研究者，要么重写整个训练循环，要么被某个领域库的假设锁死。torchgfn 的目标是提供领域无关（domain-agnostic）、以可扩展性为第一优先级的通用工具箱，把"验证一个 GFlowNet 方法学想法"的成本降到最低。

## 方法核心

### 2.1 五层解耦架构（论文 Figure 1 / 附录 Figure 2）

pointed DAG 上的 MDP 由以下组件交互实现，每层可独立替换：

| 组件 | 职责 | 关键设计 |
|---|---|---|
| `Env` | 无状态（stateless）step 函数：给定 $(s, a)$ 产生 $s'$ | 无状态设计允许多节点广播环境查询，解耦"策略采样快、环境查询慢"的瓶颈（附录 B.1） |
| `States` | 状态批次 + 元数据（如动作掩码 forward_masks/backward_masks） | 数据可以是任意类型（tensor / torch_geometric Data / numpy），异构数据用 tensordict 统一批处理 |
| `Actions` | 构图动作；离散环境中是 $0$ 到 $n_{actions}-1$ 的整数，退出动作占索引 $n_{actions}-1$ | 简单离散环境自动子类化 |
| `Containers` | Transitions / Trajectories / StatesContainer 三种，包装采样产物供损失消费 | 支持 ReplayBuffer 做离策略经验回放 |
| `Estimator` + `Sampler` + `GFlowNet` | Estimator 是 nn.Module 包装器（输出动作分布参数）；Sampler 定义采样逻辑；GFlowNet 类封装损失 | Sampler 与 Estimator 通过 PolicyMixin 接口解耦，循环策略（RNN/Transformer 带 carry）与无状态策略共用同一 rollout API |

### 2.2 损失即对象

库的核心架构决策：训练目标本身是可互换的 `GFlowNet` 对象。已实现的损失（附录 B.7）：

- flow matching loss（Bengio et al., 2021）
- detailed balance loss（Bengio et al., 2023）及其 modified 变体（Deleu et al., 2022，即 DAG-GFlowNet 用的版本）
- trajectory balance loss（Malkin et al., 2022）
- sub-trajectory balance loss（Madan et al., 2023）
- log partition variance loss（Zhang et al., 2023）

换损失 = 换类。论文正文给出的对照示例：TB 训练需要 `TBGFlowNet(pf=..., pb=...)`；改成 SubTB 只需换成 `SubTBGFlowNet(pf=..., pb=..., logF=ScalarEstimator(...))`，其余采样与优化循环一行不动。这是"实验协议"用法的核心：控制变量比较不同目标时，环境、网络、采样全部保持恒定。

### 2.3 Gym 环境集

内置环境覆盖六类 GFlowNet 使用场景（正文 §1）：① 所有状态皆终止态的离散环境（HyperGrid）；② 定长轨迹、部分状态终止（DiscreteEBM）；③ 自回归序列（BitSequence）；④ 状态依赖动作空间的连续环境（Box，前向策略支撑在四分之一圆盘/弧上，用 Beta 混合分布）；⑤ 图的离散采样（BayesianStructure 即 DAG-GFlowNet 复现、Ring）；⑥ 连续分布的扩散采样器（DiffusionSampling，Sendera et al., 2024）。

### 2.4 v2 新特性（附录 D）

图生成一等公民支持（torch_geometric 集成进 GraphStates）、条件 GFlowNet（conditioning tensors）、自定义采样器（如 LocalSearchSampler，Kim et al., 2024 的 Local Search GFlowNets）、循环策略（PolicyMixin 统一 rollout API）、扩散采样。

## 理论结果

无。这是系统论文，贡献是软件架构而非定理。

## 实验与证据

### 4.1 性能基准（附录 E.3，2026-02-10 快照）

九个场景配置、四个环境族（HyperGrid 三档、Ising 6×6/10×10、Box2D 两种、BitSequence 两档），全部用 TB 损失，参数量尽量对齐（标准差 ±10.2%），50 次预热 + 100 次计时迭代，3 个种子。HyperGrid batch size 128 的每迭代时间（正文 Table 2）：

| 规模 | gflownet | torchgfn | gfnx |
|---|---|---|---|
| Small ($8^2=64$ 状态) | 141.4 ms | 30.8 ms | 2.5 ms |
| Medium ($16^4=65{,}536$) | 223.0 ms | 77.8 ms | 5.9 ms |
| Large ($32^4=1{,}048{,}576$) | 364.7 ms | 113.3 ms | 10.8 ms |

总体结论（Figure 3）：torchgfn 比 gflownet 快约 2–10×，比 gfnx 慢约 10–20×，居中。gflownet 的设计阻止批维度向量化，Ising 10×10 下每迭代 15.8 秒（torchgfn 208 ms，gfnx 9.8 ms）；gfnx 靠 JIT 编译取胜但内存占用高（HyperGrid small 182 MB vs torchgfn 24 MB）。基准脚本随库分发（`benchmark/` 目录），可随时用最新 commit 重跑。

### 4.2 复现性作为证据

tutorials/examples 含已发表结果的复现，且全部纳入 CI 测试：Local Search GFlowNets（Kim et al., 2024）、DAG-GFlowNet（Deleu et al., 2022）、continuous GFlowNets（Lahlou et al., 2023）、TB（Malkin et al., 2022）、robust scheduling（Zhang et al., 2023）。

### 4.3 GitHub 仓库现状（2026-09-01 核实）

- 最新 release：**v2.4.1**（2026-04-05），PyPI 同步发布；v2.4.0（2026-03-20）为大版本，含 Relative Trajectory Balance 损失、条件 GFlowNet 重构、torch.compile 工具与基准套件。
- 发布节奏：2025-07 至 2026-04 共 10 个 release（2.0.1→2.4.1），持续活跃。
- 315 stars；核心维护者 josephdviviano、younik、hyeok9855、saleml，另有社区贡献者（replay buffer 累积、mRNA 环境等）。
- 依赖：python≥3.10，torch≥2.6.0，torch_geometric≥2.6.1，tensordict≥0.6.1。

## 与谁对话

- 作者阵容即血统：Lahlou（continuous GFlowNet 理论一作）、Bengio（GFlowNet 创始人）、Viviano/Younis/Choi 是 Mila 系维护主力。库是 GFNOrg 组织的旗舰项目。
- 它扮演的角色类似 stable-baselines3 之于深度 RL：不是提出新算法，而是把 2021-2024 的方法结晶成受测试保护的标准组件，让"新损失 vs TB/DB/SubTB"的对照实验有公认协议。
- 与本仓库其他条目的连接：N061（Deleu 博士论文）的 DAG-GFlowNet 在库里有官方复现（`train_bayesian_structure.py`）；扩散采样环境直接对接 Sendera et al. 2024（改进离策略扩散采样器）一线工作；RecurrentDiscretePolicyEstimator 与 PolicyMixin 的 carry 管理为 N082 这类非 Markov 策略（历史依赖的循环策略）提供了工程落点。
- 竞争格局：JAX 阵营的 gfnx 主打速度（10-20× 快），torchgfn 主打可扩展性与生态。论文明确承认不指望完全追平 JAX，路线图（v3）是改造 States/Actions 结构以兼容 torch.compile。

## 局限与批判

- 性能天花板：States/Actions 的动态结构与 torch.compile 不兼容（论文自述），JIT 差距短期无法弥合；对大规模训练（如语言模型 posterior 采样）不是首选。
- 环境定义仍然重：用户需要正确实现 States 子类、掩码更新、Preprocessor，论文也把"简化环境定义"列为未来工作第一条。掩码实现错误是静默失败的重灾区——论文承认无掩码时无效轨迹会多到损失无法在合理墙钟时间内收敛，但库无法替用户验证掩码语义正确性。
- 基准的公平性有限：参数量对齐容差 ±10.2%，且 gfnx 的 JIT step 无法分解计时，逐相位比较只覆盖两个库。
- 论文版本与库现实脱节快：arXiv 版描述 v2.3.0 依赖图，而库已到 v2.4.1（RTB 损失、条件重构未入论文）；引用它作为"库的定义"时需注明版本。
- 缺 RL 基线：与 MaxEnt RL / SAC 系方法的对照（GFlowNet 文献常见争论点）要靠外部实现，论文将其列为未来工作第 5 条。

## 对后续研究的启示

- 方法学论文的实验部分应默认用 torchgfn 落地：新损失写成 `GFlowNet` 子类、新采样策略写成 `Sampler` 子类即可获得全套环境与 CI 保护的对照组，审稿人可复跑。
- `benchmark/` 目录的"可再计算基准"值得推广为社区规范——性能声明随 commit 更新而非定格在论文表格里。
- PolicyMixin 的 carry 抽象说明库已为非 Markov 策略（历史依赖采样器、Transformer 策略）预留接口；在 torchgfn 上实现 N082 的 path-dependent 采样器是检验该抽象是否够用的自然试金石。
- 无状态 Env + 广播查询的设计为昂贵 reward（量化化学、湿实验代理）的分布式 GFlowNet 铺路，这一层还没有论文系统利用。
- gfnx 与 torchgfn 的分工（快 vs 可扩展）意味着方法探索在 torchgfn、大规模扫参在 gfnx 的双库工作流会成为常态；两库间的环境/损失语义对齐是潜在的坑。
