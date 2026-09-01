# GFlowNet 学习路线升级包

> 本文为理论指南 §10–12（六周计划/练习/实验配置）的修订升级，落实审查报告 R08。
> 来源：GFlowNet 调研 2026-08 审查扩充（E05）。核心索引见 [README](README.md)。

---

> 依据：`review/R08_learning_path.md`（MAJOR 5 条、MINOR 14 条）逐条落实。
> 升级对象：`docs/GFLOWNET_THEORY_GUIDE_CN.md` §10/§11/§12 与 `docs/GFLOWNET_RESEARCH_RESOURCE_CATALOG_CN.md` §0.2/§6/§7/§9。
> 外部事实核验日期：2026-08-14。torchgfn 默认超参数与文件路径均按 GitHub `master`（对应 v2.4.1，2026-04-05 发布）核对；参考结果量级来自本机实测（Apple M2 Max，CPU，单种子）与 TB 论文附录 B.1。
> 使用方式：每一节开头标注落地位置（"替换 guide §X" / "追加到 catalog §Y"）；未标注的段落随所属节一起落地。

---

## 1. 修订版六周学习路线【替换 guide §10 全节】

每周按 6–10 小时设计；若数学基础较强，可以压缩到三周。

> **节奏声明（与 catalog §0.2 互引）**：本计划假设在读兼职投入（每周 6–10 小时）。catalog §0.2 的"两周建立理论骨架"假设全职冲刺（每天 6–8 小时），两者覆盖同一批 P0 材料，选一个执行即可，不要混用两份进度表。
>
> **通过标准双栏制**：每周通过标准分为"口头解释"（能不看资料说明白）与"数值对照"（有一个可检查的具体数字或输出）两栏；只过第一栏不算通过。

### 第 0 周：补齐先修概念

每个主题直接对应 catalog §6.3 先修材料表的一行，缺哪补哪：

| 主题 | catalog §6.3 对应行 | 最低要求 |
|---|---|---|
| 未归一化密度、配分函数、能量 \(E(x)=-\log R(x)\) | 概率与信息论（Deep Learning Book, Part I） | 能对一个 4 状态例子手算 \(Z\) 与 \(P^\star\) |
| DAG、拓扑序、Markov policy | 概率图模型（Stanford CS228） | 能给一个 6 节点 DAG 写出全部拓扑序 |
| KL、TV、JS 和 importance sampling | 概率与信息论 + 深度生成模型（CS236） | 能对两个 3 点分布手算 TV 与 KL |
| TD learning、自举和 credit assignment | 强化学习（David Silver RL Course） | 能写出 TD(0) 更新式并解释自举 |
| MCMC 的 detailed balance 与 mixing | 概率图模型（CS228 采样部分） | 能验证一个 2 状态链满足 detailed balance |
| Kantorovich OT 的 coupling 线性规划 | 最优传输（Computational OT）+ OT 实践（POT） | 见本周数值锚点 |
| 最大熵 RL 与普通 RL 的目标差别 | 强化学习行的补充项"另补 maximum-entropy RL" | **只需概念级理解**（目标里多一项策略熵）；严格版本第 4 周读 T14 时补 |

通过标准：

- 口头解释：能说明"最大熵 RL 在目标里加了什么、为什么普通 RL 会塌缩到单一最优解"。
- 数值对照：手写一个 \(2\times2\) 离散 OT LP（给定成本矩阵与两组边缘），解出最优 coupling 和最优值，并用 POT 的 `ot.emd` 验证同一答案。

### 第 1 周：流守恒和终止分布

阅读：

- Notion GFlowNet Tutorial；
- Emmanuel Bengio 博客；
- 原始 NeurIPS 2021 论文第 1–3 节；
- Foundations 第 2 节。

手推（与练习互链：本周第 3 项 ≈ 练习 1 的核心）：

1. 在一个 diamond DAG 上列出所有轨迹。
2. 从轨迹流计算状态流、边流、\(P_F,P_B\)。
3. 证明守恒与终止奖励推出 \(P_T=R/Z\)。
4. 改变内部路径分流比例，验证 \(P_T\) 不变。

通过标准：

- 口头解释：不看资料写出核心定理，并解释为什么图是 DAG 而不是神经网络结构。
- 数值对照：diamond DAG 上给出一张完整数值表（每条边的流、每个终点的 \(P_T\)），改分流比例后 \(P_T\) 各项数值逐位不变。

### 第 2 周：DB、TB 和 SubTB

阅读：

- Foundations 第 2.4–3.3 节；
- TB 论文；
- SubTB 论文。

手推（约 1.5 小时；第 1 项 = 练习 3）：

1. DB 沿路径 telescoping 得到 TB。
2. 对一条长度 3 的路径写出全部 6 个非空 SubTB 区间。
3. 说明 \(\lambda\) 改变的是优化信用尺度，而不是精确终点目标。

实验（约 2–3 小时；**运行并修改现成脚本，不再从零搭建**）：

- 安装 `torchgfn`（v2.4.1；用 Python 3.10–3.13，见 §12.0 注意事项），运行 [`tutorials/examples/train_hypergrid.py`](https://github.com/GFNOrg/torchgfn/blob/master/tutorials/examples/train_hypergrid.py)。该脚本已内置 FM/TB/DB/SubTB/ZVar/ModifiedDB 六种 loss、replay buffer、精确 \(P_T\) 枚举（`get_exact_P_T`）与真实/学习分布对比图（`--plot`），正好覆盖本周实验需求：

```bash
python tutorials/examples/train_hypergrid.py \
  --ndim 2 --height 8 --n_trajectories 16000 \
  --loss TB --validate_environment --no_cuda
# 再把 --loss 换成 FM / DB / SubTB 各跑一遍
```

- 修改任务（三选二即可）：
  1. 记录四种 loss 在相同轨迹预算下的 `l1_dist` 演化，画 \(L_1\)、平均奖励、loss 三条曲线（脚本已算好 `l1_dist`，补 10–20 行日志即可；注意 master 版 `--plot` 的第三幅 L1 子图因列表未填充而空白，数值以 `--validate_environment` 的输出为准）；
  2. 把 `ReplayBuffer` 换成同库的 `NormBasedDiversePrioritizedReplayBuffer`（一行改动），对比 L1 演化；
  3. 加 `--uniform_pb` 与不加各跑一遍，为第 3 周的 \(P_B\) 对比预热。

通过标准：

- 口头解释：能从现象上举例说明低 loss、较高平均奖励、低 TV 为什么不是同一件事。
- 数值对照（锚点见 §12.0.2，基于 `train_hypergrid_simple.py` 全默认实测；完整版脚本默认开 replay，动态相近但非逐点可比）：\(2\)D \(8\times8\)、\(R_0=0.1\)、TB：约 5 千条轨迹内 `l1_dist`（\(=\sum_x|\hat P(x)-P^\star(x)|=2\,\mathrm{TV}\)）应降到 \(5\times10^{-2}\) 以下，且 4 个 mode 状态全部被访问过。若 16k 条轨迹后 L1 仍在 \(10^{-1}\) 量级，通常是 logZ 学习率或 batch 设置出了问题，而不是"还没训练够"。

### 第 3 周：训练行为与内部流

精读 Shen et al. 2023（约 4 小时）：

- 第 3 节：低奖励过采样与评估；
- 第 4 节：flow distribution 和 generalization；
- 第 5–6 节：共享子结构与 GTB；
- 附录中的简化假设。

实验分两档：

- **核心（必做，约 2–3 小时，全部复用第 2 周脚本旗标）**：
  - reward-prioritized replay：沿用第 2 周修改任务 2 的 prioritized buffer 对照；
  - 固定 uniform \(P_B\)（`--uniform_pb`）与学习 \(P_B\) 对比；
  - 记录终点 \(L_1\)、轨迹长度、mode 覆盖。
- **可选/延伸（不计入本周预算，适合想深入 §4.3 评估论证的读者）**：
  - 自写一个"相同奖励、不同对象"的终点环境（可直接用练习 5 的四对象构造），观察 reward histogram 指标失灵；
  - 记录中间状态流与轨迹熵。

通过标准：

- 口头解释：能解释"\(P_B\) 不影响精确目标分布"与"\(P_B\) 会强烈影响训练"为何不矛盾。
- 数值对照：给出 uniform \(P_B\) vs 学习 \(P_B\) 两条 \(L_1\) 曲线并指出差异方向；与 TB 论文 Fig. 2 的观察（\(64\times64\) 网格上学习 \(P_B\) 明显更快收敛）一致或能解释为何不一致。

### 第 4 周：VI、RL 与 MCMC

阅读：

- GFlowNets and Variational Inference；
- A Variational Perspective；
- Entropy-Regularized RL；
- Discrete Probabilistic Inference as Control。

手推（第 4 项 = 练习 2 的展开）：

1. 写出前向轨迹分布 \(Q(\tau)\)。
2. 写出反向目标轨迹分布 \(P(\tau)\propto R(x)P_B(\tau|x)\)。
3. 比较 TB log-ratio 与 \(\log Q-\log P\)。
4. 在树和多路径 DAG 上比较朴素 MaxEnt RL 终点分布。

通过标准：

- 口头解释：每次说"等价"时，都能补出是目标值、零点、期望梯度，还是某种奖励校正后的算法等价。
- 数值对照：在练习 2 的"1 条 vs 10 条等长路径"构造上算出三个数：均匀轨迹采样 \(P(x_2)=10/11\)、reward matching \(P(x_2)=1/2\)、以及你的 DAG 上均匀前向策略的具体值。

### 第 5 周：连续与非无环 GFlowNet

阅读：

- Continuous GFlowNet；
- AAAI 2024 non-acyclic theory；
- ICML 2025 revisiting non-acyclic。

重点问题：

- DAG 证明哪里使用了"每条轨迹有限且不重复状态"？
- 有环时为何 flow 变为 expected visit count？
- 什么条件保证最终吸收？
- 为什么无效环会增加总流/推断成本？

通过标准：

- 口头解释：能说明把一个可逆编辑环境直接套用 DAG TB 会有什么理论缺口。
- 数值对照：构造一个含单个环的 3 状态图，令策略以概率 \(p\) 留在环内，手算 expected visit count 随 \(p\) 的表达式（\(\propto 1/(1-p)\)），并给出 \(p=0.5,0.9,0.99\) 三个数值，说明 \(p\to1\) 时流爆炸。

### 第 6 周：2026 前沿与复现

阅读顺序：

1. Stable GFlowNets；
2. \(f\)-TB；
3. Learning Shortest Paths；
4. OT 论文；
5. 根据兴趣选 RapTB、PPO 或 GFlowRL。

复现分两档：

- **核心（必做，约 2–3 小时；即练习 10 的数值版）**：
  - 两个源状态、两个终止状态；
  - 手算最短路代价矩阵；
  - 用 `scipy.optimize.linprog` 或 POT 求 OT；
  - 解 edge-flow LP；
  - 比较两个最优值和诱导 coupling。
- **可选/延伸（+4–8 小时，适合准备沿 O07/O08 方向做研究的读者）**：
  - 神经 TB + flow regularization，观察近似误差和路径长度折中。

通过标准：

- 口头解释：能准确列出 OT 定理的全部前提，并给出一个不满足前提的普通 GFlowNet 反例。
- 数值对照：edge-flow LP 最优值与 Kantorovich OT 最优值在数值精度内相等（相对误差 \(<10^{-6}\)），且两侧诱导的 coupling 一致。

---

## 2. HyperGrid 最小起步配置【追加到 guide §12，作为新的 §12.0，置于 §12.1 之前】

### §12.0.1 最小配置表（torchgfn v2.4.1 实测默认值）

以下数值按 torchgfn `master`（v2.4.1，2026-04-05 发布）的 `tutorials/examples/train_hypergrid.py` 与 `train_hypergrid_simple.py` 核对（核验日期 2026-08-14）。第一次实验直接采用默认值即可复现下方参考量级。

| 参数 | 完整版脚本旗标 | 默认值 | 说明 |
|---|---|---|---|
| 维度 | `--ndim` | 2 | 2D 便于精确枚举与画图 |
| 边长 | `--height` | 8 | `train_hypergrid_simple.py` 默认为 32 |
| 奖励基底 | `--R0` | 0.1 | 越小探索越难（TB 论文用 0.1/0.01/0.001 三档） |
| 奖励台地 | `--R1` | 0.5 | 见下方奖励公式 |
| 奖励峰 | `--R2` | 2.0 | mode 处奖励为 \(R_0+R_1+R_2=2.6\) |
| loss | `--loss` | TB | 可选 FM/TB/DB/SubTB/ZVar/ModifiedDB |
| batch | `--batch_size` | 16 | 每次迭代采 16 条轨迹 |
| 策略学习率 | `--lr` | \(10^{-3}\) | AdamW，`--weight_decay` 1e-4 |
| \(\log Z\) 学习率 | `--lr_Z` | \(10^{-1}\) | **注意旗标名**：simple 版脚本为 `--lr_logz`，均为策略学习率的 100 倍 |
| 轨迹总预算 | `--n_trajectories` | \(10^6\) | 教学场景 1.6 万条即可看到收敛趋势 |
| replay | `--replay_buffer_size` | 2048 | 默认启用**普通 FIFO** `ReplayBuffer`；prioritized 需手动换成 `NormBasedDiversePrioritizedReplayBuffer` |
| 探索 | `--epsilon` / `--temperature` | 0.0 / 1.0 | 默认纯 on-policy |
| 网络 | `--hidden_dim` / `--n_hidden` | 256 / 3 | MLP，K-hot 输入；simple 版为 256 / 2 |
| SubTB | `--subTB_lambda` / `--subTB_weighting` | 0.9 / geometric_within | 仅 `--loss SubTB` 时生效 |
| 评估 | `--validate_environment --validation_interval 100 --validation_samples 200000` | 关 / 100 / 200000 | `l1_dist` \(=\sum_x|\hat P(x)-P^\star(x)|=2\,\mathrm{TV}\)，由 20 万条新采样估计 |

奖励公式（torchgfn "original" 奖励，与 TB 论文一致）：

\[
R(s)=R_0+R_1\prod_{d=1}^{D}\mathbf 1\!\left(\left|\tfrac{s^d}{H-1}-0.5\right|\in(0.25,0.5]\right)+R_2\prod_{d=1}^{D}\mathbf 1\!\left(\left|\tfrac{s^d}{H-1}-0.5\right|\in(0.3,0.4)\right).
\]

**环境版本注意事项**（均为 2026-08-14 实测）：

- torchgfn v2 与 v1 API 不兼容，文档以 [torchgfn.readthedocs.io](https://torchgfn.readthedocs.io/en/latest/) 为准；
- 建议 Python 3.10–3.13。**Python 3.14 下 v2.4.1 的 HyperGrid 全状态枚举会崩溃**（内部多进程返回 `itertools.islice`，而 3.14 移除了 itertools 对象的 pickle 支持）；如必须用 3.14，可把 `HyperGrid._generate_combinations_in_batches` 替换为等价的串行生成器绕过。

### §12.0.2 参考结果量级

单种子、CPU、全默认超参（`train_hypergrid_simple.py`，TB，batch 16，1000 次迭代 = 1.6 万条轨迹），`l1_dist` 含 20 万采样的估计噪声（\(8\times8\) 时噪声下限约 0.014），锚点取区间而非精确值：

| 配置 | \(L_1=2\,\mathrm{TV}\) 演化 | mode 状态覆盖 | 实测耗时（M2 Max，CPU） |
|---|---|---|---|
| 2D \(8\times8\)，\(R_0=0.1\) | 0.34（1.6k 轨迹）→ 0.09（3.2k）→ **0.02–0.06 区间波动（≥4.8k，首次 <0.05 在约 4.8k）** | 4/4，1.6k 轨迹内全部发现 | 约 1 分 43 秒（含每 100 迭代一次的 20 万样本评估） |
| 2D \(32\times32\)，\(R_0=0.1\) | 1.15（1.6k）→ 0.41（4.8k）→ **0.16（16k，仍在下降）** | 36/36，4.8k 轨迹内全部发现 | 约 5 分 48 秒 |

文献对照：TB 论文 Fig. 2 在 4D \(8^4\) 与 2D \(64\times64\)、\(R_0\in\{0.1,0.01,0.001\}\) 上给出 \(L_1\) 随训练轨迹数的曲线（DB/TB 快于 FM；\(64\times64\) 上学习 \(P_B\) 明显快于 uniform \(P_B\)），可作为更大网格的量级参照。

### §12.0.3 计算资源与预计耗时

全部第 2、3 周实验（可枚举 HyperGrid 上六组对比 + 多种子）**在笔记本 CPU 上即可完成，无需 GPU、无需排队申请算力**：

- 教学预算（1.6 万条轨迹/组）：每组 2–6 分钟（本机实测，见上表）；六组对比 × 3 种子约 1–2 小时；
- 论文级预算（\(10^6\) 条轨迹/组）：TB 论文附录 B.1 报告单 CPU 核约 2 小时/组，其全部 24 组 DB/TB 设置 × 5 种子合计约 10 CPU 天——按教学预算裁剪即可。

### §12.0.4 guide §12.2 对比组 → 脚本旗标映射

| §12.2 对比组 | `train_hypergrid.py` 旗标 |
|---|---|
| FM | `--loss FM` |
| DB + fixed uniform \(P_B\) | `--loss DB --uniform_pb` |
| DB + learned \(P_B\) | `--loss DB` |
| TB | `--loss TB` |
| SubTB(\(\lambda\)) | `--loss SubTB --subTB_lambda 0.9` |
| TB + prioritized replay | `--loss TB --replay_buffer_size 2048` 并把脚本中 `ReplayBuffer` 换成 `NormBasedDiversePrioritizedReplayBuffer`（一行改动） |

### 两处正文修订

- 【替换 guide §12.1 第 4 条】原文"至少设计一组多路径状态合并"改为：**"确认所选环境具有多路径状态合并（HyperGrid 天然满足：到达内部格点 \((s^1,\dots,s^D)\) 的单调路径数为多项式系数 \(\tbinom{s^1+\cdots+s^D}{s^1,\dots,s^D}\)，无需额外改造环境），并在报告中说明其规模。"**
- 【替换 guide §12.4"分开优化 \(\log Z\) 与策略"一条】改为：**"分开优化 \(\log Z\) 与策略时，记录各自学习率和梯度尺度。经验值：\(\log Z\) 学习率取策略学习率的约 100 倍（torchgfn 默认 \(10^{-1}\) vs \(10^{-3}\)；TB 论文附录 B.1 同此设置，并报告 \(10^{-3}\) 是策略端不发生模式塌缩的最大学习率）。这是 TB 能否收敛的第一敏感项，初学者最容易在此卡住。"**

---

## 3. 十个练习：元数据与自检要点【追加到 guide §11；练习 2、9 题面按 3.2 节替换】

### 3.1 元数据总表【追加到 guide §11 引言处】

按"热身 → 核心 → 诊断 → 进阶"分组做，可修复原顺序难度不单调的问题；与六周计划互链，避免隐性重复劳动。

| 练习 | 难度 | 依赖章节 | 对应周次 | 分组 |
|---|---|---|---|---|
| 1 两条路径、一个终点 | ★ | §2.3–2.5 | 第 1 周（≈该周手推 3） | 热身 |
| 3 TB 的 telescoping | ★ | §3.2–3.4 | 第 2 周（=该周手推 1） | 热身 |
| 7 reward scale 与温度 | ★ | §2.3、§13 | 第 1–2 周任意 | 热身 |
| 2 路径数偏置 | ★★ | §2.4、§5.1 | 第 4 周（=该周手推 4） | 核心 |
| 4 固定 \(P_B\) 的唯一 flow | ★★ | §2.6 | 第 2–3 周 | 核心 |
| 5 相同奖励分布、错误对象分布 | ★★ | §4.3 | 第 3 周（可选实验的解析版） | 核心 |
| 6 off-policy 支持反例 | ★★ | §4.1 | 第 3 周 | 诊断 |
| 8 TV 与极端 TB loss | ★★ | §4.4–4.5 | 第 3–4 周 | 诊断 |
| 9 最小流即最短路 | ★★★ | §5.5、§6.4 | 第 5–6 周 | 进阶（OT） |
| 10 从最短路到 OT | ★★★ | §5.5 | 第 6 周（=该周核心复现的解析版） | 进阶（OT） |

### 3.2 练习 2 与练习 9 的题面补丁【分别替换 guide §11 对应题面】

**练习 2：路径数偏置（补前提与构造提示）**

令两个终点 \(x_1,x_2\) 奖励相同，但分别有 1 条和 10 条等长路径。**精确定义两种被比较的采样分布**：(a) *均匀轨迹采样*——在全部完整轨迹的集合上取均匀分布，终点概率正比于其路径数（这也是朴素 MaxEnt RL 的轨迹级最优分布 \(\pi(\tau)\propto R(x_\tau)\) 在等奖励时的特例：熵最大化作用在轨迹上，不做路径数校正）；(b) *均匀前向策略*——在每个状态对合法动作取均匀 \(P_F\)，终点概率由图内部分叉结构决定，与 (a) 一般不同。分别计算两者的终点分布，再与 GFlowNet reward matching 比较。

*构造提示*：格点图从 \((0,0)\) 到 \((2,3)\) 的单调路径恰有 \(\binom{5}{2}=10\) 条；再从 \(s_0\) 挂一条长度 5 的单链到 \(x_1\)，即得"1 条 vs 10 条等长路径"。

**练习 9：最小流即最短路（补设定声明与提示）**

**设定**：在 §5.5 的非无环、expected-visit-count 流意义下（即 §6.4/O07 的框架，图允许环），固定一个源与一个终点。证明在所有能送 1 单位质量的可行流中，最小总内部边流等于最短路长度。

*提示*：考虑流分解定理——任何可行流可分解为若干条源到汇的路径流加若干环流；环流只增加总边流而不运送质量，故最优解不含环；剩下的每条路径长度不小于最短路长度。

### 3.3 每题自检要点【追加到 guide §11 末尾，作"自检要点"小节】

做完后对照关键中间结论；对不上先查自己的定义，再查计算。

- **练习 1**：全族流都满足守恒且 \(P_T(x)=1\)，故全族 reward-matching；\(P_B(a|x)=q\)、\(P_B(b|x)=1-q\)（终点入流归一化）；\(q\in(0,1)\) 才有非退化局部策略，端点值产生零流状态。
- **练习 2**：均匀轨迹采样给出 \(P(x_2)=10/11\)（10 倍偏置）；reward matching 给出 \(1/2\)-\(1/2\)，与路径数无关；均匀前向策略的答案依赖图内部分叉结构，一般与两者都不同——这正是必须先定义分布再计算的原因。
- **练习 3**：对每条边写 \(F(s_t)P_F(s_{t+1}|s_t)=F(s_{t+1})P_B(s_t|s_{t+1})\)，沿轨迹连乘后中间 \(F(s_t)\) 成对消去，剩 \(Z\prod P_F=R(x)\prod P_B\)；消去成立要求 DB 在轨迹每条边上同时成立，且 \(F(s_0)=Z\)。
- **练习 4**：逆拓扑序递推 \(F(s\to s')=F(s')P_B(s|s')\) 从 \(F(x)=R(x)\) 出发唯一确定全部边流；\(F(s)=\sum_{s'}F(s\to s')\)，\(P_F(s'|s)=F(s\to s')/F(s)\)；对 \(s_0\) 求和应恰好回收 \(Z=\sum_xR(x)\)。
- **练习 5**：目标 \(P^\star=(\tfrac1{22},\tfrac1{22},\tfrac{10}{22},\tfrac{10}{22})\)；构造如 \((\tfrac2{22},0,\tfrac{20}{22},0)\)——reward histogram 与目标完全相同（两奖励档质量 \(\tfrac2{22},\tfrac{20}{22}\) 不变），但对象级 \(\mathrm{TV}=\tfrac12\)。结论：histogram 对同奖励档内部偏置零灵敏。
- **练习 6**：在唯一被覆盖的路径上，调 \(\log Z\) 与 \(P_F\) 可使 TB 残差恰为 0；未覆盖分支上 \(P_F\) 完全自由，终点分布误差可任意大。结论：定理里的 full support（训练分布对所有轨迹正概率）不可删除。
- **练习 7**：乘正常数 \(c\) 在归一化中消去，目标分布不变；而 \(R^\beta\) 改变相对质量——\(\beta\uparrow\) 分布变尖、熵变小、低奖励模式被压缩。两者是"不变量 vs 温度旋钮"的关系。
- **练习 8**：新模式目标质量 \(\varepsilon\)、模型给 \(\delta\ll\varepsilon\)：TV 增量 \(\le\varepsilon\)（可任意小），但该模式轨迹的 log-ratio \(\sim\log(\varepsilon/\delta)\) 随 \(\delta\to0\) 无界，对应 TB loss \(\sim\log^2(\varepsilon/\delta)\)。结论：TV 小与 TB loss 无界可共存（§4.4 的反方向也成立）。
- **练习 9**：三步——(i) 沿一条最短路送 1 单位得可行流，总边流 \(=d(s,t)\)；(ii) 任意可行流按流分解定理拆成路径 + 环，环只增流量不助运输，去掉不失可行性；(iii) 每条承载路径长度 \(\ge d(s,t)\)，故总边流 \(\ge d(s,t)\)。
- **练习 10**：两个方向——coupling \(\to\) flow：把每对 \((u,x)\) 的质量 \(\Pi(u,x)\) 沿任一条 \(u\rightsquigarrow x\) 最短路发送，总边流 \(=\sum_{u,x}d(u,x)\Pi(u,x)\)；flow \(\to\) coupling：令 \(\Pi(u,x)=\Pr(\tau\text{ 从 }u\text{ 到 }x)\)，边缘自动为 \(L,R\)，且每条实际路径 \(\ge d(u,x)\) 给出反向不等式；两侧夹逼即 §5.5 定理。

---

## 4. torchgfn 三步代码走读（30–60 分钟）【追加到 catalog §7"第一份复现实验"之前；guide §9.2 末尾加一行互链】

只读三处文件即可建立"库怎么把 §2–§3 的数学变成代码"的完整图景。文件路径按 `master`（v2.4.1）核验有效（2026-08-14）。

| 步骤 | 文件 | 时长 | 看什么 |
|---|---|---|---|
| 1 | [`tutorials/examples/train_hypergrid_simple.py`](https://github.com/GFNOrg/torchgfn/blob/master/tutorials/examples/train_hypergrid_simple.py)（约 240 行） | 10–15 分钟 | 最小闭环：`HyperGrid` 环境构造 → `DiscretePolicyEstimator`（\(P_F,P_B\)）→ `TBGFlowNet` → `Sampler` 采轨迹 → `loss_from_trajectories` → `env.validate` 报 `l1_dist`。重点看 \(\log Z\) 单独一个参数组、学习率 `--lr_logz 1e-1`。注意：脚本 docstring 仍写"仅 TB"，实际已支持 FM/TB/DB 三种 loss |
| 2 | [`tutorials/examples/train_hypergrid.py`](https://github.com/GFNOrg/torchgfn/blob/master/tutorials/examples/train_hypergrid.py)（约 720 行） | 15–20 分钟 | 完整实验件：`set_up_gflownet` 按 `--loss` 分派六种目标；`ReplayBuffer`；`get_exact_P_T` 用 \(u(s')=\sum_{s\in\mathrm{Par}(s')}u(s)P_F(s'|s)\) 精确枚举终止分布；`--plot` 画真实/学习分布对比；docstring 顶部就是 TB 论文与 SubTB 论文的复现命令行 |
| 3 | [`src/gfn/gflownet/`](https://github.com/GFNOrg/torchgfn/tree/master/src/gfn/gflownet) loss 类 | 15–25 分钟 | 对照 guide §3 逐个核对：`flow_matching.py`（§3.1）、`detailed_balance.py`（§3.2，含 ModifiedDB）、`trajectory_balance.py`（§3.3，含 ZVar）、`sub_trajectory_balance.py`（§3.5，`lamda` 与加权方案）、`base.py`（`GFlowNet` 抽象与 `loss_from_trajectories` 接口） |

走读自检：能回答"(1) \(\log Z\) 存在哪个对象里、为什么它需要 100 倍学习率；(2) `get_exact_P_T` 的递推与练习 4 的逆拓扑递推是什么关系（提示：一个沿 \(P_F\) 正向、一个沿 \(P_B\) 反向）；(3) SubTB 的 `weighting=geometric_within` 加权对应论文哪个公式"。

---

## 5. 资源修正清单【catalog 行级修订】

### 5.1 catalog §7 代码表修订

| 动作 | 内容 |
|---|---|
| **替换（P0 教程行）** | 原"Mila Workshop notebooks（josephdviviano/torchgfn-tutorials）"P0 行替换为：**P0 · [torchgfn 官方 notebooks](https://github.com/GFNOrg/torchgfn/tree/master/tutorials/notebooks)** —— 随库 CI 持续测试，含 `getting_started.ipynb`（HyperGrid FM 入门）、`intro_discrete.ipynb`、`intro_graphs.ipynb`（graph GFlowNet）、`intro_continuous.ipynb`、`trust_pcl_equivalence.ipynb` 等 9 个 notebook。原仓库已改名为 `josephdviviano/gflownet-tutorials`（旧 URL 靠重定向存活），2023-11-09 后停更、基于 v1 旧 API，**降级 P2 历史材料或直接删除** |
| **替换（torchgfn 行文档链接）** | `gfn.readthedocs.io`（v1 旧文档）→ [`torchgfn.readthedocs.io/en/latest/`](https://torchgfn.readthedocs.io/en/latest/)；备注补"当前 v2.4.1（2026-04-05）；v1/v2 API 不兼容，勿按旧文档学新库"。（guide §9.2 已是新链接，无需再改） |
| **修订（gfnx 行）** | 补库论文 [gfnx: Fast and Scalable Library for Generative Flow Networks in JAX（arXiv:2511.16592）](https://arxiv.org/abs/2511.16592)与[文档](https://gfnx.readthedocs.io/en/latest/)链接 |
| **增补（新 P1 行）** | **P1 · [alexhernandezgarcia/gflownet](https://github.com/alexhernandezgarcia/gflownet)** · 适合任务：科学发现应用、连续/混合空间、多目标 GFN · 注意：P0 课程 IFT 6760B 讲者本人的研究框架库（课程 Resources 页列为配套库），Mila 多篇论文内部使用；活跃维护（核验时最近提交 2026-08-13） |

### 5.2 讲座视频列【追加到 catalog §6.2 之后，作新的 §6.4"论文讲座视频"；或并入 §1 论文表新增一列】

P0 论文精读前先看 5–15 分钟作者讲解，是最高性价比的预热。以下链接均已核验含视频（SlidesLive 内嵌，会议虚拟站点，免注册）：

| 编号 | 论文 | 讲座视频 |
|---|---|---|
| T01 | GFlowNet 原始论文 | [NeurIPS 2021 virtual poster](https://neurips.cc/virtual/2021/poster/26729) |
| T03 | Trajectory Balance | [NeurIPS 2022 virtual poster](https://neurips.cc/virtual/2022/poster/53114)（附官方 slides PDF） |
| T05 | SubTB | [ICML 2023 virtual poster](https://icml.cc/virtual/2023/poster/25261)；另有 [Oral A6 场次](https://icml.cc/virtual/2023/session/25589) |
| T07 | Shen et al. 训练诊断 | [ICML 2023 virtual poster](https://icml.cc/virtual/2023/poster/24459) |
| T17 | Divergence Measures | [NeurIPS 2024 virtual poster](https://neurips.cc/virtual/2024/poster/95464) |
| T32 | When Do GFlowNets Learn the Right Distribution? | [ICLR 2025 virtual poster](https://iclr.cc/virtual/2025/poster/30708) |

另：catalog 已链接的 ICML/ICLR virtual 页（T47、T49、T50、T53–T56 等）本身自带视频，建议在对应行备注"页面含 talk 视频"。

### 5.3 提问渠道【追加到 catalog §9.1 追踪入口列表末尾】

- **提问渠道**：[torchgfn Issues](https://github.com/GFNOrg/torchgfn/issues) 与 [Discussions](https://github.com/GFNOrg/torchgfn/discussions)（实现/API 问题，维护者响应活跃）；[gfnx Issues](https://github.com/d-tiapkin/gfnx/issues)（JAX 侧）。自学卡住超过 1 小时的实现问题，先搜已关闭 issue，再开新帖附最小复现。

### 5.4 catalog §0.2 与 §1 标注修订

| 动作 | 内容 |
|---|---|
| **追加（§0.2 表格下方）** | 节奏前提声明：**"两周建立理论骨架"假设全职冲刺（每天 6–8 小时）；在读兼职（每周 6–10 小时）请直接采用理论指南 §10 的六周计划，两条路线覆盖同一批 P0 材料，选一即可。** |
| **替换（§0.2 OT 路线行）** | 顺序改为 **O01 → T02 → T36 → T19（可选深读）→ O02 → O07 → O08**。理由：O01 是纯 OT 先修（guide 第 0 周已要求 Kantorovich LP），应最先补齐；T36（有限离散、更简洁）先于 T19（一般可测空间、测度论重）符合"先具体后抽象"；只关心离散情形的读者可将 T19 降为选读 |
| **替换（§1 T54 行建议列）** | 双值标注"P0/P1"改为 **P1（LLM/序列方向读者视为 P0）**，消除与 §0.1 标记语义的冲突 |
| **替换（§1 T12 行建议列）** | "P0 精读"改为 **P0-if-continuous/OT（不做连续空间或 OT 方向可降为 P1）**，把无条件 P0 论文收敛到 10–12 篇 |

---

## 6. 核验记录（本文件内部附录，不落地到正式文档）

| 事实 | 核验方式（2026-08-14） |
|---|---|
| torchgfn v2.4.1（2026-04-05 发布） | GitHub Releases API + PyPI 实际安装 |
| `train_hypergrid.py` / `train_hypergrid_simple.py` 路径与全部默认超参 | 下载 `master` 原文件逐行核对 |
| `src/gfn/gflownet/` 含 FM/DB/TB/SubTB/losses/base/mle 模块 | GitHub contents API |
| `tutorials/notebooks/` 含 getting_started 等 9 个 notebook | GitHub contents API |
| `l1_dist` \(=\sum_x|\hat P-P^\star|\)、由 20 万新采样估计 | 读 v2.4.1 安装包 `gfn/env.py` 源码 |
| 参考结果两行（8×8 / 32×32） | 本机实测（M2 Max，CPU，单种子，Python 3.14 + 串行枚举补丁；补丁不改训练逻辑） |
| Python 3.14 崩溃（itertools pickle） | 本机实测复现 traceback |
| CPU 2 小时/\(10^6\) 轨迹、logZ 学习率 100 倍、\(10^{-3}\) 稳定上界 | TB 论文（arXiv:2201.13259）附录 B.1 原文 |
| 旧教程仓改名 gflownet-tutorials、2023-11-09 停更 | GitHub repo API |
| alexhernandezgarcia/gflownet 活跃（2026-08-13 push）、课程 Resources 页列为配套库 | GitHub repo API + 课程页抓取 |
| gfnx 论文 arXiv:2511.16592 与 readthedocs | arXiv 页 + HTTP 200 |
| 6 条讲座视频链接 | 会议虚拟站点逐页抓取，确认含 Video/SlidesLive |
| torchgfn Discussions 已启用 | GitHub 仓库页抓取 |

