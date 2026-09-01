# GFlowNet 探索策略谱系

> 本文扩展理论指南训练/探索部分，配合资料清单探索线论文卡片阅读。
> 来源：GFlowNet 调研 2026-08 审查扩充（E07）。核心索引见 [README](README.md)。

---

> 联网核验日期：2026-08-14；venue 均以 OpenReview / 会议官方 virtual 页 / PMLR / AAAI proceedings 为准，arXiv 元数据仅作辅助。
> 动机：审查指出探索线缺演化史——GAFN（内在奖励起点）、Thompson（workshop）、Adaptive Teachers（辅助网络起点）等前身缺失，使 2026 方法（ACE、f-TB、α-GFN）看起来"凭空出现"。本节按**"探索信号注入到哪里"**把方法排成一条谱系，并逐支给出与"分布正确性"（是否仍采样 \(P^\star(x)=R(x)/Z\)）的张力。

## 1. 谱系树

组织主轴不是时间，而是**探索信号注入的位置**：注入到行为/数据采集策略的方法几乎都"免费"（不改终止分布），注入到奖励的方法必然改目标，注入到 loss/目标结构的方法则介于两者之间。

```text
GFlowNet 探索谱系（按"信号注入位置"）
│
├─[行为策略·加噪][基线] ε-greedy / 前向策略 tempering
│     Bengio 2021 (NeurIPS)、Malkin 2022 TB (NeurIPS)
│     └ 只改数据采集分布 → 由 off-policy 一致性保证不改终止分布
│
├─[奖励侧] 内在奖励增广 —— GAFN (ICLR 2023 Spotlight, 2210.03308)
│     └ 改 reward=R+r → 改目标；仅当内在奖励退火到 0 时"渐近"无偏 (Thm.1)
│
├─[行为策略·后验] Thompson 采样 TS-GFN (ICML 2023 SPIGM Workshop, 2306.17693)
│     └ 前向策略集成→近似后验→按后验挑行为策略；仍是 off-policy，目标不变
│
├─[回放] 优先/结构化回放 —— PRT (Shen 2023, ICML) → submodular replay:
│     RapTB (ICML 2026, T54)、Signal-from-Structure (预印本 2026, 2601.21061)
│     └ 复用历史轨迹（off-policy 合法）；优先级/窄支撑损害"有限训练"覆盖，但不改最优目标
│
├─[辅助网络找盲区] 双网络解耦谱系（本条是审查点名的"断代"主线）：
│     Adaptive Teachers (ICLR 2025, 2410.01432)
│       → Sibling / SA-GFN (ICLR 2025)
│       → Loss-Guided / LGGFN (AAAI 2026, 2505.15251)
│       → ACE / Divergent-TB (ICML 2026, 2602.17827)
│     └ 主网络照常拟合 R/Z，副网络只塑造"去哪采样" → 目标不变，各自附探索性证明
│
└─[损失/目标结构] 探索内置于 loss/目标：
      f-TB (ICML 2026, 2605.15417) —— off-policy 同一全局最小值，可证保持目标
      α-GFN (预印本 2026, 2602.01749) —— 前后向混合，收敛到唯一流；R∝ 仅经验支持
```

## 2. 分支详解

### 2.1 基线：ε-greedy / 前向策略 tempering
- **核心机制**：数据采集时对前向策略加均匀噪声（ε-greedy）或调高采样温度使 \(P_F\) 变平，扩大轨迹覆盖；训练 loss 仍是标准 TB/DB。
- **代表**：Bengio et al. 2021（NeurIPS 2021，T01）、Malkin et al. TB 2022（NeurIPS 2022，T03）。
- **与分布正确性的张力**：**保持目标（有理论依据）**。TB/DB/SubTB 是 off-policy 一致的——只要行为策略**全支撑**，其唯一全局最小值即 \(P_F^\top(x)=R(x)/Z\)（Foundations, JMLR 2023, T02）。代价：温度过高会拖慢收敛，且"全支撑"在大空间只是理想假设。

### 2.2 奖励侧：内在奖励增广（GAFN）
- **核心机制**：用 RND 等好奇心信号构造中间/终点内在奖励，以"增广流"注入网络，为稀疏奖励提供密集反馈；把轨迹回报重定义为终点奖励 + 内在奖励。
- **代表**：Generative Augmented Flow Networks，Pan et al.，**ICLR 2023 Spotlight**，arXiv:2210.03308。
- **与分布正确性的张力**：**默认不保持**。改的是有效奖励 \(R+r\)，训练中的目标是被扰动的。其 Theorem 1 只给**条件+渐近**保证：需 \(L_\text{GAFN}\equiv0\)、\(R(x)+r(x)>0\)，**且边级内在奖励收敛到 0**（RND 随熟悉度衰减）时才渐近还原 \(R/Z\)；论文亦自承纯 state-based 增广"拟合不好目标分布"。这是整条谱系后续"把探索信号搬出奖励"的直接动因。

### 2.3 行为策略·后验：Thompson 采样（TS-GFN）
- **核心机制**：用前向策略头的集成维护一个策略后验，把"训练该采哪条轨迹"当主动学习问题，用 Thompson 采样从后验挑行为策略采数据。
- **代表**：Thompson Sampling for Improved Exploration in GFlowNets，Rector-Brooks et al.，**ICML 2023 SPIGM Workshop（非主会）**，arXiv:2306.17693。
- **与分布正确性的张力**：**保持目标**。它显式利用"GFN 可稳定 off-policy 运行"这一性质，只替换行为策略、不动奖励与训练目标，故终止分布不变；代价约 +15% 计算。

### 2.4 回放：优先 / 次模（submodular）回放
- **核心机制**：把高奖励/高信息轨迹存入 buffer 反复 off-policy 重用。PER（Schaul et al., ICLR 2016）是 RL 起点；GFN 版 PRT（优先回放训练）在 Shen 2023 提出；submodular/结构化回放用多样性目标维护 buffer，抵消优先级带来的支撑收窄。
- **代表**：PRT——Shen et al.，ICML 2023（T07）；submodular replay——RapTB，ICML 2026（T54）；Signal-from-Structure（次模上界乐观探索），预印本 2026，arXiv:2601.21061。
- **与分布正确性的张力**：**最优处保持，有限训练有风险**。回放本身 off-policy 合法、不改最优目标；但优先级会偏置经验梯度权重、窄化有效支撑，在有限预算下加剧"low-loss / high-TV"（连 T32）。次模回放正是为对冲此风险而生。

### 2.5 辅助网络找盲区：teacher / sibling / loss-guided / ACE
- **核心机制**：维护第二张 GFN 专门去主模型的"盲区"采数据，回流给主模型。四代演化差别在**"盲区"如何定义**：Adaptive Teachers 用 teacher 采 student **高 loss** 区（reward=student loss）；Sibling 用解耦副网络 + 内在奖励采**新颖**区；Loss-Guided 把辅助奖励写成 \(R_\text{aux}=R+\lambda L_\text{main}\)（loss 直接当信号，比 novelty 更精准）；ACE 用 Divergent-TB 让探索网络采 canonical 网络**欠采样但高奖励**的区域。
- **代表**：Adaptive Teachers for Amortized Samplers，Kim et al.，**ICLR 2025 Poster**，arXiv:2410.01432；Sibling/SA-GFN，Madan et al.，**ICLR 2025 Poster**；Loss-Guided/LGGFN，Malek et al.，**AAAI 2026**，arXiv:2505.15251；ACE，**ICML 2026（主会 Poster）**，arXiv:2602.17827。
- **与分布正确性的张力**：**保持目标（构造上 + 各自证明）**。主/canonical 网络始终用标准 off-policy 目标拟合 \(R/Z\)，副网络只改"去哪采"；ACE 进一步**证明** DTB 最小化会把探索策略推离被过采样的轨迹，给互补探索一个严格依据。共同代价：训练开销近乎翻倍（两张网络），且激进的 teacher 仍受 2.4 的有限训练覆盖问题约束。

### 2.6 损失/目标结构：把探索写进 loss（f-TB、α-GFN）
- **核心机制**：不加网络、不改数据，直接改 loss 的几何或目标的流平衡结构来调节探索-利用。f-TB 把 TB 的平方误差推广到整个 \(f\)-散度族；α-GFN 用参数 \(\alpha\) 混合前后向策略 \(P_\alpha=\alpha P_F+(1-\alpha)P_B\)，从"平衡流"放宽到"非平衡流"。
- **代表**：\(f\)-Trajectory Balance，Silva et al.，**ICML 2026（主会 Poster）**，arXiv:2605.15417；α-GFN（Controlling Exploration-Exploitation via Markov Chain Perspectives），Chen et al.，**预印本 2026（投 ICLR 2026，未见接收确认）**，arXiv:2602.01749。
- **与分布正确性的张力**：**分层**。f-TB **可证保持目标**——on-policy 梯度等于所选 \(f\)-散度（可调 mode-covering/-seeking），但 **off-policy 仍是同一个全局最小值**（Prop.），即探索偏好只进 loss 几何、不进目标。α-GFN 只证了"**收敛到唯一流**"（借 MC 可逆性），对 \(\alpha\neq0.5\) 是否仍严格 \(P_F^\top(x)\propto R(x)\)**只有经验支持**（Spearman 与 vanilla 相当，且作者自陈与 mode 发现"非严格耦合"，个别设置相关性下降 0.03）；作者以"α 调的是学习过程、不像 reward tempering 那样改目标"来论证，但这是论证而非定理。

## 3. 谱系对比表

| 分支（信号注入位置） | 是否保持目标分布 \(R/Z\) | 依据 | 额外成本 | 代表作 · venue |
|---|---|---|---|---|
| ε-greedy / 前向 tempering（行为策略） | 保持 | off-policy 一致性，需全支撑（T02 Foundations 定理） | ≈0 | Bengio 2021 · NeurIPS 2021；Malkin TB 2022 · NeurIPS 2022 |
| 内在奖励增广 GAFN（奖励侧） | **不保持**；内在奖励→0 时**渐近**无偏 | GAFN Thm.1（条件+渐近） | RND 网络 + 中间奖励 | GAFN · **ICLR 2023 Spotlight** (2210.03308) |
| Thompson TS-GFN（行为策略·后验） | 保持 | 纯行为策略替换，off-policy 有效 | 策略头集成，≈+15% 计算 | TS-GFN · **ICML 2023 SPIGM Workshop** (2306.17693) |
| 优先/次模回放（回放数据） | 最优处保持；有限训练可能损覆盖 | off-policy 合法；无终止分布定理护栏 | replay buffer + 多样性维护 | PRT · ICML 2023 (T07)；RapTB · ICML 2026 (T54) |
| 辅助网络找盲区（副行为网络） | 保持 | 主网标准目标 + ACE 的 DTB 证明 | 第二张 GFN，训练≈翻倍 | Adaptive Teachers/Sibling · **ICLR 2025**；LGGFN · **AAAI 2026**；ACE · **ICML 2026** |
| f-TB（loss 几何） | **保持（可证）** | off-policy 同一全局最小值（Prop.） | ≈0（仅换 loss） | f-TB · **ICML 2026** (2605.15417) |
| α-GFN（目标流结构） | 收敛唯一流可证；\(R\propto\) 仅经验 | MC 可逆性→唯一流；Spearman 经验 | ≈0（换目标权重） | α-GFN · **预印本 2026** (2602.01749) |

## 4. 最重要的一个理论张力

**探索是否破坏 reward matching，取决于信号注入到"行为策略"还是"奖励"。** GFlowNet 平衡目标（TB/DB/SubTB）的 off-policy 一致性意味着：只要行为策略全支撑，其唯一全局最小值就是 \(R(x)/Z\)，**与用什么行为策略采数据无关**——所以注入到行为侧的探索（ε-greedy、tempering、Thompson、回放、以及 Adaptive Teachers→Sibling→Loss-Guided→ACE 整条辅助网络线）在最优处"免费"，有定理护栏。相反，注入到奖励侧（GAFN 内在奖励）**必然把目标改成 \((R+r)/Z\)**，只能靠把 bonus 退火到 0 来渐近还原（GAFN Thm.1）。因此整条现代谱系的**演化主线，本质是把探索信号从"奖励"迁出、搬进"辅助行为策略"或"loss 几何"，以在提升 mode 发现的同时守住分布正确性**——这正是 2023→2026 方法演进的隐藏驱动力。第二层张力是："最优处无偏"≠"有限训练无害"：激进探索（窄 teacher、强优先回放）在有限预算下仍会加剧 low-loss/high-TV（T32），这也是次模回放与"互补而非贪婪"式 ACE 出现的原因。

## 5. 开放问题

1. **探索强度 ↔ 终止分布误差的定量界**：off-policy 一致性只在全局最优+全支撑成立，缺"探索激进度→有限训练 TV/L1"的非渐近界；把 Stable-GFN（T51）的 TB→TV 思路推广到"带探索的训练轨迹"是自然切口。
2. **奖励侧与行为侧的统一算子**：能否把 GAFN 的内在奖励安全改写为"不改目标"的行为策略（即 ACE/DTB 式互补探索），给出一个统一的"探索算子不改终止分布"的充分条件？ACE 作者自己把"非平稳 curiosity reward 能否进 DTB"列为 open。
3. **单网络能否吃掉双网络红利**：辅助网络谱系普遍训练开销翻倍；f-TB 的 loss 几何 / α 的流混合是否能在**单网络**下达到 teacher/ACE 级别的 mode 发现，从而免掉第二张网络？
4. **α-GFN 类"目标结构"方法缺终止分布保持定理**：目前只证收敛到唯一流 + 经验 Spearman，需要 \(\alpha\neq0.5\) 时 \(P_F^\top(x)\propto R(x)\) 的严格充分条件或反例——否则它介于"保持"与"不保持"之间的暧昧状态无法进正文的强声明。
5. **探索评测协议缺位**：现有工作多报 mode count / discovery rate，与分布保真（TV/L1、Spearman）弱耦合（α-GFN 自陈二者"非严格耦合"）；亟需在可枚举任务上同时报"发现速度"和"终止分布误差"的统一协议，才能公平比较这七个分支。

---

### 附：venue 核验记录（2026-08-14 联网）

| 论文 | arXiv | 核验到的 venue | 来源 |
|---|---|---|---|
| GAFN | 2210.03308 | ICLR 2023 Spotlight | OpenReview `urF_CBK5XC0`；官方代码库 README |
| TS-GFN | 2306.17693 | ICML 2023 SPIGM **Workshop**（非主会） | icml.cc/virtual/2023；ML Anthology bibtex |
| Adaptive Teachers | 2410.01432 | ICLR 2025 Poster | OpenReview `BdmVgLMvaf`；proceedings.iclr.cc 2025 |
| Sibling / SA-GFN | — | ICLR 2025 Poster | OpenReview `HH4KWP8RP5`；iclr.cc/virtual/2025/poster/30233 |
| Loss-Guided / LGGFN | 2505.15251 | AAAI 2026（Vol.40 No.29） | ojs.aaai.org/…/39613；IP-Paris research portal |
| ACE / Divergent-TB | 2602.17827 | ICML 2026 主会 Poster | icml.cc/virtual/2026/poster/62783 |
| f-TB | 2605.15417 | ICML 2026 主会 Poster | icml.cc/virtual/2026/poster/61247 |
| α-GFN | 2602.01749 | 预印本 2026（投 ICLR 2026，未确认接收） | arXiv（2026-02）；OpenReview `tp9y1547fH`（Submitted） |

