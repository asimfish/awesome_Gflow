# GFlowNet 调研：全局导读、术语表与符号表

> 本文为整套调研的导读入口。
> 来源：GFlowNet 调研 2026-08 审查扩充（E20）。核心索引见 [README](README.md)。

---

> 生成日期：2026-08-14。基准版本：`docs/` 当日版本（GUIDE 1230 行、CATALOG 402 行、OT分析 272 行；GUIDE 较 R10 审查版新增 4 行，本页行号一律以当日版本为准）。
> 响应审查：R10 S1（GUIDE 无目录索引，浏览体验最大短板）、S2、S8（中文译名首现缺英文对照）；R06 MINOR（`d_G` vs `d`、\(P^\star\) 四种写法、expected visit counts 三种写法等术语/记号变体）。
> 纪律：本页未修改 `docs/` 下任何文件；全部锚点按 GitHub slugger 规则生成（与 docs 现有跨文档锚点同一规则，见 R10 M6）；数学记号与三份文档保持一致，行内用 \(\ \)、独立公式用 \[\ \]，不使用 `$`。
>
> 本页四个组件：**§1 全局导读**（三文档怎么配合）｜**§2 GUIDE 两级目录**（92 个锚点条目）｜**§3 术语表**（34 条 + 7 组变体归一化）｜**§4 符号表**（39 个 + 6 组双义提示）。

约定简称（与 R06/R10 一致）：**GUIDE** = `GFLOWNET_THEORY_GUIDE_CN.md`，**CATALOG** = `GFLOWNET_RESEARCH_RESOURCE_CATALOG_CN.md`，**OT分析** = `GFLOWNET_OT_POTENTIAL_ANALYSIS_CN.md`。

---

## 1. 全局导读：三份文档如何配合使用

### 1.1 谁读哪份

| 文档 | 一句话定位 | 主要内容 | 适合谁 |
|---|---|---|---|
| [GUIDE](../docs/GFLOWNET_THEORY_GUIDE_CN.md)（1230 行，16 章） | 理论正文 + 学习路线 | 统一数学框架、六种训练目标、正确性定理与有限训练的鸿沟、与 RL/VI/MCMC/OT 的精确关系、六周学习路线、实验规范、误区速查 | 想系统建立理论、准备动手复现的读者 |
| [CATALOG](../docs/GFLOWNET_RESEARCH_RESOURCE_CATALOG_CN.md)（402 行，10 节） | 唯一论文目录 + 资源导航 | 100 篇编号论文（T01–T57 理论、O01–O08 OT 专题、A01–A35 应用）、三条推荐路线、顶会覆盖审计、课程/代码、论文精读模板 | 找论文、排 reading list、核验发表状态的读者 |
| [OT分析](../docs/GFLOWNET_OT_POTENTIAL_ANALYSIS_CN.md)（272 行，6 节） | O08（arXiv:2606.06272）单篇深读 | 定理重述与证明思路、真正新颖点、潜力论证、四大不足、八个后续方向评级、决定性实验协议 | 评估 GFlowNet×OT 研究方向、写研究计划的读者 |

三文档的配合原则（来自各自文头声明，R06 复核一致）：

- **CATALOG 是论文条目的唯一权威源**：每篇论文只在其主目录出现一次并配简介，GUIDE 与 OT分析 只用编号引用。查任何一篇论文，先去 CATALOG。
- **OT分析 是 GFN×OT 专题的完整版**，CATALOG [§2](../docs/GFLOWNET_RESEARCH_RESOURCE_CATALOG_CN.md#2-gflownet--optimal-transport-论文)、[§4](../docs/GFLOWNET_RESEARCH_RESOURCE_CATALOG_CN.md#4-gflownet--ot-是否很有潜力) 是其精简版。注意 CATALOG「最值得做的研究课题」表与 OT分析 [§5 评级表](../docs/GFLOWNET_OT_POTENTIAL_ANALYSIS_CN.md#5-我最看好的后续课题)是**互补的两张表**（CATALOG L235 已声明）：研究方向优先级以 OT分析为完整口径（首推 primal-dual GFlowNet OT）。
- **GUIDE [§5.5](../docs/GFLOWNET_THEORY_GUIDE_CN.md#55-与最优传输指定-2026-论文的准确解读) 负责 O08 定理的准确解读与限定**；完整推导与潜力评估在 OT分析。定理内容两处重述经 R06 逐项核对一致。
- 卫星页：`docs/README.md` 是入口页；《GFlowNet 分子应用全景》是 CATALOG §3.1（A01–A16）的专题扩展。两者不在本导读的三文档范围内。

### 1.2 三类读者的建议顺序

1. **第一次接触 GFlowNet（建立框架）**
   CATALOG [§0 先看这里](../docs/GFLOWNET_RESEARCH_RESOURCE_CATALOG_CN.md#0-先看这里)（标记说明 + 90 分钟入门三件套）→ GUIDE [先给结论](../docs/GFLOWNET_THEORY_GUIDE_CN.md#先给结论) 与 [§1](../docs/GFLOWNET_THEORY_GUIDE_CN.md#1-什么时候应该使用-gflownet)–[§2](../docs/GFLOWNET_THEORY_GUIDE_CN.md#2-统一数学框架) → GUIDE [§3](../docs/GFLOWNET_THEORY_GUIDE_CN.md#3-主要训练目标它们到底约束了什么)（读到 [§3.8 对照表](../docs/GFLOWNET_THEORY_GUIDE_CN.md#38-目标函数对照表)为止）→ 按 CATALOG [§0.2「两周建立理论骨架」路线](../docs/GFLOWNET_RESEARCH_RESOURCE_CATALOG_CN.md#02-三条推荐路线)精读 T01→T02→T03→T05→T07→T08→T09→T14→T17→T32 → 需要长期计划再进 GUIDE [§10 六周学习路线](../docs/GFLOWNET_THEORY_GUIDE_CN.md#10-六周学习路线)。
2. **要跑训练与复现实验（工程落地）**
   GUIDE [§3](../docs/GFLOWNET_THEORY_GUIDE_CN.md#3-主要训练目标它们到底约束了什么) 与 [§4](../docs/GFLOWNET_THEORY_GUIDE_CN.md#4-正确性定理与有限训练之间的鸿沟)（尤其 [§4.1 六条件](../docs/GFLOWNET_THEORY_GUIDE_CN.md#41-零-loss-推出正确分布真正需要什么)）→ GUIDE [§12](../docs/GFLOWNET_THEORY_GUIDE_CN.md#12-第一个严谨实验的建议配置)（环境、对比、[必报指标](../docs/GFLOWNET_THEORY_GUIDE_CN.md#123-必报指标)、[工程检查表](../docs/GFLOWNET_THEORY_GUIDE_CN.md#124-工程检查表)）→ CATALOG [§7 代码与复现](../docs/GFLOWNET_RESEARCH_RESOURCE_CATALOG_CN.md#7-代码与复现)（torchgfn + [第一份复现实验](../docs/GFLOWNET_RESEARCH_RESOURCE_CATALOG_CN.md#第一份复现实验)五步协议）→ 训练不符合预期时查 GUIDE [§13 误区速查](../docs/GFLOWNET_THEORY_GUIDE_CN.md#13-常见误区速查)与 [§4.2](../docs/GFLOWNET_THEORY_GUIDE_CN.md#42-shen-et-al-2023训练为什么会长期欠拟合)。
3. **评估 GFN×OT 方向、写研究计划（专题深入）**
   GUIDE [§5.5](../docs/GFLOWNET_THEORY_GUIDE_CN.md#55-与最优传输指定-2026-论文的准确解读)（定理准确解读 + [不能省略的限定](../docs/GFLOWNET_THEORY_GUIDE_CN.md#不能省略的限定)）→ OT分析 全文（[§1 定理](../docs/GFLOWNET_OT_POTENTIAL_ANALYSIS_CN.md#1-这篇论文究竟证明了什么) → [§4 不足](../docs/GFLOWNET_OT_POTENTIAL_ANALYSIS_CN.md#4-当前论文的主要不足) → [§5 评级](../docs/GFLOWNET_OT_POTENTIAL_ANALYSIS_CN.md#5-我最看好的后续课题) → [§6 决定性实验](../docs/GFLOWNET_OT_POTENTIAL_ANALYSIS_CN.md#6-最能验证这个方向的实验)）→ CATALOG [§2](../docs/GFLOWNET_RESEARCH_RESOURCE_CATALOG_CN.md#2-gflownet--optimal-transport-论文)（O01–O08 与[对 O08 的精读问题](../docs/GFLOWNET_RESEARCH_RESOURCE_CATALOG_CN.md#21-对-o08-的精读问题)）→ CATALOG [§4 课题表](../docs/GFLOWNET_RESEARCH_RESOURCE_CATALOG_CN.md#4-gflownet--ot-是否很有潜力)（与 OT分析 §5 互补使用）。

做组会汇报或 slides 的维护者，另见 `review/R10_structure_readability.md` 的 50 项高浓缩资产清单，以及 GUIDE [§14 一页式知识地图](../docs/GFLOWNET_THEORY_GUIDE_CN.md#14-一页式知识地图)与 [§16 最后五句话](../docs/GFLOWNET_THEORY_GUIDE_CN.md#16-最后应记住的五句话)。

### 1.3 两条使用纪律

- **发表状态口径**：三份文档一致坚持 Workshop 不写成主会（O08 = ICML 2026 SPIGM Workshop，非主会；R06 核对 5 处提及零偏差）。引用任何 2026 论文前，先看 CATALOG [§5 顶会覆盖审计](../docs/GFLOWNET_RESEARCH_RESOURCE_CATALOG_CN.md#5-20242026-顶会覆盖审计)的口径。
- **行号会漂移、锚点较稳**：`docs/` 仍在演进（R10 审查版 GUIDE 1226 行 → 当日 1230 行）。交叉引用优先使用章节锚点而非行号；本页行号仅作定位辅助。

---

## 2. GUIDE 两级目录（TOC）

> **覆盖**：GUIDE 全部 92 个正文标题（17 个 `##`：「先给结论」+ 16 个编号章节；75 个 `###`/`####`，其中第 10/11/13 章的周次、练习与误区以行内紧凑锚点列出）。
> **锚点规则**：GitHub slugger（小写、删标点与全角符号、空格转 `-`、保留 CJK 与 `_`/`-`；en/em dash 一并删除，如 `2024–2026` → `20242026`）。非 GitHub 渲染管线请使用 github-slugger 兼容的 anchor 插件（R10 M6）。
> **移植方法**：
> ① 整段并入 GUIDE「先给结论」之后（R10 S1 的建议位置）——把 `../docs/GFLOWNET_THEORY_GUIDE_CN.md#` 全局替换为 `#`；
> ② 并入 `docs/README.md`——把 `../docs/` 全局替换为空；
> ③ 保留在 `expansion/` 作独立导读页——链接现状即可直接使用。

- [先给结论](../docs/GFLOWNET_THEORY_GUIDE_CN.md#先给结论)
- [1. 什么时候应该使用 GFlowNet](../docs/GFLOWNET_THEORY_GUIDE_CN.md#1-什么时候应该使用-gflownet)
  - [1.1 合适的问题](../docs/GFLOWNET_THEORY_GUIDE_CN.md#11-合适的问题)
  - [1.2 不应默认使用的情形](../docs/GFLOWNET_THEORY_GUIDE_CN.md#12-不应默认使用的情形)
- [2. 统一数学框架](../docs/GFLOWNET_THEORY_GUIDE_CN.md#2-统一数学框架)
  - [2.1 状态、轨迹和终止对象](../docs/GFLOWNET_THEORY_GUIDE_CN.md#21-状态轨迹和终止对象)
  - [2.2 轨迹流、状态流和边流](../docs/GFLOWNET_THEORY_GUIDE_CN.md#22-轨迹流状态流和边流)
  - [2.3 流守恒与奖励匹配](../docs/GFLOWNET_THEORY_GUIDE_CN.md#23-流守恒与奖励匹配)
  - [2.4 前向策略、反向策略和核心定理](../docs/GFLOWNET_THEORY_GUIDE_CN.md#24-前向策略反向策略和核心定理)
  - [2.5 一个最小例子：正确终点不等于唯一内部流](../docs/GFLOWNET_THEORY_GUIDE_CN.md#25-一个最小例子正确终点不等于唯一内部流)
  - [2.6 反向策略不只是“倒着采样”](../docs/GFLOWNET_THEORY_GUIDE_CN.md#26-反向策略不只是倒着采样)
- [3. 主要训练目标：它们到底约束了什么](../docs/GFLOWNET_THEORY_GUIDE_CN.md#3-主要训练目标它们到底约束了什么)
  - [3.1 Flow Matching（FM）](../docs/GFLOWNET_THEORY_GUIDE_CN.md#31-flow-matchingfm)
  - [3.2 Detailed Balance（DB）](../docs/GFLOWNET_THEORY_GUIDE_CN.md#32-detailed-balancedb)
  - [3.3 Trajectory Balance（TB）](../docs/GFLOWNET_THEORY_GUIDE_CN.md#33-trajectory-balancetb)
  - [3.4 为什么 DB 可以推出 TB](../docs/GFLOWNET_THEORY_GUIDE_CN.md#34-为什么-db-可以推出-tb)
  - [3.5 Subtrajectory Balance（SubTB）](../docs/GFLOWNET_THEORY_GUIDE_CN.md#35-subtrajectory-balancesubtb)
  - [3.6 Guided Trajectory Balance（GTB）](../docs/GFLOWNET_THEORY_GUIDE_CN.md#36-guided-trajectory-balancegtb)
  - [3.7 \(f\)-Trajectory Balance](../docs/GFLOWNET_THEORY_GUIDE_CN.md#37-f-trajectory-balance)
  - [3.8 目标函数对照表](../docs/GFLOWNET_THEORY_GUIDE_CN.md#38-目标函数对照表)
- [4. 正确性定理与有限训练之间的鸿沟](../docs/GFLOWNET_THEORY_GUIDE_CN.md#4-正确性定理与有限训练之间的鸿沟)
  - [4.1 “零 loss 推出正确分布”真正需要什么](../docs/GFLOWNET_THEORY_GUIDE_CN.md#41-零-loss-推出正确分布真正需要什么)
  - [4.2 Shen et al. 2023：训练为什么会长期欠拟合](../docs/GFLOWNET_THEORY_GUIDE_CN.md#42-shen-et-al-2023训练为什么会长期欠拟合)
  - [4.3 这篇论文的评估边界](../docs/GFLOWNET_THEORY_GUIDE_CN.md#43-这篇论文的评估边界)
  - [4.4 2026：训练 loss 能否认证终止分布](../docs/GFLOWNET_THEORY_GUIDE_CN.md#44-2026训练-loss-能否认证终止分布)
  - [4.5 三种“误差”不要混为一谈](../docs/GFLOWNET_THEORY_GUIDE_CN.md#45-三种误差不要混为一谈)
- [5. GFlowNet 与其他方法的精确关系](../docs/GFLOWNET_THEORY_GUIDE_CN.md#5-gflownet-与其他方法的精确关系)
  - [5.1 与强化学习：相同的壳，不同的默认目标](../docs/GFLOWNET_THEORY_GUIDE_CN.md#51-与强化学习相同的壳不同的默认目标)
  - [5.2 与变分推断：轨迹空间上的近似推断](../docs/GFLOWNET_THEORY_GUIDE_CN.md#52-与变分推断轨迹空间上的近似推断)
  - [5.3 与 MCMC：同一未归一化目标，不同的成本位置](../docs/GFLOWNET_THEORY_GUIDE_CN.md#53-与-mcmc同一未归一化目标不同的成本位置)
  - [5.4 与 normalizing flows / flow matching](../docs/GFLOWNET_THEORY_GUIDE_CN.md#54-与-normalizing-flows--flow-matching)
  - [5.5 与最优传输：指定 2026 论文的准确解读](../docs/GFLOWNET_THEORY_GUIDE_CN.md#55-与最优传输指定-2026-论文的准确解读) ·
    附节 [不能省略的限定](../docs/GFLOWNET_THEORY_GUIDE_CN.md#不能省略的限定)
- [6. 理论扩展](../docs/GFLOWNET_THEORY_GUIDE_CN.md#6-理论扩展)
  - [6.1 条件 GFlowNet 与摊销边缘化](../docs/GFLOWNET_THEORY_GUIDE_CN.md#61-条件-gflownet-与摊销边缘化)
  - [6.2 中间奖励与不完整轨迹](../docs/GFLOWNET_THEORY_GUIDE_CN.md#62-中间奖励与不完整轨迹)
  - [6.3 连续状态空间](../docs/GFLOWNET_THEORY_GUIDE_CN.md#63-连续状态空间)
  - [6.4 非无环环境](../docs/GFLOWNET_THEORY_GUIDE_CN.md#64-非无环环境)
- [7. 经典文献脉络](../docs/GFLOWNET_THEORY_GUIDE_CN.md#7-经典文献脉络)
  - [7.1 奠基与训练目标](../docs/GFLOWNET_THEORY_GUIDE_CN.md#71-奠基与训练目标)
  - [7.2 与 VI、RL 和更一般空间的连接](../docs/GFLOWNET_THEORY_GUIDE_CN.md#72-与-virl-和更一般空间的连接)
  - [7.3 环、最短路与最优传输](../docs/GFLOWNET_THEORY_GUIDE_CN.md#73-环最短路与最优传输)
- [8. 截至 2026-07-31 的前沿](../docs/GFLOWNET_THEORY_GUIDE_CN.md#8-截至-2026-07-31-的前沿)
  - [8.1 稳定性和可认证性](../docs/GFLOWNET_THEORY_GUIDE_CN.md#81-稳定性和可认证性)
  - [8.2 统一 loss 与 divergence](../docs/GFLOWNET_THEORY_GUIDE_CN.md#82-统一-loss-与-divergence)
  - [8.3 前缀信用与 replay](../docs/GFLOWNET_THEORY_GUIDE_CN.md#83-前缀信用与-replay)
  - [8.4 探索—利用可控化](../docs/GFLOWNET_THEORY_GUIDE_CN.md#84-探索利用可控化)
  - [8.5 把成熟 RL 优化器迁入 GFlowNet](../docs/GFLOWNET_THEORY_GUIDE_CN.md#85-把成熟-rl-优化器迁入-gflownet)
  - [8.6 最小流、最短路和 OT](../docs/GFLOWNET_THEORY_GUIDE_CN.md#86-最小流最短路和-ot)
  - [8.7 大语言模型规模化](../docs/GFLOWNET_THEORY_GUIDE_CN.md#87-大语言模型规模化)
  - [8.8 2024–2026 顶会主线](../docs/GFLOWNET_THEORY_GUIDE_CN.md#88-20242026-顶会主线)
  - [8.9 当前真正开放的问题](../docs/GFLOWNET_THEORY_GUIDE_CN.md#89-当前真正开放的问题)
- [9. 博客、教程和代码资源怎么用](../docs/GFLOWNET_THEORY_GUIDE_CN.md#9-博客教程和代码资源怎么用)
  - [9.1 入门材料](../docs/GFLOWNET_THEORY_GUIDE_CN.md#91-入门材料)
  - [9.2 实现资源](../docs/GFLOWNET_THEORY_GUIDE_CN.md#92-实现资源)
- [10. 六周学习路线](../docs/GFLOWNET_THEORY_GUIDE_CN.md#10-六周学习路线)
  - 周次速查：[第 0 周 先修概念](../docs/GFLOWNET_THEORY_GUIDE_CN.md#第-0-周补齐先修概念) ·
    [第 1 周 流守恒与终止分布](../docs/GFLOWNET_THEORY_GUIDE_CN.md#第-1-周流守恒和终止分布) ·
    [第 2 周 DB/TB/SubTB](../docs/GFLOWNET_THEORY_GUIDE_CN.md#第-2-周dbtb-和-subtb) ·
    [第 3 周 训练行为与内部流](../docs/GFLOWNET_THEORY_GUIDE_CN.md#第-3-周训练行为与内部流) ·
    [第 4 周 VI/RL/MCMC](../docs/GFLOWNET_THEORY_GUIDE_CN.md#第-4-周virl-与-mcmc) ·
    [第 5 周 连续与非无环](../docs/GFLOWNET_THEORY_GUIDE_CN.md#第-5-周连续与非无环-gflownet) ·
    [第 6 周 前沿与复现](../docs/GFLOWNET_THEORY_GUIDE_CN.md#第-6-周2026-前沿与复现)
- [11. 建议完成的十个推导与实验](../docs/GFLOWNET_THEORY_GUIDE_CN.md#11-建议完成的十个推导与实验)
  - 练习速查：[1 两条路径](../docs/GFLOWNET_THEORY_GUIDE_CN.md#练习-1两条路径一个终点) ·
    [2 路径数偏置](../docs/GFLOWNET_THEORY_GUIDE_CN.md#练习-2路径数偏置) ·
    [3 TB 的 telescoping](../docs/GFLOWNET_THEORY_GUIDE_CN.md#练习-3tb-的-telescoping) ·
    [4 固定 \(P_B\) 的唯一 flow](../docs/GFLOWNET_THEORY_GUIDE_CN.md#练习-4固定-p_b-的唯一-flow) ·
    [5 相同奖励分布](../docs/GFLOWNET_THEORY_GUIDE_CN.md#练习-5相同奖励分布错误对象分布) ·
    [6 off-policy 反例](../docs/GFLOWNET_THEORY_GUIDE_CN.md#练习-6off-policy-支持反例) ·
    [7 reward scale 与温度](../docs/GFLOWNET_THEORY_GUIDE_CN.md#练习-7reward-scale-与温度) ·
    [8 TV 与极端 TB loss](../docs/GFLOWNET_THEORY_GUIDE_CN.md#练习-8tv-与极端-tb-loss) ·
    [9 最小流即最短路](../docs/GFLOWNET_THEORY_GUIDE_CN.md#练习-9最小流即最短路) ·
    [10 从最短路到 OT](../docs/GFLOWNET_THEORY_GUIDE_CN.md#练习-10从最短路到-ot)
- [12. 第一个严谨实验的建议配置](../docs/GFLOWNET_THEORY_GUIDE_CN.md#12-第一个严谨实验的建议配置)
  - [12.1 环境](../docs/GFLOWNET_THEORY_GUIDE_CN.md#121-环境)
  - [12.2 对比](../docs/GFLOWNET_THEORY_GUIDE_CN.md#122-对比)
  - [12.3 必报指标](../docs/GFLOWNET_THEORY_GUIDE_CN.md#123-必报指标)
  - [12.4 工程检查表](../docs/GFLOWNET_THEORY_GUIDE_CN.md#124-工程检查表)
- [13. 常见误区速查](../docs/GFLOWNET_THEORY_GUIDE_CN.md#13-常见误区速查)
  - 误区速查：[会均匀产生所有高奖励解](../docs/GFLOWNET_THEORY_GUIDE_CN.md#gflownet-会均匀地产生所有高奖励解) ·
    [乘常数更偏向高奖励](../docs/GFLOWNET_THEORY_GUIDE_CN.md#乘大一点-reward-会更偏向高奖励) ·
    [TB 比 DB 更正确](../docs/GFLOWNET_THEORY_GUIDE_CN.md#tb-比-db-理论上更正确) ·
    [任何 replay 都无偏](../docs/GFLOWNET_THEORY_GUIDE_CN.md#off-policy-不需要-importance-weights所以任何-replay-都无偏) ·
    [平均奖励达标即学对](../docs/GFLOWNET_THEORY_GUIDE_CN.md#平均奖励达到目标均值说明分布学对了) ·
    [保证快于 MCMC](../docs/GFLOWNET_THEORY_GUIDE_CN.md#gflownet-保证比-mcmc-更快找到新模式) ·
    [都偷偷学了 OT](../docs/GFLOWNET_THEORY_GUIDE_CN.md#所有-gflownet-都偷偷学了-ot) ·
    [\(P_T\) 与 \(P_F\) 是同一分布](../docs/GFLOWNET_THEORY_GUIDE_CN.md#p_tx-与-p_ftau-是同一个分布) ·
    [样本是精确 i.i.d.](../docs/GFLOWNET_THEORY_GUIDE_CN.md#训练后样本是精确-iid)
- [14. 一页式知识地图](../docs/GFLOWNET_THEORY_GUIDE_CN.md#14-一页式知识地图)
- [15. 本次调研方法与本地资料](../docs/GFLOWNET_THEORY_GUIDE_CN.md#15-本次调研方法与本地资料)
- [16. 最后应记住的五句话](../docs/GFLOWNET_THEORY_GUIDE_CN.md#16-最后应记住的五句话)

---

## 3. 术语表（中英对照）

### 3.1 核心术语（34 条）

使用说明：术语写法遵循全套文档「正文保留英文原文」的主流做法（R10 已确认执行良好），中文列为推荐译名，供首次出现处做一次性括号对照（R10 S8 的修复方式），不必每处都加。「首次出现」给出全套文档中最早出现与正式定义的位置，行号以 2026-08-14 版本为准。

| # | 术语（英文原文） | 中文 | 一句话定义 | 首次出现 |
|---|---|---|---|---|
| 1 | GFlowNet（Generative Flow Network） | 生成流网络 | 训练逐步构造对象的随机策略，使终止对象满足 \(P_T(x)=R(x)/Z\) 的生成模型框架。 | GUIDE L9 |
| 2 | flow（trajectory / state / edge flow） | 流（轨迹流/状态流/边流） | 图上的非负未归一化概率质量：\(F(\tau)\) 赋给完整轨迹并诱导状态流与边流；不是 normalizing flow 的可逆映射，也不是 ODE flow matching。 | GUIDE L21（直觉）；§2.2 L79–93（定义与辨析） |
| 3 | flow conservation | 流守恒 | 每个内部状态的总入流 = 状态流 = 总出流。 | GUIDE L22（R10 S8 建议补英文处）；§2.3 L97–103（公式） |
| 4 | reward matching | 奖励匹配 | 把终止边流固定为奖励 \(F(x\to s_f)=R(x)\)，使终止分布正比于 \(R\) 的边界约束及其精确解。 | GUIDE §2.3 L95–109；英文写法首现 L187 |
| 5 | Markovian flow | 马尔可夫流 | 轨迹概率可沿边分解为 \(\prod P_F\)（或 \(\prod P_B\)）的流，Foundations 的核心对象。 | GUIDE L129；CATALOG T02（L53） |
| 6 | forward policy | 前向策略 \(P_F\) | 由出边流归一化得到的逐步生成策略 \(P_F(s'\mid s)=F(s\to s')/F(s)\)。 | GUIDE L24（直觉）；L121–127（定义） |
| 7 | backward policy | 反向策略 \(P_B\)（CATALOG 作「后向策略」） | 由入边流归一化的回溯策略 \(P_B(s\mid s')=F(s\to s')/F(s')\)；固定 \(P_B\) 即选定把终止奖励向祖先分摊的规则。 | GUIDE L30（首现）；L121–127（定义）；§2.6 |
| 8 | terminating object 与 sink state | 终止对象 \(x\) 与形式汇点 \(s_f\) | \(x\in\mathcal X=\operatorname{Par}(s_f)\) 是可获奖励的对象状态；\(s_f\) 是全图唯一形式汇点；二者约定必须统一（GUIDE §12.4 检查项）。 | GUIDE L66–68；约定 L1092 |
| 9 | terminating distribution | 终止分布 \(P_T\) | 从 \(s_0\) 按 \(P_F\) 采样、最终终止于 \(x\) 的边缘概率 \(P_T(x)=\sum_{\tau\to x}P_F(\tau)\)。 | GUIDE L9–15（目标式）；L139–145（定理形式） |
| 10 | target distribution | 目标分布 \(P^\star\) | 理想采样目标 \(P^\star(x)=R(x)/Z\)；另存 \(\pi^\star\)、\(\pi_{\text{target}}\)、`p*(x)` 三种变体写法（见 §3.2 第 2 组）。 | CATALOG L40；GUIDE L453 |
| 11 | partition function | 配分函数（归一化常数）\(Z\) | \(Z=\sum_xR(x)\)，同时等于总流 \(F(s_0)=F(s_f)\)；TB 中作为参数 \(Z_\theta\) 学习。 | GUIDE L12–14（定义）；中文首现 L302（R10 S8 建议补英文处） |
| 12 | Flow Matching（FM） | 流匹配（GFlowNet 训练目标） | 状态级守恒残差目标：每个状态的入流与「出流 + 终止奖励」匹配；与连续生成模型的 flow matching 同名不同物。 | GUIDE §3.1 L191–224；辨析 L93 与 §5.4 |
| 13 | Detailed Balance（DB） | 细致平衡 | 边级约束 \(F(s)P_F(s'\mid s)=F(s')P_B(s\mid s')\)；名称借鉴 MCMC，但不要求生成链可逆。 | GUIDE §3.2 L226–258 |
| 14 | Trajectory Balance（TB） | 轨迹平衡 | 完整轨迹约束 \(Z\prod_tP_F=R(x)\prod_tP_B\)，把终端奖励直接连到整条轨迹。 | GUIDE §3.3 L260–304（恒等式 L271–274） |
| 15 | Subtrajectory Balance（SubTB(\(\lambda\))） | 子轨迹平衡 | 对任意子轨迹施加 balance 并按 \(\lambda^{n-m}\) 加权，在 DB（局部）与 TB（全局）之间连续插值。 | GUIDE §3.5 L332–356 |
| 16 | Guided Trajectory Balance（GTB） | 引导轨迹平衡 | 用人为指定的 guide \(q(\tau\mid x)\) 重新分配路径信用而不改变理想终止边缘。 | GUIDE §3.6 L358–376；作为 Shen 2023 干预 L442 |
| 17 | \(f\)-Trajectory Balance（\(f\)-TB） | \(f\)-轨迹平衡 | 把平方 log-ratio 代理 loss 推广到 \(f\)-divergence 家族，保留 off-policy 正确零点。 | GUIDE §3.7 L378–391；CATALOG T49（L115） |
| 18 | telescoping | 望远镜式消去 | 沿轨迹连乘 DB 等式、中间 \(F(s_t)\) 全部相消得到 TB 的推导技巧。 | GUIDE §3.4 L306–330 |
| 19 | credit assignment | 信用分配 | 终端奖励如何分摊到早期动作与中间状态的问题；各训练目标的信用范围不同（§3.8 对照表）。 | GUIDE L30（R10 S8 建议补英文处） |
| 20 | amortization / amortized sampling · marginalization | 摊销 / 摊销采样 · 摊销边缘化 | 用一次性训练成本换取之后低成本的重复采样或边缘量估计。 | GUIDE L45（R10 S8 建议补英文处）；§6.1 L683 |
| 21 | on-policy / off-policy | 在策略 / 离策略 | 训练数据是否来自当前前向策略；balance loss 在全支持 off-policy 下保留同一正确零点，但不保证 SGD 动态不受采样分布影响。 | GUIDE L383–384（首现）；§4.1 L417–427 |
| 22 | full support | 全支持 | 训练分布最终覆盖所有需要约束的状态/边/轨迹（正目标质量部分）；正确性定理的必要前提。 | GUIDE L411（中文）；L1021（英文） |
| 23 | mode coverage / mode collapse | 模式覆盖 / 模式坍缩 | 是否按目标质量覆盖各高奖励模式；探索失败或共享前缀偏置可致训练期漏掉模式。 | GUIDE L384（covering/seeking）；L578（collapse） |
| 24 | PRT / SSR | 优先回放训练 / 相对边流参数化 | Shen 2023 的两类干预：回放偏向高奖励样本纠正过采样；用相对边流改变共享结构的归纳偏置。 | GUIDE §4.2 L440–441；CATALOG T07（L58） |
| 25 | non-acyclic | 非无环 | 允许有向环的推广框架：流改释为 expected visit counts，需要吸收性与有限期望长度。 | GUIDE L32（R10 S8 建议补英文处）；§6.4 L713–725 |
| 26 | expected visit counts | 期望访问次数 | 非无环设定下状态/边流的含义：一条轨迹访问该状态/边次数的期望；现存三种英文变体（见 §3.2 第 1 组）。 | GUIDE L615、L719；OT分析 L49；CATALOG T19（L75） |
| 27 | absorbing / absorption | 吸收（吸收策略/吸收条件） | 策略以概率 1 在有限期望步数内到达 \(s_f\)；非无环理论与 OT 定理的关键条件。 | GUIDE L717；OT分析 L184；CATALOG T19（L75） |
| 28 | minimum-flow | 最小（内部总边）流 | 在满足守恒与两端边界的流中最小化 \(\sum_{e\in E^\circ}F(e)\)（= 期望轨迹长度）的内部流选择原则。 | GUIDE L32（中文）、L662（英文首现）；OT分析 L39–42、L86 |
| 29 | shortest path | 最短路 | 图上从 \(u\) 到 \(x\) 的最少边数路径长度，充当 OT 的 ground cost（即 \(d_G\)）。 | GUIDE L32（R10 S8 建议补英文处）；CATALOG O07（L139） |
| 30 | Kantorovich OT / Kantorovich coupling | Kantorovich 最优传输 / 耦合（运输计划） | 在边缘为 \(L,R\) 的联合分布集 \(\Gamma(L,R)\) 中最小化 \(\sum_{u,x}d_G(u,x)\Pi(u,x)\) 的线性规划；\(\Pi\) 即 transport plan。 | GUIDE L32（首现）、L629–647（定理 3.2）；OT分析 L57–73 |
| 31 | Kantorovich potential | Kantorovich 势（对偶变量） | 图 OT 对偶问题的势函数；互补松弛意味着最优流只走 tight edges，可作 critic 使用。 | OT分析 L239–246 |
| 32 | Schrödinger bridge | 薛定谔桥（熵正则路径 OT） | 固定两端边缘下最小化 \(\mathbb E_P[c(\tau)]+\varepsilon\operatorname{KL}(P\Vert P_0)\) 的路径分布问题。 | OT分析 §3.4 L152–172；CATALOG O04（L136） |
| 33 | unbalanced OT | 非平衡最优传输 | 允许源/目标总质量不相等（质量产生/消失）的 OT 推广；与未知 \(Z\) 并列为 GFN×OT 的关键开放问题。 | OT分析 L198、L232 |
| 34 | HyperGrid | 超网格（基准环境） | 可精确枚举终止分布的标准网格环境，用于区分「loss 下降」「平均奖励上升」与「分布真的正确」。 | GUIDE L189（首现）、L859；CATALOG §7（L324） |

### 3.2 术语与记号变体归一化（回应 R06）

以下变体在 2026-08-14 版本中仍然并存；「统一建议」沿用 R06 的裁定，供后续修订 `docs/` 时执行。本页自身按统一建议书写。

| # | 概念 | 现存写法（位置） | 统一建议 |
|---|---|---|---|
| 1 | 期望访问次数 | expected visit counts（GUIDE L719；OT分析 L93）；expected-visit-count（GUIDE L615）；expected visit count（GUIDE L964）；expected visit flow（GUIDE L725；CATALOG L75）；期望访问次数（OT分析 L49） | 统一主写法 **expected visit counts（期望访问次数）**：各文档首现处中英对照一次，其后固定英文 |
| 2 | 目标分布记号 | \(P^\star\)（CATALOG L40；GUIDE L453、L1072）；\(\pi^\star\)（GUIDE L549）；\(\pi_{\text{target}}\)（GUIDE L494、L501）；`p*(x)`（CATALOG L346 精读模板） | 统一为 **\(P^\star\)**；至少先在 GUIDE 内部消除三种写法混用 |
| 3 | 图最短路距离 | \(d_G(u,x)\)（OT分析 L57、L188）；\(d(u,x)\)（GUIDE L626、L633、L645） | 统一为 **\(d_G(u,x)\)**（显式标注依赖图结构，信息量更大） |
| 4 | \(P_B\) 的中文名 | 反向策略（GUIDE L30、§2.4、§2.6）；后向策略（CATALOG L53、L64、L86、L92） | 统一为 **反向策略（backward policy）**，以 GUIDE 为主文档口径 |
| 5 | 轨迹长度记法 | \(\lvert\tau\rvert\)：\(u\rightsquigarrow x\) 段内部边数（GUIDE L621；OT分析 L55）；\(n_\tau\)：论文按内部状态访问数计 | 保持 **\(\lvert\tau\rvert\)**，并保留两文档已有的「至多相差一个不影响最优解的加性常数」注记 |
| 6 | \(x\) 的称谓 | 终止对象（GUIDE L68；CATALOG L38）；终止状态（GUIDE L595 §5.5 设定） | 对 \(x\) 统一「**终止对象**」，\(s_f\) 称「**形式汇点**」；GUIDE §12.4（L1092）已将此列为工程检查项 |
| 7 | “flow”的三义 | GFlowNet 的流（图上未归一化质量）；normalizing flow（可逆映射）；flow matching（连续 ODE 速度场） | 不合并：三者不可互换定理或实现；跨语境引用时按 GUIDE §5.4 对照表（L580–588）标注语境 |

> 已修复、不再列入的历史变体：R06 曾指出的 CATALOG O08「最小化总流量」（现已改为「最小化内部总边流」）、O02/T54 论文标题挂错、OT分析开篇 \(P_T(x)=R(x)\) 漏写 \(Z\)（现已补归一化声明）等，在 2026-08-14 版本中均已解决。

---

## 4. 符号表

约定：与三份文档一致——集合用花体（\(\mathcal X\)）、策略用条件分布记号、流用 \(F(\cdot)\)。条件竖线本页一律写 \(\mid\)（仅为表格排版避让 `|`，数学含义与文档中的 \(P_F(s'|s)\) 完全相同）。

### 4.1 状态、轨迹与图（10 个）

| 符号 | 含义 | 定义与约定 | 主要出处 |
|---|---|---|---|
| \(s,\ s'\) | 状态与后继状态 | 有限有向图（DAG 或非无环图）的节点；\(s'\in\operatorname{Child}(s)\) | GUIDE §2.1 |
| \(s_0\) | 唯一源状态（source state） | 所有轨迹的起点；\(F(s_0)=Z\) | GUIDE L66 |
| \(s_f\) | 形式汇点（sink / final state） | 全图唯一形式终点；终止边界 \(F(x\to s_f)=R(x)\) | GUIDE L67、L105–109 |
| \(x\)、\(\mathcal X\) | 终止对象及其集合 | \(\mathcal X=\operatorname{Par}(s_f)\)；与 \(s_f\) 的约定须统一（§12.4） | GUIDE L68、L1092 |
| \(\operatorname{Par}/\operatorname{Child}\)、\(\operatorname{In}/\operatorname{Out}\) | 父/子节点集、入/出边集 | 守恒式与 LP 约束的求和范围 | GUIDE L100–102、L609–612 |
| \(\tau\) | 完整轨迹 | \(\tau=(s_0\to s_1\to\cdots\to x\to s_f)\) | GUIDE L69–73 |
| \(\lvert\tau\rvert\) | 轨迹长度 | OT 语境指 \(u\rightsquigarrow x\) 段的内部边数（论文记 \(n_\tau\)，差一个加性常数） | GUIDE L621；OT分析 L55 |
| \(\tau_{m:n}\) | 子轨迹 | \((s_m\to\cdots\to s_n)\)，SubTB 的监督单位 | GUIDE L334–345 |
| \(U\)、\(u\) | 首步状态集合及其元素 | 源点之后第一步可达的状态；源边缘 \(L\) 的支撑 | GUIDE L594；OT分析 L29 |
| \(E^\circ\) | 内部边集合 | 不含与 \(s_0\)、\(s_f\) 相邻的边；minimum-flow 目标的求和域 | GUIDE L600；OT分析 L55 |

### 4.2 流与守恒（5 个）

| 符号 | 含义 | 定义与约定 | 主要出处 |
|---|---|---|---|
| \(F(\tau)\) | 轨迹流 | 每条完整轨迹的非负未归一化质量 | GUIDE L79 |
| \(F(s)\) | 状态流 | \(\sum_{\tau\ni s}F(\tau)\)；非无环语境下 = 期望访问次数 | GUIDE L81–85、L719 |
| \(F(s\to s')\)、\(F(e)\) | 边流 | \(\sum_{\tau\ni(s\to s')}F(\tau)\)；LP 中简记 \(F(e)\) | GUIDE L81–85、L602–605 |
| \(Z\)、\(Z_\theta\)、\(Z(y)\) | 配分函数 / 总流 | \(Z=\sum_\tau F(\tau)=F(s_0)=F(s_f)=\sum_xR(x)\)；\(Z_\theta\) 为 TB 的学习参数；\(Z(y)\) 为条件版本 | GUIDE L12–14、L89–91、L271、L670 |
| \(\operatorname{div}F(s)\) | 散度 | 出流 − 入流；OT 约束为 \(\operatorname{div}F(s)=L(s)-R(s)\) | OT分析 L46；GUIDE L608–613 |

### 4.3 策略与分布（9 个）

| 符号 | 含义 | 定义与约定 | 主要出处 |
|---|---|---|---|
| \(P_F(s'\mid s)\) | 前向策略 | \(F(s\to s')/F(s)\) | GUIDE L123–127 |
| \(P_B(s\mid s')\) | 反向策略 | \(F(s\to s')/F(s')\)；DAG 上「固定 \(P_B\) + 终止流」唯一确定 Markovian flow | GUIDE L123–127、L177 |
| \(P_T(x)\) | 终止分布 | \(\sum_{\tau\to x}P_F(\tau)\)；精确解处 \(=R(x)/Z\)；条件版 \(P_T(x\mid y)\) | GUIDE L139–145、L668–672 |
| \(P^\star(x)\) | 目标分布 | \(R(x)/Z\)；统一写法（变体见 §3.2 第 2 组） | CATALOG L40；GUIDE L453 |
| \(R(x)\) | 奖励 / 未归一化目标密度 | 非负、只需知道到归一化常数；OT 语境兼作已归一化目标边缘 | GUIDE L17；OT分析 L30、L36 |
| \(E(x)\) | 能量 | \(E(x)=-\log R(x)\) | GUIDE L568 |
| \(Q(\tau)\)、\(P(\tau)\) | 前向 / 目标轨迹分布 | VI 视角：\(Q\) 是 proposal；\(P(\tau)\propto R(x)P_B(\tau\mid x)\) 是反向扩展的目标 | GUIDE L552、L946–947 |
| \(P_0(\tau)\) | 参考过程 | Schrödinger bridge 中 KL 项的基准动力学 / 先验策略 | OT分析 L160、L168 |
| \(y\) | 条件变量 | 条件 GFlowNet 的输入；奖励 \(R(x\mid y)\) 与 \(Z(y)\) 随之变化 | GUIDE L668–687 |

### 4.4 训练目标与超参数（7 个）

| 符号 | 含义 | 定义与约定 | 主要出处 |
|---|---|---|---|
| \(\mathcal L_{\mathrm{FM}}/\mathcal L_{\mathrm{DB}}/\mathcal L_{\mathrm{TB}}\) | 各 balance loss | 对应约束的平方 log-ratio 残差；SubTB 为子轨迹残差的 \(\lambda\) 加权和 | GUIDE L196–205、L239–249、L279–289 |
| \(\theta\)（下标） | 神经参数化 | \(F_\theta\)、\(P_{F,\theta}\)、\(P_{B,\theta}\)、\(Z_\theta\) 表示被学习的量 | GUIDE L200 起 |
| \(q\) | ① 内部流自由度参数；② GTB 的 guide 分布 | ① §2.5 中 \(q\in[0,1]\) 刻画同一 \(P_T\) 下的流族；② §3.6 中 \(q(\tau\mid x)\) 为轨迹上的归一化 guide | GUIDE L157、L360（双义，见 §4.7） |
| \(\lambda\) | ① SubTB 长度权重；② 流正则系数 | ① 以 \(\lambda^{n-m}\) 加权子轨迹 loss；② OT 实验中 regularized TB 的 flow-regularization 权重 | GUIDE L348；OT分析 L202（双义，见 §4.7） |
| \(\beta\) | 温度 | \(R^\beta\) 改变目标分布的熵与集中度；乘正常数则不改变目标 | GUIDE L841、L1032–1037、L1096 |
| \(c\) | TB 逐轨迹残差上界 | \(\mathcal L_{\mathrm{TB}}(\tau)\le c^2\Rightarrow\operatorname{TV}\le 1-e^{-2c}\)；局部界串联退化为 \(1-e^{-2Lc}\) | GUIDE L485–503（双义，见 §4.7） |
| \(\alpha\) | 探索—利用调节参数 | \(\alpha\)-GFN 从 Markov-chain 视角显式控制探索与利用强度 | GUIDE L782；CATALOG T46（L112） |

### 4.5 OT 专用记号（6 个）

| 符号 | 含义 | 定义与约定 | 主要出处 |
|---|---|---|---|
| \(L(u)\)、\(R(x)\)（边缘对） | 源 / 目标边缘（source / target marginals） | \(F(s_0\to u)=L(u)\)、\(F(x\to s_f)=R(x)\)，且 \(\sum_uL(u)=\sum_xR(x)=1\) | OT分析 L29–37；GUIDE L594–597 |
| \(d_G(u,x)\) | 图最短路代价（ground cost） | 有向图上未必对称，是合法 OT cost 但不必构成 Wasserstein metric；GUIDE 记 \(d(u,x)\)，统一建议 \(d_G\) | OT分析 L57、L188；GUIDE L626 |
| \(\Pi(u,x)\) | coupling / 运输计划 | 边缘为 \(L,R\) 的联合质量分配；最优流按策略采样的轨迹端点分布即诱导一个 coupling | GUIDE L631–647；OT分析 L63、L70 |
| \(\Gamma(L,R)\) | coupling 可行集 | 全部满足边缘约束的 \(\Pi\) 的集合 | GUIDE L632；OT分析 L62 |
| \(\mathbb E[\lvert\tau\rvert]\) | 期望轨迹长度 | \(=\sum_{e\in E^\circ}F(e)\)（expected visit counts 语义下），即 minimum-flow 的目标值 | GUIDE L615–621；OT分析 L49–55 |
| \(\varepsilon\) | 熵正则系数 | Schrödinger bridge 目标 \(\min\mathbb E_P[c(\tau)]+\varepsilon\operatorname{KL}(P\Vert P_0)\) 中的 KL 权重 | OT分析 L156–161 |

### 4.6 评估记号（2 个）

| 符号 | 含义 | 定义与约定 | 主要出处 |
|---|---|---|---|
| \(\operatorname{TV}\) | 总变差距离 | \(\operatorname{TV}(P_\theta,P^\star)=\frac12\sum_x\lvert P_\theta(x)-P^\star(x)\rvert\)；对象级分布误差的首选必报指标 | GUIDE L1071–1074 |
| \(L_1\) / JS / KL | 其他分布距离 | 对象级误差的补充指标（KL 需注意零概率处理） | GUIDE L476、L1078 |

### 4.7 双义符号提示（6 组）

以下符号在不同章节承担不同角色，均沿原文使用；跨节或跨文档引用时应注明语境。

| 符号 | 语境 A | 语境 B | 使用建议 |
|---|---|---|---|
| \(\lambda\) | SubTB(\(\lambda\)) 的子轨迹长度加权（GUIDE L348） | OT 神经实验中 regularized TB 的流正则系数（OT分析 L202；GUIDE L662） | 两义无关，跨节引用时写明「SubTB 权重」或「流正则系数」 |
| \(L\) | 源边缘 \(L(u)\)（OT分析 L29；GUIDE L594） | GUIDE §4.4 的轨迹长度上界（\(1-e^{-2Lc}\)，L498–503）；另 \(\mathcal L\) 为 loss 记号 | OT 语境默认边缘义；引用 §4.4 时写「长度上界 \(L\)」 |
| \(q\) | §2.5 内部流自由度标量 \(q\in[0,1]\)（GUIDE L157） | GTB 的 guide 分布 \(q(\tau\mid x)\)（GUIDE L360） | 前者是标量、后者是轨迹分布，符号形状可区分但引用时应点明 |
| \(c\) | TB 逐轨迹残差上界（GUIDE L485–497） | OT 语境的路径代价 \(c(\tau)\) 与条件 cost（OT分析 L158、L146） | 跨文档引用时注明「残差界」或「运输代价」 |
| \(\varepsilon\)/\(\epsilon\) | 小目标质量与 reward 下界（GUIDE L1041、L1096） | Schrödinger bridge 熵正则系数（OT分析 L160） | 均为局部记号，不跨节复用 |
| \(R\) | 奖励函数 \(R(x)\)（全套文档） | OT 语境中兼作已归一化目标边缘（\(\sum_xR(x)=1\)，OT分析 L36） | 同一对象在归一化假设下的双重角色；OT分析 L20 已显式声明该假设 |

---

## 5. 统计与维护说明

- **TOC**：92 个锚点条目（17 个章级 + 75 个小节级，覆盖 GUIDE 除文档主标题外的全部标题）。
- **术语表**：34 条核心术语 + 7 组变体归一化建议。
- **符号表**：39 个符号（10 + 5 + 9 + 7 + 6 + 2）+ 6 组双义提示。
- 行号基于 2026-08-14 版本（GUIDE 1230 行 / CATALOG 402 行 / OT分析 272 行）；`docs/` 更新后行号会漂移，锚点与章节号更稳定。若 GUIDE 标题文字改动，需按 GitHub slugger 规则重算对应 slug（含引号、全角标点与 en/em dash 的删除规则）。
- 本页锚点在 GitHub 渲染下有效；换用其他渲染管线时按 R10 M5/M6 的建议固定 slugger 与数学定界符插件。

