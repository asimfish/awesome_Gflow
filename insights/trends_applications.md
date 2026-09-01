# GFlowNet 2025–2026：应用落地与开源工具生态核实报告

> 核实日期：2026-09-01。所有 GitHub star / 最近提交时间、PyPI 版本、PR 合并状态均由当日 GitHub REST API 与 PyPI JSON API 直接读取，非二手转述。
> 定位：本文只做**增量与落地证据**核实，不重复盘点 `papers/current_papers.tsv`（206 篇，检索截止 2026-08-25）与 `surveys/` 下的方法学专题。
> 口径说明：
> - "最近提交"取仓库 `pushed_at`；提交频次用 commits API 按 `since` 拉取，单次上限 100 条，达到上限时报为"≥100"。
> - 判断某工具链"是否集成 GFlowNet"时，用 GitHub issue/PR 全文搜索作为代理指标，并做了正对照（`GRPO` in `huggingface/trl` = 1650 命中）以证明搜索本身有效。
> - 搜不到就写"未检索到"。

---

## 1. 结论摘要

1. **工业界唯一可验证的 GFlowNet 产品化动作来自 Recursion/Valence Labs，且只发生过一次**：2025-06-27 开源 `synflownet-boltz`（SynFlowNet + Boltz-2 生成式筛选），此后该仓库**零提交**。
2. **Recursion 自家的 GFlowNet 主库已进入停滞**：`recursionpharma/gflownet` 2025-09-01 至今仅 2 次提交，且均为 Dockerfile/tox 杂务。该公司 2025-06 裁员 20%，2026 年财报口径转向 foundation model，未再提 GFlowNet。
3. **LLM 后训练是 2025–2026 唯一真正"进主流工具链"的方向**：FlowRL 作为 recipe 被合并进 verl（PR #3924，2025-11-06 合并），目前存活于 `verl-project/verl-recipe` 子模块。这是 GFlowNet 目标函数第一次进入一个 2 万 star 级别的生产训练框架。
4. **但集成面极窄**：TRL 与 OpenRLHF 对 `GFlowNet` / `trajectory balance` 的 issue/PR 全文搜索命中均为 **0**。verl 是孤例。
5. **FlowRL 自身已被 GFlowRL 判定为在大规模下失效**：GFlowRL（arXiv 2607.13394，2026-07-15）称 FlowRL 的可学习配分函数网络在 MoE 配置（至 235B）上不收敛，改用 in-batch Monte Carlo 估计 logZ。
6. **工具链出现代际分化**：`torchgfn` 是唯一同时具备高频提交、频繁发版、环境持续扩张的库（v2.4.1 / 2026-04-05，2026-04 起 ≥100 次提交）；`gfnx`（JAX，2025-11 首发）提出了标准化评测意图但 PyPI 停在 0.0.1；社区索引 `Awesome-GFlowNets` 已停更近两年。
7. **教学资源基本停滞在 2023–2024**：Mila GFlowNet Workshop 只办过 2023-11-08~10 一届，未检索到 2025/2026 新办；2025–2026 的新增教学材料实际上以 `torchgfn` 仓库内的 notebook 形式交付，而非独立课程。

---

## 2. 药物发现的工业界落地

### 2.1 Recursion / Valence Labs：SynFlowNet-Boltz 是唯一有公开产品化叙事的动作

Recursion 官方新闻页 [Beyond Boltz-2: Toward More Powerful Drug Discovery Tools](https://www.recursion.com/news/beyond-boltz-2-toward-more-powerful-drug-discovery-tools)（2025-06-27）明确写了三件事：

- 开源 SynFlowNet-Boltz trainer，用于复现论文里的 generative screening 结果。
- SynFlowNet 由 Valence Labs（Recursion 的 AI 研究部门）在 2024 年提出，定位是解决生成模型产出"不可合成分子"的问题。
- 把 Boltz-2 的结合亲和力预测当作 reward，让 SynFlowNet 在 Enamine REAL 空间（宣称 760 亿+ 可合成化合物）里做生成式搜索。

论文侧口径（同一篇新闻转述）：TYK2 靶点上，从 SynFlowNet 生成的数千分子中挑 10 个做 ABFE 评估，**10/10 被预测为结合**；对照的固定库筛选中，Enamine HLL（460,160 化合物）top-10 里 8/10 结合，Kinase Library（64,960）10/10 结合，随机 10 个 0/10 结合。新闻本身给了保留意见——Boltz-2 在 TYK2 上表现本来就强，结果"可能偏乐观"。

这条证据的性质要说清楚：**它是 in-silico 到 in-silico 的验证链**（生成 → Boltz-2 打分 → ABFE 模拟），新闻里没有任何湿实验合成或活性测定数据。所以"落地"程度是"进入了公司公开宣传的工具链"，不是"产出了进入管线的分子"。

### 2.2 三个 SynFlowNet 仓库的维护状态：发布即冻结

| 仓库 | star | fork | open issues | 创建 | 最近提交 |
|---|---|---|---|---|---|
| [recursionpharma/synflownet-boltz](https://github.com/recursionpharma/synflownet-boltz) | 68 | 11 | 1 | 2025-06-27 | **2025-06-27** |
| [mirunacrt/synflownet](https://github.com/mirunacrt/synflownet) | 135 | 22 | 2 | 2024-05-03 | 2025-01-31 |
| [recursionpharma/gflownet](https://github.com/recursionpharma/gflownet) | 294 | 54 | 24 | 2022-02-24 | 2026-05-21 |

`synflownet-boltz` 的 `created_at` 与 `pushed_at` 是同一天同一小时内——**发布之后一次提交都没有**。这是个强信号：它是论文配套的一次性 artifact release，不是在维护的产品组件。

依赖链是三层：`synflownet-boltz` → `mirunacrt/synflownet` → `recursionpharma/gflownet`。底层的 `recursionpharma/gflownet` 从 2025-09-01 到核实日只有 **2 次提交**（均在 2026-05），最新一条是 `chore: edits from edit_tox_and_dockerfiles legion function (#147)`（2026-05-21），属于基础设施杂务而非功能开发。24 个未关闭 issue。

对比 Boltz 本体 [jwohlwend/boltz](https://github.com/jwohlwend/boltz)：4188 star，最近提交 2026-05-29。GFlowNet 那一侧是配角，且配角在掉队。

### 2.3 Recursion 的公司层面变化解释了这个停滞

- 2025-06-10 SEC 文件：裁员 20%（约 160 人），一次性支出约 1100 万美元，现金跑道延到 2027 Q4。来源：[Fierce Biotech](https://www.fiercebiotech.com/biotech/recursion-lays-20-staff-wake-pipeline-cutbacks)、[BioSpace](https://www.biospace.com/business/recursion-downsizes-by-20-to-boost-cash-position)、[GEN](https://www.genengnews.com/topics/artificial-intelligence/recursion-eliminating-20-of-workforce-citing-pipeline-pruning-and-capital-markets/)。
- 管线从 11 个砍到 6 个活跃项目，聚焦肿瘤与罕见病。
- 员工数：1002（2023）→ 774（2025）→ 603（2026 Q1），降幅 39.8%。来源：[Revelio Labs](https://www.reveliolabs.com/companies/recursion-pharmaceuticals/employees)（2026-03 数据）。
- 2026-08-05 Q2 财报新闻稿（[GlobeNewswire](https://www.globenewswire.com/news-release/2026/08/05/3339126/0/en/Recursion-Reports-Second-Quarter-Financial-Results-Genentech-Options-First-Neuroscience-Target-into-Early-Discovery-Program.html)）由新 CEO Najat Khan 署名，关键词是 "foundation models"、"end-to-end learning system"、"AI-native chemistry platform"，**通篇未出现 GFlowNet 或 SynFlowNet**。

把 2.2 的仓库数据和这里的公司数据放一起，结论是：Valence Labs 的 GFlowNet 路线在 2025 年中完成了一次开源交付后，没有在公司叙事里继续升级为产品。

### 2.4 其他药企/初创：未检索到直接使用证据

针对"其他药企或初创公开使用 GFlowNet"这一问题，检索结果里只有两类东西，都不构成落地证据：

- **学术方法论文**，作者是高校而非企业：AbFlowNet（[arXiv 2505.12358](https://arxiv.org/abs/2505.12358)，v1 2025-05-18，孟加拉工程技术大学 + UC Riverside + USC + Princeton）把每个扩散步当作 GFlowNet 状态，用 TB 目标注入结合能奖励；PG-AbD（[AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/34370)）用 PLM + Potts 模型的 Products-of-Experts 当 GFlowNet 的 reward，抗体多样性在 RabDab 上 +13.5%、SabDab 上 +31.1%。
- **有生成式平台但未声明用 GFlowNet 的公司**：例如 OpenProtein.AI 与勃林格殷格翰的合作扩展（[公告](https://www.openprotein.ai/publications/strategic-partnership-with-boehringer-ingelheim/)，2026-03-31），其技术描述是 PoET 基础模型，**没有提 GFlowNet**。把这类公告算作 GFlowNet 落地是过度归因。

AbFlowNet 的官方实现 [Patchwork53/abflownet](https://github.com/Patchwork53/abflownet) 只有 3 star（最近提交 2026-08-29），说明即使在学术侧也没有形成采用。

一个值得单独记的非生物领域数据点：[nshen7/alpha-gfn](https://github.com/nshen7/alpha-gfn)（121 star，25 fork，最近提交 2026-01-23）用 GFlowNet 做量化投资的公式化 alpha 因子挖掘，README 明确写"由于行业 NDA，本仓库只用于演示"。这是唯一检索到的、明确暗示存在闭源工业部署的 GFlowNet 应用，但**具体部署内容无法核实**。关联论文 AlphaSAGE（Chen et al., 2025）。

---

## 3. 材料方向：Crystal-GFN 的后续

### 3.1 SHAFT 进入期刊，且首次做了"完整晶体结构"生成

Crystal-GFN（[arXiv 2310.04925](https://arxiv.org/abs/2310.04925)）只生成空间群、组成、晶格参数，**不生成原子坐标**。其后续 SHAFT 于 2026 年发表在 RSC 的 *Digital Discovery*：[Efficient symmetry-aware materials generation via hierarchical generative flow networks](https://pubs.rsc.org/en/content/articlehtml/2026/dd/d4dd00392f)（DOI [10.1039/D4DD00392F](https://doi.org/10.1039/d4dd00392f)）。

论文自述的两步贡献值得记，因为它把 ablation 设计写得很干净：

1. 先做一个 **flat（非层级）GFlowNet**，在单一策略里生成空间群 + 晶格 + 原子坐标——论文称这本身就是 GFlowNet 生成完整原子结构的首次实现。
2. 再做 SHAFT，把材料空间分解为"空间群 → 晶格参数 → 原子"的层级子空间。flat 基线的存在就是为了**隔离出层级分解本身的收益**。

对比对象是 CDVAE 与 DiffCSP，指标为 validity / stability / diversity。同一路线的早期版本 CHGFlowNet 见 [OpenReview dJuDv4MKLE](https://openreview.net/pdf?id=dJuDv4MKLE)。

本仓库 TSV 已收录该条（N069）。

### 3.2 Mila 的材料代码库仍在活跃，但要小心认错仓库

Crystal-GFN 的实际代码家在 [alexhernandezgarcia/gflownet](https://github.com/alexhernandezgarcia/gflownet)：344 star，32 fork，84 open issues，最近提交 2026-08-29。2025-09-01 以来提交分布为 2025-12: 2、2026-04: 16、2026-05: 19、2026-06: 15、2026-07: 43、2026-08: 5（达 100 条 API 上限，故为 ≥100），最新一条是 2026-08-13 合并 PR #452（`math-importance-sampling`）。**这是除 torchgfn 外唯一活跃的通用 GFlowNet 库。**

陷阱：搜索结果常把 [milaforscience/gflownet](https://github.com/milaforscience/gflownet) 当作官方库。该仓库 **0 star、0 fork，创建与最后提交都是 2024-06-12**，是个空镜像。引用时应指向 `alexhernandezgarcia/gflownet`。

### 3.3 催化剂方向

Catalyst GFlowNet（[arXiv 2510.02142](https://arxiv.org/abs/2510.02142)，2025-10）做析氢反应（HER）电催化剂设计，本仓库 TSV 已收录（N070）。分子晶体的分布式生成框架见 TSV A16（[arXiv 2607.05266](https://arxiv.org/abs/2607.05266)）。这两条属于"论文层面持续产出"，未检索到工业部署证据。

---

## 4. LLM 推理与后训练：唯一进了主流工具链的方向

### 4.1 FlowRL 进 verl 的完整时间线（逐条 API 核实）

| 事件 | 编号 | 日期 | 状态 |
|---|---|---|---|
| FlowRL arXiv v1 | [2509.15207](https://arxiv.org/abs/2509.15207) | 2025-09-18 | v3 更新于 2025-11-04 |
| 官方实现开源 | [Xuekai-Zhu/FlowRL](https://github.com/Xuekai-Zhu/FlowRL) | 2025-09-17 创建 | 182 star / 19 fork / 6 open issues，最近提交 2025-11-24 |
| verl recipe PR 提交 | [#3924](https://github.com/verl-project/verl/pull/3924) | 2025-10-27 | **已合并 2025-11-06** |
| 修复 PR | [#4397](https://github.com/verl-project/verl/pull/4397) | — | **已合并 2025-12-04**（"FlowRL actor to pure implementation"） |
| recipe 迁出主仓 | verl-recipe #7 | 2026-01-05 | `feat: migrate recipes from verl` |
| verl 主仓 examples 重构 | [#6126](https://github.com/verl-project/verl/pull/6126) | — | 已合并 2026-04-30 |
| recipe 版本钉固 | verl-recipe #86 | 2026-04-20 | `[refactor] pin verl versions of each recipe` |

**当前位置需要特别说明，否则会误判为"已被删除"**：`verl-project/verl` 主仓的 git tree（main 分支，1463 条路径，未截断）里搜 `flowrl` 命中 **0**，因为 `recipe` 现在是 gitmodule（mode `160000`），指向 [verl-project/verl-recipe](https://github.com/verl-project/verl-recipe)。在该子模块的 tree（751 条路径）里，`flowrl/` 命中 19 条路径，包含 `flowrl_actor.py`、`flowrl_fsdp_worker.py`、`flowrl_ray_trainer.py`、`main_flowrl.py`、`config/flowrl_trainer.yaml`、`run_flowrl_qwen2.5_7b.sh`。

宿主体量：`verl-project/verl` 23,225 star / 4,469 fork / 1,157 open issues，最近提交 2026-08-31。`verl-recipe` 328 star，创建于 2025-11-25，最近提交 2026-08-31，顶层 41 条目录项中除去 9 个非 recipe 文件后为 **32 个 recipe 目录**（flowrl 与 dapo、prime、spin、gvpo、retool、fapo 等并列）。

维护强度的诚实评估：`flowrl/` 目录自迁入后只有 **2 次提交**（2026-01-05 迁移、2026-04-20 钉版本），都是仓库级批量操作，没有针对 FlowRL 的功能迭代。它的 `REQUIRED_VERL.txt` 给了两条路径——Option A 钉 `verl==0.4.0`（论文复现），Option B 钉 rolling main 的具体 commit。这种"必须钉版本才能跑"的形态说明它与主干的耦合是脆的。

### 4.2 GFlowRL：FlowRL 在规模上被自己人否掉

[GFlowRL: Scaling Distribution-Matching RL to Large Language Models](https://arxiv.org/abs/2607.13394)（v1 2026-07-15）的核心主张直接针对 FlowRL 的工程弱点：

- **删掉可学习配分函数网络**。FlowRL 用一个随机初始化的 3 层 MLP 吃 LLM hidden states 输出标量 logZ（见 [ICLR 2026 版 FlowRL](https://proceedings.iclr.cc/paper_files/paper/2026/file/f657eb3343d4bb85a1d77821d1fbe4b8-Paper-Conference.pdf) §F）。GFlowRL 改用 in-batch Monte Carlo 估计：对 GRPO 本来就要采的 G 个 rollout，取 `logẐ_t(x) = (1/G) Σ [β r(x,y⁽ⁱ⁾) + log π_ref(y⁽ⁱ⁾|x) − log π_old(y⁽ⁱ⁾|x)]`，并施加 stop-gradient 当基线。零额外参数。
- **两个稳定器**：rollout/trainer 分布漂移的重要性采样校正 + 非对称 flow-gap 裁剪。
- **结论性对比**：14B dense 达 Codeforces 2048 Elo（距 o3-mini 25 Elo）；MoE 30B-A3B 达 1999 Elo；MoE 235B-A22B 可稳定训练，而**FlowRL 在所有被测 MoE 配置上不收敛**。论文还报了梯度尺度差异（FlowRL 均值 ~10¹⁴ vs GFlowRL ~0.07）。
- 论文自称是第一个扩到该规模的 GFlowNet 式 RL 方法，同样基于 veRL 实现。

方法论上，这个 in-batch 估计与 log-partition variance loss 是同源思路（都利用"TB 最优时每条轨迹给出相同隐式 target"），差别在于 GFlowRL 取组内均值当 stop-gradient 基线，而非最小化 target 的方差。

本仓库 TSV 已收录（A33），但当时的备注是"论文非常新，需要重点检查"——现在可以填实了。**未检索到 GFlowRL 被合并进 verl 或任何主流框架的 PR。**

### 4.3 TBA：学术认可高，工程采用低

[Trajectory Balance with Asynchrony](https://arxiv.org/abs/2503.18929)（v1 2025-03-24，v2 2025-12-03，NeurIPS 2025）把 off-policy TB 目标塞进异步分布式 RL：多个 searcher 节点持续生成轨迹进中央 replay buffer，单个 trainer 节点按 reward 或 recency 采样更新。报告的加速：相对 VinePPO 近 50×，相对速度优化过的异步 DPO 基线约 1.5×–5×；红队任务上相对非分布式同步 GFlowNet 基线约 7×。

官方实现 [bbartoldson/TBA](https://github.com/bbartoldson/TBA)：**32 star**，4 fork，0 open issue，最近提交 2025-11-05。项目页 [tba-llm.github.io](https://tba-llm.github.io/)。

NeurIPS 2025 主会接收 + 32 star，这个反差是 GFlowNet-for-LLM 方向的典型特征：论文层面被认可，代码层面无人采用。

### 4.4 主流 RLHF 工具链的集成情况：verl 是孤例

| 框架 | star | 最近提交 | `GFlowNet` issue/PR 命中 | `trajectory balance` 命中 |
|---|---|---|---|---|
| [verl-project/verl](https://github.com/verl-project/verl) | 23,225 | 2026-08-31 | 5（含 FlowRL PR 链） | — |
| [huggingface/trl](https://github.com/huggingface/trl) | 19,186 | 2026-08-31 | **0** | **0** |
| [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | 9,961 | 2026-08-13 | **0** | — |

正对照：同一搜索接口下 `GRPO` 在 `huggingface/trl` 命中 1650，证明 0 命中不是接口失效。

**结论：截至 2026-09-01，GFlowNet 系目标函数在开源 LLM 后训练生态里的采用面 = verl 的一个 recipe。** TRL 与 OpenRLHF 连讨论都没有。

---

## 5. 开源工具链现状（全部为 2026-09-01 实测）

### 5.1 总表

| 仓库 | 定位 | star | fork | open issues | 创建 | 最近提交 | 判断 |
|---|---|---|---|---|---|---|---|
| [GFNOrg/torchgfn](https://github.com/GFNOrg/torchgfn) | PyTorch 通用库 | 315 | 57 | 52 | 2022-06-29 | 2026-08-22 | **活跃，事实标准** |
| [alexhernandezgarcia/gflownet](https://github.com/alexhernandezgarcia/gflownet) | Mila 科学应用库（Crystal-GFN 家） | 344 | 32 | 84 | 2022-10-17 | 2026-08-29 | **活跃** |
| [d-tiapkin/gfnx](https://github.com/d-tiapkin/gfnx) | JAX 环境+基准 | 92 | 2 | 1 | 2025-11-16 | 2026-05-10 | 新，势头未定 |
| [recursionpharma/gflownet](https://github.com/recursionpharma/gflownet) | 图/分子专用库 | 294 | 54 | 24 | 2022-02-24 | 2026-05-21 | **停滞（近一年仅 2 次杂务提交）** |
| [GFNOrg/gfn-lm-tuning](https://github.com/GFNOrg/gfn-lm-tuning) | LLM 摊销推理微调 | 191 | 22 | 7 | 2023-10-05 | 2026-08-30 | 有维护 |
| [GFNOrg/gfn-diffusion](https://github.com/GFNOrg/gfn-diffusion) | 扩散采样器 | 38 | 7 | 3 | 2024-02-05 | 2026-08-30 | 小而有维护，**无 license** |
| [mirunacrt/synflownet](https://github.com/mirunacrt/synflownet) | 合成路径动作空间 | 135 | 22 | 2 | 2024-05-03 | 2025-01-31 | 冻结 |
| [recursionpharma/synflownet-boltz](https://github.com/recursionpharma/synflownet-boltz) | SynFlowNet+Boltz-2 | 68 | 11 | 1 | 2025-06-27 | 2025-06-27 | **发布即冻结** |
| [Xuekai-Zhu/FlowRL](https://github.com/Xuekai-Zhu/FlowRL) | FlowRL 官方实现 | 182 | 19 | 6 | 2025-09-17 | 2025-11-24 | 冻结（功能已迁 verl-recipe） |
| [bbartoldson/TBA](https://github.com/bbartoldson/TBA) | TBA 官方实现 | 32 | 4 | 0 | 2025-04-09 | 2025-11-05 | 冻结 |
| [zdhNarsil/Awesome-GFlowNets](https://github.com/zdhNarsil/Awesome-GFlowNets) | 社区论文索引 | 504 | 36 | 3 | 2022-10-05 | **2024-10-01** | **停更近 2 年** |
| [GFNOrg/gflownet](https://github.com/GFNOrg/gflownet) | 2021 原始实现 | 686 | 80 | 9 | 2021-06-07 | **2023-02-28** | 历史归档 |
| [zdhNarsil/GFlowNet-CombOpt](https://github.com/zdhNarsil/GFlowNet-CombOpt) | 图组合优化 | 69 | 13 | 3 | 2023-05-17 | 2023-05-30 | 冻结 |
| [nshen7/alpha-gfn](https://github.com/nshen7/alpha-gfn) | 量化 alpha 因子挖掘（演示版） | 121 | 25 | — | 2024-05-08 | 2026-01-23 | 演示用，工业版闭源 |
| [milaforscience/gflownet](https://github.com/milaforscience/gflownet) | — | 0 | 0 | 0 | 2024-06-12 | 2024-06-12 | **空镜像，勿引用** |

### 5.2 torchgfn：唯一具备"库"品相的项目

- PyPI `torchgfn` 最新 **2.4.1**，上传 2026-04-05；累计 18 个版本，最早 0.1.0（2023-05-23）。要求 Python ≥3.10。
- 近一年发版节奏：v2.1.1(2025-08-08) → v2.2.0(2025-09-03) → v2.2.1(2025-09-04) → v2.2.2(2025-09-06) → v2.3.0(2025-10-24) → v2.3.1(2025-10-30) → v2.4.0(2026-03-20) → v2.4.1(2026-04-05)。
- 提交密度：2026-04 起 ≥100 次提交（2026-04: 71、2026-05: 3、2026-07: 3、2026-08: 23），最新为 2026-08-20 合并 PR #532。
- 文档站 [torchgfn.readthedocs.io](https://torchgfn.readthedocs.io/en/latest/)。**license 字段为 `NOASSERTION`**，做商用集成前需人工确认。

环境覆盖（`src/gfn/gym/`）已远超 HyperGrid：`hypergrid`、`box` / `box_cartesian`、`bitSequence` / `bitSequenceNonAutoregressive`、`discrete_ebm`、`graph_building`、`bayesian_structure`（含完整 helpers：priors / scores / jsd / sampling）、`diffusion_sampling`、`perfect_tree`、`set_addition`、**`chip_design`**（带 `plc_client.py`，即芯片布局那套接口）。

教学与示例（`tutorials/`）：notebook 有 `getting_started`、`intro_discrete`、`intro_continuous` / `intro_continuous_beta`、`intro_graphs`、`seven_segments`、`policy_gradient_gflownets`、`trust_pcl_equivalence`、`hypergrid_difficulty_comparison`、`hypergrid_rewards`；脚本有 `train_diffusion_rtb.py`、`train_diffusion_sampler.py`、`train_bayesian_structure.py`、`train_chip_design.py` / `_medium.py`、`train_hypergrid_ppo.py`、`train_hypergrid_local_search.py`、`train_hypergrid_gafn.py`、`train_with_compile.py`，以及一个完整的 **`mRNA_design/`** 应用示例（含 CAI / MFE 计算器）。

`trust_pcl_equivalence.ipynb` 与 `policy_gradient_gflownets.ipynb` 的存在有额外意义：库本身在教 GFlowNet 与 RL 的等价关系，这条线呼应 `surveys/GFLOWNET_RL_EQUIVALENCE_CN.md`。

### 5.3 gfnx：2025-11 出现的 JAX 新库，意图是标准化

论文 [gfnx: Fast and Scalable Library for Generative Flow Networks in JAX](https://arxiv.org/abs/2511.16592)（v1 2025-11-20；作者 Tiapkin, Agarkov, Morozov, Maksimov, Tsyganov, Gritsaev, Samsonov）。

- 设计：JIT-able 环境 + reward 模块 + 指标全部 JAX 实现；每个环境配一个 CleanRL 风格单文件 baseline，用 Equinox 建网络，环境与训练循环端到端 JIT（借 purejaxrl 的做法）。
- 声明的加速：CPU 序列生成环境最高 **55×**、GPU 贝叶斯网络结构学习最高 **80×**（对比 torchgfn 与作者原实现）。
- 覆盖环境：合成 hypergrid、多种编辑机制的序列生成、分子生成 reward、系统发生树构建、贝叶斯结构学习、Ising 能量采样（论文称 8 个环境）。
- 论文明确写了目标是 "standardize empirical evaluation"。

工程成熟度的保留意见：**PyPI 只有 0.0.1 一个版本，上传于 2025-11-16，此后未发新版**；仓库 92 star / 2 fork / 1 open issue，最近提交 2026-05-10。文档站 [gfnx.readthedocs.io](https://gfnx.readthedocs.io)。9 个月没发版、只有 2 个 fork，说明它目前是"作者自用 + 论文配套"，还没有形成社区。

**该论文未收录在本仓库 TSV 中**（grep `2511.16592` = 0 命中），而 `surveys/GFLOWNET_EVALUATION_ECOSYSTEM_CN.md` 已在正文里大量引用 gfnx 的指标设计。这是个应当补齐的引用缺口。

---

## 6. 基准与评测

### 6.1 标准化评测套件：有人在做，但还没成事实标准

现状是**两个候选、零共识**：

- **gfnx**（§5.3）是唯一明确以"标准化"为目标的项目，把 TV、logZ 偏差、mode 指标等做进 `metrics/` 模块，并复现 Shen et al. 2023 的 QM9 reward-分布 TV 设定。但 PyPI 停在 0.0.1、2 个 fork，没有第三方采用证据。
- **torchgfn** 事实上承担了基准角色（HyperGrid 难度分级、bit sequence、贝叶斯结构学习 + DAG-GFlowNet 复现、diffusion sampler、chip design），但它把自己定位为库而非评测套件，不提供跨方法排行榜。

未检索到 GFlowNet 领域出现类似 `lm-evaluation-harness` 的独立评测 harness，也未检索到公开 leaderboard。

### 6.2 常用基准的分布

`surveys/GFLOWNET_EVALUATION_ECOSYSTEM_CN.md` 已详细写过 HyperGrid / QM9 / sEH fragment / 组合与 diffusion sampler 基准的 ground-truth 分档，此处只补 2025–2026 的增量：

- **TF Bind 8**：本次检索未在 gfnx 的 8 个环境清单或 torchgfn 的 `gym/` 里找到对应实现。它仍是散见于各论文自带代码的基准，**未进任何主流库**。
- **AMP（抗菌肽）**：gfnx 收录（论文环境清单第 5 项）。
- **chip design**：torchgfn 新增，带 Google PLC 客户端接口。这是把"真实系统级基准"引入 GFlowNet 库的第一例。
- **mRNA design**：torchgfn 以完整示例形式提供（CAI + MFE 双目标）。

### 6.3 低方差 logZ 估计：2026 年的实际进展

这一条是本次检索里方法论层面最实质的增量，四条路线：

1. **GFlowRL 的 in-batch MC 估计**（[2607.13394](https://arxiv.org/abs/2607.13394)，2026-07-15）：如 §4.2，直接用 rollout group 的 TB target 均值当 stop-gradient 基线，彻底去掉 logZ 网络。这是**从"学 logZ"转向"估 logZ"**的路线转折，且是在 LLM 这个最大规模场景上验证的。它自承与 log-partition variance loss 同源，但报告后者在其设定下"几乎不优于 backbone"。
2. **Sub-EB / 评估平衡**（ICLR 2026 poster，[iclr.cc/virtual/2026/poster/10007783](https://www.iclr.cc/virtual/2026/poster/10007783)，Niu / Wu / Qian）：证明 flow balance 本身诱导出一个合规的 policy evaluator，用子轨迹上的 evaluation balance 目标学这个 evaluator，从而打通 value-based 与 policy-based 两种训练范式，并支持参数化 backward policy 与离线数据。代码 [niupuhua1234/Sub-EB](https://github.com/niupuhua1234/Sub-EB)（**1 star**，创建 2026-02-26）。
3. **RapTB**（[arXiv 2603.00454](https://arxiv.org/abs/2603.00454)，v1 2026-02-28，v3 2026-07-20）：指出 SubTB 在任意起点窗口上施加约束会产生冲突的边界条件，导致**终止漂移**（`log p_term(τ)` 退化到极负值，在停止动作是决策变量时严重打击命中率）。RapTB 把稠密监督限制在 rooted prefix 上以保持与全局 Z 一致；论文还用 RootSubTBLogZ（限 rooted 窗口 + 重新引入可学 `Z_θ`）做诊断实验，确认终止漂移是主要失效模式。
4. **DTB / ACE**（[arXiv 2602.17827](https://arxiv.org/abs/2602.17827)，v1 2026-02-19）：训练一个 exploration GFlowNet 主动避开 canonical 模型已知的区域，canonical loss 为两个分布混合下的 TB 期望（带 stop-gradient 权重）。评测同时报 top-K 平均奖励、logZ 收敛速率与 TV。

这四条里只有 GFlowRL 有大规模验证，另外三条的代码采用度接近零（Sub-EB 1 star）。

另有 [Information-Geometric Forward Policy Training in GFlowNets](https://arxiv.org/abs/2608.03967)（2026-08）从 Fisher-Rao 度量与自然梯度角度重构 forward policy 训练，属理论增量，未检索到实现。

**TSV 收录状态**：RapTB（2603.00454）、DTB（2602.17827）、AbFlowNet（2505.12358）、gfnx（2511.16592）、SynFlowNet arXiv 版（2405.01155）均为 0 命中。

---

## 7. 教学资源：2025–2026 基本停滞

这是本次调研中最清晰的负面结论。

**未检索到的**：
- 2025 或 2026 年新办的 GFlowNet 专题 workshop。[gflownet.org](https://www.gflownet.org/) 上唯一的活动仍是 **2023-11-08~10 的 Mila GFlowNet Workshop**（首届，约 100 名研究者参加，其中一半来自 ML 之外的生物/物理/化学领域，见 [Mila 新闻](https://mila.quebec/en/news/gflownets-ai-at-the-service-of-scientific-discovery)）。站点 [Resources 页](https://www.gflownet.org/resources)列的仍是 2022–2023 的 workshop 论文。
- 2025–2026 新开的大学课程或 MOOC 模块。
- 2025–2026 新录制的系统性讲座视频系列。

**仍在被引用的旧资源**（均为 2022–2024）：
- [The GFlowNet Tutorial](https://milayb.notion.site/The-GFlowNet-Tutorial-95434ef0e2d94c24aab90e69b30be9b3)（Notion，高层介绍）
- Emmanuel Bengio 的 [Colab 实践 notebook](https://colab.research.google.com/drive/1fUMwgu2OhYpQagpzU5mhe9_Esib3Q2VR)
- [zdhNarsil/Awesome-GFlowNets](https://github.com/zdhNarsil/Awesome-GFlowNets)：504 star，但**最近提交 2024-10-01**，已停更近 2 年。任何依赖它做文献扫描的流程都会漏掉 2025–2026 的全部工作。

**2025–2026 实际新增的教学材料，形态变了**：不再是独立课程或博客，而是**库内 notebook**。torchgfn 的 v2.2–v2.4 系列 release notes 明确列出新增教学内容：`getting_started` notebook（HyperGrid FM 示例）、`seven_segments` 图 GFlowNet 教程（由 josephdviviano 与 younik 贡献，PR #498）、更难的 hypergrid 任务（PR #495）、`policy_gradient_gflownets` 与 `trust_pcl_equivalence` 等等价性 notebook。gfnx 则为每个环境提供单文件 baseline，本身即教学材料。

判断：入门路径已从"读 Notion 教程 + 跑 Colab"迁移到"装 torchgfn 跑 notebook"。这对工程实践是改善，但**丢掉了概念性的、面向非 ML 背景科学家的教学层**——而 2023 年那届 workshop 一半参与者恰恰来自这个群体。

---

## 8. 相对本仓库现状的增量清单

`papers/current_papers.tsv` 共 206 行（不含表头）。venue 分布头部为：预印本 2026（11）、NeurIPS 2024（10）、ICLR 2025（10）、ICML 2026（8）、ICML 2025（8）、ICLR 2024（8）。

**已覆盖、本报告只做落地状态补充的**：
- N019 FlowRL（2509.15207）→ 补：verl PR #3924 已于 2025-11-06 合并，现居 verl-recipe 子模块，flowrl 目录仅 2 次提交。
- N018 TBA（2503.18929）→ 补：官方实现仅 32 star。
- A33 GFlowRL（2607.13394）→ 补：原备注"需重点检查"，现可确认其核心是用 in-batch MC 替换 logZ 网络，并声明 FlowRL 在 MoE 上发散。
- A06 Crystal-GFN / N069 SHAFT → 补：SHAFT 已正式发表于 *Digital Discovery*（RSC，2026），DOI 10.1039/D4DD00392F；代码家在 `alexhernandezgarcia/gflownet` 而非 `milaforscience/gflownet`。
- A11 SynFlowNet → 补：arXiv 编号 2405.01155（v3 2025-04-09）；Recursion 于 2025-06-27 开源 synflownet-boltz 且此后零提交。

**未收录、建议补入的**：

| 建议 ID | 条目 | 链接 | 日期 | 理由 |
|---|---|---|---|---|
| — | gfnx（库论文） | [2511.16592](https://arxiv.org/abs/2511.16592) | 2025-11-20 | evaluation 专题已大量引用其指标设计，但索引里没有本体 |
| — | RapTB | [2603.00454](https://arxiv.org/abs/2603.00454) | 2026-02-28 | 指出 SubTB 的终止漂移失效模式，与 stability 专题直接相关 |
| — | DTB / ACE | [2602.17827](https://arxiv.org/abs/2602.17827) | 2026-02-19 | exploration lineage 专题的 2026 增量 |
| — | Sub-EB | [ICLR 2026 poster](https://www.iclr.cc/virtual/2026/poster/10007783) | 2026-04-23 | 打通 value-based / policy-based，与 RL 等价性专题相关 |
| — | AbFlowNet | [2505.12358](https://arxiv.org/abs/2505.12358) | 2025-05-18 | 扩散步当 GFlowNet 状态的抗体设计，分子应用全景缺此条 |
| — | PG-AbD | [AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/34370) | 2025 | PoE reward 设计 |
| — | 信息几何 forward policy | [2608.03967](https://arxiv.org/abs/2608.03967) | 2026-08 | Fisher-Rao / 自然梯度视角 |

另建议在 `surveys/GFLOWNET_RESEARCH_RESOURCE_CATALOG_CN.md` 的代码资源节里加三条勘误：
1. `milaforscience/gflownet` 是 0 star 空镜像，正确地址是 `alexhernandezgarcia/gflownet`。
2. `Awesome-GFlowNets` 停更于 2024-10-01，不可作为 2025–2026 的文献来源。
3. `GFNOrg/gflownet`（686 star）停更于 2023-02-28，是历史归档，不是活跃实现。

---

## 9. 判断与未解问题

**三条主判断**：

1. **"落地"的定义要收紧。** 唯一可核实的工业动作（SynFlowNet-Boltz）是一次性开源发布 + 全 in-silico 验证链，发布方随后大幅裁员并把公司叙事转向 foundation model，配套仓库零提交。把这条算作"GFlowNet 已在药物发现落地"是超出证据的。准确表述是：**GFlowNet 在药物发现中完成了一次可复现的工具交付，但没有可核实的产品化延续。**

2. **重心已从分子转到 LLM 后训练，但迁移只完成了一半。** FlowRL 进 verl 是这个方向唯一的真实工程战果；但 TRL 与 OpenRLHF 零命中，FlowRL 自己在 MoE 规模上被 GFlowRL 证明发散，而 GFlowRL 尚未进任何框架。所谓"GFlowNet 进入 RLHF 主流"目前只有一个孤例支撑。

3. **工具链正在两极化。** 一端是 torchgfn（18 个 PyPI 版本、近半年 ≥100 提交、环境从 HyperGrid 扩到 chip design 与 mRNA 设计）和 alexhernandezgarcia/gflownet（≥100 提交）；另一端是一长串"论文发完就冻结"的官方实现（synflownet-boltz 0 次后续提交、TBA 32 star、Sub-EB 1 star、Awesome 索引停更 2 年）。中间层——那种被多个团队共同依赖的、有第三方 fork 与 issue 流量的库——除 torchgfn 外基本不存在。

**未解问题**（本次检索无法定论，需后续跟踪）：

- GFlowRL 是否会被合入 verl-recipe，或被 TRL/OpenRLHF 接受？这是判断"分布匹配式 RL 能否成为标准选项"的关键观测点。
- `recursionpharma/gflownet` 的停滞是暂时的资源重分配，还是路线放弃？Valence Labs 在 2026 年下半年若无新 GFlowNet 输出，可视为后者。
- gfnx 能否从"作者自用"变成社区基准？观测指标：PyPI 是否发 0.0.2+、fork 数是否离开个位数、是否有第三方论文用它报结果。
- TF Bind 8 这类经典基准长期不进主流库，是否意味着社区已实质放弃它？
- 教学层的空缺（面向非 ML 背景科学家）由谁填？2023 年那届 workshop 证明存在这个受众。

**方法论备注**：本报告所有 GitHub 数字均为 2026-09-01 单日快照。star 数与 issue 数会漂移，`pushed_at` 与提交分布是更可靠的活跃度指标，复核时应优先看后者。verl 相关结论必须同时查主仓与 `verl-recipe` 子模块，只查主仓会得出"FlowRL 已被移除"的错误结论。
