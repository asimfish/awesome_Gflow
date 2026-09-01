# GFN 与 VI / MCMC / SMC：推断方法论的统一视图

> 本文扩展理论指南 §5.2–5.3。
> 来源：GFlowNet 调研 2026-08 审查扩充（E09）。核心索引见 [README](README.md)。

---

> 衔接 [理论指南](../docs/GFLOWNET_THEORY_GUIDE_CN.md) §5.2（变分推断）、§5.3（MCMC）。沿用其记号：前向策略 \(P_F(s'\mid s)\)、反向策略 \(P_B(s\mid s')\)、奖励 \(R(x)\ge 0\)、配分函数 \(Z=\sum_x R(x)\)、目标 \(\pi^\star(x)=R(x)/Z\)、完整轨迹 \(\tau=(s_0\to\cdots\to s_n=x)\)、可学习配分估计 \(Z_\theta\)、轨迹平衡损失 \(\mathcal L_{\mathrm{TB}}(\tau)\)。
> 声明纪律同 guide：区分**期望梯度等价**与逐样本 loss 相同；venue 按官方页面写准，主会/Workshop/预印本分开；调研截止 2026-07-31，其后或无官方录用页面的工作按预印本处理。

本节把 guide §5.2-5.3 的两条“精确关系”补齐三块缺口：(a) VI 等价性的**梯度层面**推导；(b) 完全缺失的 **SMC（序贯蒙特卡洛）**视角；(c) **GFN+MCMC 混合方法**综述。统一主线见文末小结。

## 1. VI 等价性：从轨迹 KL 的梯度到 TB 梯度

把生成路径 \(\tau\) 视为潜变量，proposal 是前向轨迹分布 \(P_F(\tau)=\prod_t P_{F,\theta}(s_t\mid s_{t-1})\)，目标轨迹分布是反向扩展
\[
\pi^\star_\tau(\tau)=\frac{R(x)\,P_B(\tau\mid x)}{Z},\qquad \sum_{\tau:\,\tau\to x}\pi^\star_\tau(\tau)=\pi^\star(x)=\frac{R(x)}{Z}.
\]
定义与 guide 完全一致的轨迹 log-ratio（即 TB 残差）
\[
\delta(\tau)=\log\frac{Z_\theta\prod_t P_{F,\theta}(s_t\mid s_{t-1})}{R(x)\prod_t P_{B,\theta}(s_{t-1}\mid s_t)},\qquad \mathcal L_{\mathrm{TB}}(\tau)=\delta(\tau)^2 .
\]

**反向 KL 与其 REINFORCE 梯度。** 取真 \(Z\) 时 \(\delta\) 就是逐轨迹对数比，于是
\[
D_{\mathrm{KL}}\!\big(P_F\,\|\,\pi^\star_\tau\big)=\mathbb E_{\tau\sim P_F}\!\Big[\log\tfrac{P_F(\tau)}{\pi^\star_\tau(\tau)}\Big]=\mathbb E_{\tau\sim P_F}[\delta(\tau)].
\]
因为采样分布 \(P_F\) 本身依赖 \(\theta\)，对它求梯度得到 score-function（REINFORCE）估计（[Malkin et al., GFlowNets and Variational Inference, ICLR 2023](https://arxiv.org/abs/2210.00580) 式(10)）：
\[
\nabla_\theta D_{\mathrm{KL}}=\underbrace{\mathbb E_{\tau\sim P_F}\!\big[\nabla_\theta\delta(\tau)\big]}_{\text{固定 }P_B\text{ 时}=\,\mathbb E[\nabla_\theta\log P_F]=0}\;+\;\underbrace{\mathbb E_{\tau\sim P_F}\!\big[\delta(\tau)\,\nabla_\theta\log P_F(\tau)\big]}_{\text{REINFORCE 项}} .
\]
第一项用到 \(\mathbb E_{P_F}[\nabla_\theta\log P_F]=\sum_\tau\nabla_\theta P_F(\tau)=\nabla_\theta 1=0\)。**REINFORCE 项就是唯一实质梯度**，其“回报”正是 log-ratio \(\delta(\tau)\)。

**control variate 从哪里进、为何等于 TB。** REINFORCE 项方差高。减去一个基线 \(b\) 不改无偏性（因 \(\mathbb E_{P_F}[b\,\nabla_\theta\log P_F]=0\)）：
\[
\nabla_\theta D_{\mathrm{KL}}=\mathbb E_{\tau\sim P_F}\!\big[(\delta(\tau)-b)\,\nabla_\theta\log P_F(\tau)\big].
\]
Malkin et al. 指出：使方差最小的**全局基线**用滑动平均维护，其更新式恰好等于 TB 中 \(\log Z_\theta\) 的更新式（TB 关于 \(\log Z_\theta\) 是二次的）。换言之，**TB 里那个可学习的 \(\log Z_\theta\) 就是反向 KL 的最优控制变量**。以下记 \(\theta\) 为前向策略参数、\(\log Z_\theta\) 为单独标量参数（故 \(\nabla_\theta\delta=\nabla_\theta\log P_F\)），对停止梯度采样的 TB 直接求导，
\[
\nabla_\theta\,\mathbb E_{\tau\sim P_F}\!\big[\tfrac12\delta(\tau)^2\big]=\mathbb E_{\tau\sim P_F}\!\big[\delta(\tau)\,\nabla_\theta\log P_F(\tau)\big],
\]
与上式在 \(b=\mathbb E[\delta]\)（即 \(\log Z_\theta\) 吸收该常数）时**逐项一致**。这就是 Malkin et al. Prop. 1 的核心：**on-policy 下 TB 的期望梯度 = 带最优控制变量的反向 KL score-function 梯度**（是 REVERSE wake-sleep 一族的更新）。

**不止 TB：DB 与 SubTB 也各有 VI 对应。** Malkin et al. 在附录中把等价推到子轨迹层面：**SubTB \(\Leftrightarrow\) 嵌套变分推断（NVI）**（Zimmermann et al., 2021），而作为其逐边特例，**DB \(\Leftrightarrow\) 每步的 nested VI 目标**。因此 guide §3 的“局部 vs 全局”目标谱系，在 VI 一侧对应“逐边 / 逐子轨迹 / 整轨迹”的变分目标谱系。对应关系可小结为：

| GFN 目标 | 对应的 VI/更新 | 散度方向与性质 |
|---|---|---|
| forward KL 训练 \(P_F\) | wake 相 / 需目标样本或权重 | mean-seeking，易发现模式但欠精 |
| on-policy TB + \(\log Z_\theta\) | reverse KL + 最优控制变量 | mode-seeking，可 off-policy 免 IS |
| SubTB / DB | nested VI（子轨迹 / 逐边） | 局部信用，方差-偏置折中不同 |

**推广与限定词。** [da Silva et al., On Divergence Measures for Training GFlowNets, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8aaae73117c9266b35eb977e70fcf95f-Abstract-Conference.html) 把该 GFN–HVI 等价从 DAG 推广到一般可测拓扑空间，并给出关键诊断：直接用 KL、Rényi-\(\alpha\) 等 divergence 训练 GFN 之所以效果差，主因是**随机梯度方差**，而非目标本身错误；他们基于 **REINFORCE leave-one-out（RLOO）** 构造了可证明降方差的一族控制变量。据此，guide §5.2 的三个限定词有了梯度层面的落点：
- 是**期望梯度等价**，不是两个逐样本 loss 相同——差异恰在方差与控制变量。
- 等价需 **on-policy 或相应控制变量条件**；上面 \(\mathbb E[\nabla\log P_F]=0\) 的抵消只在 on-policy 成立。
- **off-policy**：VI 若换采样分布需 importance weighting（引入高方差）；TB 的 log-ratio 回归在全支持 off-policy 数据上仍保留正确零点，无需重要性权重。
- 方向选择也有后果：forward KL \(D_{\mathrm{KL}}(\pi^\star_\tau\|P_F)\) 是 mean-seeking（易发现模式但欠精、需目标样本或权重），reverse KL 是 mode-seeking（可漏模式），TB 在多峰实验里兼得两者之长且更受益于 off-policy 探索。

Malkin et al. 的两点实验观察（是经验结论，非定理，勿绝对化）：
- **观察 1**：on-policy 下 reverse-KL 与 TB 梯度在全批极限一致，但 TB 对学习率更鲁棒、JSD 收敛更低——差异来自方差而非期望梯度。
- **观察 2**：off-policy reverse-KL 明显劣于其 on-policy 版（因为要用 importance weighting，方差不可接受），而 off-policy TB 反而拟合最好、方差最低——这正是 GFN 相对 VI 的实际优势来源。

## 2. SMC 视角：proposal + 重加权 + 重采样，GFN 缺了后两件

序贯蒙特卡洛（SMC）用一群带权粒子逼近目标。给定一列中间目标 \(\pi_0,\dots,\pi_n=\pi^\star\)，每步对粒子 \(i\) 由 proposal 扩展并算**增量权重**：
\[
s_t^{(i)}\sim q_t(\cdot\mid s_{t-1}^{(i)}),\qquad
\alpha_t^{(i)}=\frac{\pi_t(s_{0:t}^{(i)})}{\pi_{t-1}(s_{0:t-1}^{(i)})\,q_t(s_t^{(i)}\mid s_{t-1}^{(i)})},
\]
再按归一化的 \(\{\alpha_t^{(i)}\}\) **重采样**，把算力集中到有希望的部分对象上；粒子数 \(N\to\infty\) 时加权经验分布渐近一致，且 \(\widehat Z=\prod_t\frac1N\sum_i\alpha_t^{(i)}\) 是 \(Z\) 的**无偏**估计。三件套是 **proposal \(q_t\)** + **重加权 \(\alpha_t\)** + **重采样**。

**GFN 前向采样只做 proposal。** 训练后 GFN 每个样本是一条独立的前向构造轨迹 \(\tau\sim P_F\)，**没有 per-step 重加权，也没有重采样**——相当于所有增量权重 \(\alpha_t\equiv\) 常数。它的“正确性”靠训练期把 \(P_F\) 拟合到 \(P_T(x)=R(x)/Z\)（把校正摊销进网络），而不是靠推断期的粒子权重；相应地，GFN 只给出**点估计** \(Z_\theta\)，而非 SMC 那种带方差保证的无偏 \(\widehat Z\)。二者的结构对应很紧：SMC 要用中间目标/twist \(\psi_t\) 来“总结未来 potential、校正部分序列的边际 \(\pi_t\)”，这正是 GFN 的**状态流** \(F(s)\)（等价于 soft value）通过 \(P_F(s'\mid s)=F(s\to s')/F(s)\) 所做的事——把下游总奖励折进部分对象。可对照：

- proposal kernel \(q_t\)　\(\leftrightarrow\)　前向策略 \(P_{F,\theta}(s_t\mid s_{t-1})\)
- 中间目标 / twist \(\pi_t,\psi_t\)　\(\leftrightarrow\)　状态流 \(F(s)\)（下游奖励摘要 / soft value）
- 重采样 + \(\widehat Z\)（推断期校正、渐近一致）　\(\leftrightarrow\)　训练期把校正摊销进 \(P_F\)、无推断期校正

**文献核实（venue 已核对）。**
- [Probabilistic Inference in Language Models via Twisted SMC, Zhao/Brekelmans/Makhzani/Grosse, **ICML 2024 Oral**](https://proceedings.mlr.press/v235/zhao24c.html)（arXiv 2404.17546）：学 twist \(\psi_t(s_{1:t})\) 使 \(p_0(s_{1:t})\psi_t\) 匹配真边际 \(\sigma(s_{1:t})\)，用重采样聚焦有希望前缀。它显式连接的是 **soft RL**（而 guide §5.1 已把 soft RL 与 GFN 相连），twist\(\leftrightarrow\)GFN 流是**结构同构**，但论文主线是 SMC-for-LLM，并非以 GFN 为标题。其副产品——双向 SMC 的 \(\log Z\) 上/下界，可用来在两个方向上估计推断分布与目标的 KL，因而是**评测任意摊销采样器（含 GFN）质量**的现成工具。
- [Reinforced Sequential Monte Carlo for Amortised Sampling](https://arxiv.org/abs/2510.11711)（arXiv 2510.11711；作者关键词标注投向 ICML，**官方主会录用页面未核实，按预印本处理**）：**显式统一** HVI/MaxEnt-RL/SMC——神经采样器的策略 \(P_F\) 充当 SMC proposal kernel、流函数 \(F\) 充当中间 twisting target；反过来用 SMC 的近目标加权样本作 off-policy 行为策略、经 SubTB 回训采样器，配自适应退火与重要性加权回放。
- 结论（诚实）：**直接“以 GFN 为 SMC proposal + 重采样”或把两者训练目标统一，目前实质只有 1 篇很新的预印本（Reinforced SMC）在正面做**；外加 1 篇已发表、结构高度相关但主线为 soft RL 的工作（twisted SMC, ICML 2024）。因此 **GFN×SMC 仍是新兴/开放方向**，尚未形成多篇独立复现的成熟研究线。（SMC-for-序列生成的旁支如 Lew et al. 2023, arXiv 2306.03081 属预印本，可作背景，不宜当作 GFN 结论。）

## 3. 混合方法：在“摊销”和“渐近正确”之间插值

- **GFN 作 MCMC proposal（back-and-forth）。** [EB-GFN, Zhang et al., **ICML 2022**](https://proceedings.mlr.press/v162/zhang22v.html)（arXiv 2202.01361）用 \(P_B\) 先破坏 \(K\) 步、再用 \(P_F\) 重构，得到一个近似 large-block Gibbs 的 Metropolis–Hastings 提议；其 Prop. 2：GFN 完美拟合时 MH 接受率恒为 1。优点是能借组合结构做**大跳**、跨模式转移，弥补局部 Gibbs 混合慢。
- **局部 MCMC 精修 GFN 样本。** [Local Search GFlowNets (LS-GFN), Kim et al., **ICLR 2024 Spotlight**](https://openreview.net/forum?id=6cFcw1Rxww)（arXiv **2310.02710**，注意是 2023-10 而非 2306）：对采得的 \(\tau\) 反复“\(P_B\) 破坏 \(K\) 步 → \(P_F\) 重构”，用**确定性**或 **back-and-forth Metropolis–Hastings**（沿用 EB-GFN 的接受比）过滤，接受的高奖励轨迹回填训练集，把样本偏置到高奖励区。
- **连续空间对应。** [Sendera et al., Improved off-policy training of diffusion samplers, **NeurIPS 2024**](https://openreview.net/forum?id=vieIamY2Gi)（arXiv 2402.05098）在连续 GFN/扩散采样器上用**目标空间的 Metropolis-adjusted Langevin（MALA）局部搜索 + replay buffer**精修样本，缓解 mode collapse。
- **搜索-引导训练（预印本，谨慎）。** [Morozov et al., Improving GFlowNets with Monte Carlo tree search](https://arxiv.org/abs/2406.13655)（arXiv 2406.13655，**预印本**）把 MCTS 式前瞻用于 GFN 采样/训练；作为最新方向可提，但尚无主会官方页面，按预印本对待。
- **背景（已在 guide）。** 最早的 [Flow Network based generative models（Bengio et al., NeurIPS 2021）](https://proceedings.neurips.cc/paper/2021/hash/e614f646836aaed9f89ce58e837e2310-Abstract.html) 就是把 GFN 作为比 MCMC/MARS 更快发现多模式的采样器提出；但 guide §附录已提醒“GFN 保证比 MCMC 更快找新模式”并无无条件保证，混合方法正是承认二者互补。
- 统一直觉：这些方案都在 **GFN 的摊销前向采样** 与 **MCMC 的渐近正确** 之间插值——用 GFN 提供跨模式大跳的 proposal 解决 mixing，用 MH/接受-拒绝或 Langevin 修正找回渐近保证；退化到 \(K=n\)（整条重构）就回到纯 GFN 采样。

## 4. 三方法对比与 mode mixing 纠误

| 维度 | 变分推断 (HVI) | MCMC | SMC | GFN（对照锚点） |
|---|---|---|---|---|
| 摊销位置 | 训练期（学采样器） | 无摊销，推断期付全部成本 | 主要推断期（可摊销 twist/proposal） | 训练期（把校正摊销进 \(P_F,F\)） |
| 渐近保证 | 变分近似，一般有 KL gap | 固定目标下不变分布可证 | 粒子数 \(\to\infty\) 渐近一致、\(\log Z\) 无偏估计 | 零损失+全支持+足够表达+全局最优才精确；神经近似无逐样本保证 |
| mode 混合行为 | 取决于散度方向（reverse 易 mode-seeking 漏峰） | 靠链在模式间混合，混合时间可指数级 | 重采样可跨模式，但受 proposal/权重退化限制 | 推断期无需单链混合，但**依赖训练期探索先发现模式** |
| 典型失效模式 | 后验方差低估、mean/mode-seeking 偏置、off-policy 需 IS 高方差 | 混合慢、卡在单一模式、需长 burn-in | 权重退化/粒子塌缩、中间目标设计敏感 | 探索失败/mode collapse/共享前缀偏置、\(Z_\theta\) 拟合差则不可信 |

**纠误：“GFN 不受 mode mixing 影响”是误说。** 准确表述是：推断期 GFN 每个样本是一条独立前向轨迹，**不需要沿同一条马尔可夫链从一个远端模式混到另一个**，因而没有 MCMC 意义下的 inter-mode mixing time。但它并未消除 mode 问题，而是把难点**从推断期的链混合搬到了训练期的探索与信用分配**：未被采样到的模式拿不到梯度信号，共享前缀偏置与 mode collapse 依旧存在（呼应 guide §5.3 与“先给结论”第 3 点）。§3 的混合方法之所以有效，恰恰反证了纯摊销前向采样会漏采/欠采模式，需要 MCMC/局部搜索补足。

常见误说 → 准确表述（与 guide 保持一致）：
- ❌“GFN 不受 mode mixing 影响” → ✅ 无推断期单链混合，但把混合难题移到了训练期探索；混合方法用来补足漏采。
- ❌“TB 有定理，所以训练稳定” → ✅ 定理只讲零残差/全支持期望 loss 的全局最优，不保证有限样本 SGD 动态稳定。
- ❌“GFN 一定比 MCMC 更快找到新模式” → ✅ 无无条件保证；优劣取决于目标结构、训练预算、混合性质与样本复用次数。
- ❌“on-policy TB 与反向 KL 是同一个 loss” → ✅ 只是**期望梯度等价**（差在方差与控制变量），off-policy 行为亦不同。

## 5. 选型建议（衔接 guide §1）

- 只需**一次性、低成本、反复**产生多样样本，且模式间有可学习共享结构：优先 GFN（把成本摊销到训练）。
- 需要**渐近正确的单一目标采样**、样本量不大、目标固定：MCMC/SMC 往往更省心，无需承担训练。
- 目标是**序贯/自回归生成 + 明确中间 potential**（如 LLM 受控生成）：twisted SMC 与 GFN 高度互通，可先用 GFN/soft-value 学 twist，再在推断期加重采样换取一致性。
- **训练期漏模式或高奖励样本不足**：叠加 §3 的局部搜索/back-and-forth MH（离散）或 MALA（连续），用少量推断期计算换取质量，而不必推翻摊销框架。

## 小结：一个统一视角

四类方法都只用未归一化的 \(R(x)\)、都想从多峰 \(\pi^\star=R/Z\) 采样，真正的分野只在**校正成本付在何处**：VI 与 GFN 付在**训练期**（摊销进采样器，且 GFN = 带内置控制变量 \(\log Z_\theta\) 的 HVI，可 off-policy 而不需重要性权重），MCMC 与 SMC 付在**推断期**（靠链混合或粒子重加权+重采样换取渐近保证）。贯穿其中的同一物件是“未来奖励的摘要”：GFN 的流 \(F(s)\)、SMC 的 twist \(\psi_t\)、soft RL 的最优值函数三者同构，都用来校正部分对象的边际。混合方法正是沿这条轴插值，也说明所谓“GFN 免疫 mode mixing”应改述为**把 inference-time 链混合换成了 training-time 探索**。

## 参考文献（含发表状态）

- Malkin, Lahlou, Deleu, Ji, Hu, Everett, Zhang, Bengio. GFlowNets and Variational Inference. **ICLR 2023**. https://arxiv.org/abs/2210.00580
- Malkin, Jain, E. Bengio, Sun, Y. Bengio. Trajectory Balance. **NeurIPS 2022**（附录 A.3 给出 on-policy TB 与某 KL 的等价）. https://arxiv.org/abs/2201.13259
- da Silva et al. On Divergence Measures for Training GFlowNets. **NeurIPS 2024**（RLOO 控制变量、方差诊断）. https://proceedings.neurips.cc/paper_files/paper/2024/hash/8aaae73117c9266b35eb977e70fcf95f-Abstract-Conference.html
- Zhao, Brekelmans, Makhzani, Grosse. Probabilistic Inference in Language Models via Twisted SMC. **ICML 2024（Oral）**. https://proceedings.mlr.press/v235/zhao24c.html
- Reinforced Sequential Monte Carlo for Amortised Sampling. **预印本**（arXiv 2510.11711；作者标注投向 ICML，官方主会页未核实）. https://arxiv.org/abs/2510.11711
- Zhang, Malkin, Liu, Volokhova, Courville, Bengio. Generative Flow Networks for Discrete Probabilistic Modeling（EB-GFN）. **ICML 2022**. https://proceedings.mlr.press/v162/zhang22v.html
- Kim, Yun, E. Bengio, D. Zhang, Y. Bengio, Ahn, Park. Local Search GFlowNets. **ICLR 2024（Spotlight）**. https://openreview.net/forum?id=6cFcw1Rxww
- Sendera, Kim, Mittal, Lemos, Scimeca, Rector-Brooks, Adam, Bengio, Malkin. Improved off-policy training of diffusion samplers. **NeurIPS 2024**. https://openreview.net/forum?id=vieIamY2Gi

