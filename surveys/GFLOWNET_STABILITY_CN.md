# GFlowNet 分布保真度理论：从损失到 TV 界

> 本文深化理论指南 §4.4/§8.1。
> 来源：GFlowNet 调研 2026-08 审查扩充（E18）。核心索引见 [README](README.md)。

---

> 本小节把散落在 guide §4.3（评估边界）、§4.4（loss 能否认证终止分布）、§5.2（VI/KL）、§8.1（稳定性）与 OT 专题里的"训练损失 → 分布误差"结果，整理成一条完整的**分布保真度（distributional fidelity）**理论线。
> 主线论文 [Stable GFlowNets（arXiv:2605.01729）](https://arxiv.org/abs/2605.01729) 是 **2026 年预印本**，本节的定理编号、公式与陈述已按其全文联网核实（见下"核实说明"），结论仍需更多独立复现。
> 声明纪律：预印本一律标注；定理逐条归属到论文；凡"论文未显式给出、由本节按标准事实推得"的部分，一律写作 **【本节推断】**。

## 1. 核心问题：低 loss 是否等于分布准？

guide §4.3 已给出一个逻辑事实：奖励直方图相同（\(\operatorname{Law}_{P_\theta}(R(X))=\operatorname{Law}_{P^\star}(R(X))\)）**不推出** \(P_\theta(X)=P^\star(X)\)——只要多个对象同奖励，就能在对象间严重错配却保持相同的均值与直方图。本节把问题再往前推一层：**训练 loss（而非评估指标）小，是否意味着终止分布 \(P_T\) 准？**

这必须拆成两个方向，二者都不平凡（guide §4.4 已提示，此处补全）：

- **正向（loss ⟸ 分布）**：分布已很准，是否保证 loss 小？**否。** 见 §2 的 Proposition 3.3–3.4。
- **反向（loss ⟹ 分布）**：loss（逐轨迹）小，是否保证分布准？**是，有显式界。** 见 §2 的 Theorem 3.5–3.6。

这条"反向可认证、正向不可认证"的不对称，是整条保真度线的骨架。

## 2. Stable GFlowNets（2605.01729）：逐条结果梳理

**核实说明**：该文全文标题为 *Stable GFlowNets with **TV Monitoring** and Probabilistic Guarantees*（guide/任务用其简称"…with Probabilistic Guarantees"），作者来自 Purdue 与 UC Santa Cruz。以下编号、公式均以其正文 §3 为准。

### 2.1 正向不可认证：小 TV 不锁死 loss

- **Proposition 3.3（增量模式覆盖的 TV 界）**：引入新奖励后，用局部对比比 \(\Lambda_{\mathcal Y}=Z^\star_{\mathcal Y}/(Z^\star_{\mathcal Y}+\sum_{\mathcal Y}R')\)，有 \(\tfrac{Z^\star-Z^\star_{\mathcal X_{sub}}}{Z^\star}(1-\Lambda_{\mathcal X})\le \operatorname{TV}(P_T,\pi_{target})\le 1-\Lambda_{\mathcal X}\)。全局 TV 由**聚合**对比比 \(1-\Lambda_{\mathcal X}\) 控制，新模式奖励质量小则 TV 可以很小。
- **Proposition 3.4（loss 尺度由最坏局部对比比决定）**：对 \(\text{GFN}\in\{\text{FM},\text{DB},\text{TB},\text{subTB}\}\)，\(\sup|\mathcal L_{\text{GFN}}|=(\log\min_{\{x\}\subseteq\mathcal X_{sub}}\Lambda_{\{x\}})^2\)。

两条合起来即 guide §4.4 第 1 点的严格版：**TV 由聚合比控制、loss 由最坏局部比控制**，二者可任意脱钩——罕访状态上一次大的相对奖励提升，就能在 TV 很小的同时制造无界 loss spike。

### 2.2 反向可认证：loss → TV 界（Theorem 3.5）

论文对**每条轨迹**的一致界给出闭式结果：

\[
\mathcal L_{\mathrm{TB}}(\tau)\le c^2\ \forall\tau
\ \Longrightarrow\
\operatorname{TV}(P_T,\pi_{target})\le 1-e^{-2c}\quad(\text{Eq. }7),
\]

且此界**与轨迹长度无关**（TB 直接约束整段路径一致性）。对局部目标（transition-level）：

\[
\mathcal L_{\mathrm{DB}}(s,s')\le c^2\ \text{或}\ \mathcal L_{\mathrm{FM}}(s')\le c^2
\ \Longrightarrow\
\operatorname{TV}(P_T,\pi_{target})\le 1-e^{-2Lc}\quad(\text{Eq. }8),
\]

\(L\) 为最大轨迹长度，界在 log 域随深度线性退化。

> **对 guide §4.4 注记的校正（重要）**：guide §4.4 曾把 \(1-e^{-2Lc}\) 标为"本报告按论文推断、论文只对 TB 给显式公式"。经核实全文，**\(1-e^{-2Lc}\) 正是论文 Theorem 3.5 的 Eq (8) 显式陈述**（transition-level 情形），并非本报告推断。这条更强、且解释了 guide §4.5 的分层："全轨迹约束（TB）给不随深度累积的全局界，局部约束（DB/FM）的界随 \(L\) 退化"是**论文定理**，而非启发式。

### 2.3 有限样本概率证书（Theorem 3.6 + Corollary 3.7）

Eq (7) 要求"所有轨迹"一致界，不可直接检查。论文改用**双向采样**构造证书，且**与状态空间大小无关**：

- **Theorem 3.6**：定义目标诱导轨迹分布 \(\hat\pi(\tau)=\pi_{target}(x_\tau)P_B(\tau|x_\tau)\)；从 \(\hat\pi\) 采 \(m\) 条（\(x\sim\pi_{target}\) 再 \(\tau\sim P_B\)）、从 \(P_F\) 采 \(n\) 条，取 \(c=\max_i\sqrt{\mathcal L_{\mathrm{TB}}(\tau_i)}\)。则以置信度 \(1-2\alpha\)，
\[
\operatorname{TV}(P_T,\pi_{target})\le (e^{2c}-1)+\tfrac{\log(1/\alpha)}{m}+\tfrac{\log(1/\alpha)}{n}\quad(\text{Eq. }9).
\]
- **Corollary 3.7（子图证书）**：把全局 \(\mathcal X\) 换成子集 \(\mathcal X_{sub}\)，得到 \(\mathcal X_{sub}\) 上的 TV 证书（Eq. 10）；当 \(\mathcal X_{sub}\) 主导奖励质量且 \(Z\approx\sum_{\mathcal X_{sub}}R\)（Eq. 11）时，可外推到全局近最优。

这正是 guide §4.4 与 §8.9(3) 的核心实践限制：Eq (9) 的 \(m\) 条采自 \(\hat\pi\)——**需要能从真实目标分布采样**，而这常是原问题本身；大空间只能退到子图证书。

### 2.4 reference flow 与稳定性—保真度折中（Def 3.8 / Remark 3.9 / Theorem 3.10）

- **Definition 3.8（轨迹 reference flow）**：注入 \(\delta(\tau)>0\)，令 \(F_{aug}(\tau)=ZP_F(\tau)+\delta(\tau)\)、\(R_{aug}(\tau)=R(\tau)+\delta(\tau)\)。
- **Remark 3.9（稳定化）**：\(\mathcal L_{aug}(\tau)=\gamma^{-2}\mathcal L_{\mathrm{TB}}(\tau)\)，reference flow 按因子 \(\gamma>1\) **成比例压低 TB loss 尺度**（Eq. 12），并给出使 \(\mathcal L_{aug}(\tau)\le c^2\) 的最小注入量 \(\delta_c(\tau)\)（Eq. 13）。它降低 loss 幅度而**不改变全局最优**。
- **Theorem 3.10（保真度折中）**：设总注入 \(\Delta=\sum_\tau\delta(\tau)\)，若 \(\mathcal L_{aug}(\tau)\le c^2\)，则
\[
\operatorname{TV}(P_T,\pi_{target})\le (1-e^{-2c})\Big(1+\tfrac{\Delta}{Z^\star}\Big)\quad(\text{Eq. }14\ \text{的上界形式}).
\]
乘性劣化因子 \((1+\Delta/Z^\star)\) 就是 guide §4.4"稳定性与保真度折中"的定量形式：注入越多、loss 越稳，但保真度上界越松，过大 \(\Delta\) 会使证书**空洞（vacuous）**（论文 §5 亦指出固定 \(\delta\) 的这一失效）。

### 2.5 可优化阈值的证书与监控（Theorem 3.11 + Algorithm 1）

- **Theorem 3.11**：把 reference flow 阈值 \(c\) 变为可优化，给出对所有 \(c\in\mathcal C\) 同时成立、置信度 \(1-2\alpha\) 的 TV 证书 \(\mathcal B_{TV}\)（Eq. 16），及训练期廉价**监控量** \(\mathcal M_{TV}\)（Eq. 15）。
- **Algorithm 1**：自适应注入 \(\delta(\tau)\) 并用 **top-\(K\) 高奖励 buffer** 作为 \(\mathcal X_{sub}\) 做子图证书。实验（Regular Tree、Hypergrid、L14-RNA1、sEH）显示 \(\mathcal M_{TV}\) 与真实 TV 相关系数 \(>0.9\)，而 \(\mathcal B_{TV}\) 在小环境非平凡、在大 Hypergrid/RNA 上偏保守。

> 任务把"top-\(K\) buffer"挂在 Theorem 3.6 名下；准确归属应为 **Corollary 3.7（子图证书）+ Algorithm 1（top-\(K\) 实现）**，Theorem 3.6 本身是全局双向采样证书。

## 3. 与 f-TB（2605.15417）、divergence 目标（T17）的方差—稳定性关系

- **T17**（[On Divergence Measures，NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8aaae73117c9266b35eb977e70fcf95f-Abstract-Conference.html)，已发表）：不同 divergence 诱导不同 GFlowNet 目标，实践失败**多来自梯度估计方差**而非 divergence 本身，用 control variate 缓解。
- **f-TB**（[arXiv:2605.15417](https://arxiv.org/abs/2605.15417)，**2026 预印本 / ICML 2026**）：translation-invariant log-prob loss ↔ \(f\)-divergence 一一对应，保留 off-policy 正确零点；不同 \(f\) 改变 mode-covering/seeking 倾向与方差。

**关系厘清**：这两条线调的是"用哪种 divergence 几何 + 如何降方差"，Stable 调的是"如何压住最坏局部对比比 → 压住 loss 幅度 → 拿到 TV 证书"。二者是**正交的稳定化杠杆**：control variate 降的是估计量**方差**（优化收敛），reference flow 降的是 loss **幅度上界**（可认证性）。【本节推断】Proposition 3.4 的 \(\sup|\mathcal L|=(\log\min\Lambda)^2\) 对 FM/DB/TB/subTB 同形，说明"最坏局部对比"是平方 log-ratio 几何的共性，换 \(f\)（f-TB）改变的是优化路径与方差，未必改变这一最坏尺度；而 **Theorem 3.5 的 loss→TV 界是针对平方 TB loss 陈述的，一般 \(f\)-TB 损失是否有对应闭式 loss→TV 界，是开放问题**（见 §6）。

## 4. 其它保真度结果与开放问题

- **VI 视角的 KL 界**：guide §5.2 指出 on-policy 期望梯度下 TB 与轨迹空间层次 VI/KL 精确相联。【本节推断】由此可走一条"KL→TV"的替代保真路径：Pinsker 给 \(\operatorname{TV}(P_F,\hat\pi)\le\sqrt{\tfrac12\mathrm{KL}(P_F\|\hat\pi)}\)，再由 data-processing 得终止边缘 \(\operatorname{TV}(P_T,\pi_{target})\le\operatorname{TV}(P_F,\hat\pi)\)。这与 Theorem 3.5 互补：3.5 用**逐轨迹最坏 loss**（\(L_\infty\) 型），KL 路线用**平均 KL**（\(L_1\) 型），前者可认证、后者更贴合期望梯度训练但不给 high-probability 证书。以上均为标准事实的组合，非某篇论文的定理。
- **OT 分析里的"balance 残差 → OT 误差界"开放问题**：OT 专题（[2606.06272](https://arxiv.org/abs/2606.06272)，**2026 SPIGM Workshop/预印本**）的 permutation 实验显示，正则 TB 的系数 \(\lambda\) 越大路径越短但**终止边缘偏差越大**、\(\lambda\) 越小采样越准但路径越长——即 balance 残差同时耦合"目标边缘 TV"与"传输代价次优性"，目前**没有把二者作为残差函数上界的定理**。【本节推断】把 Stable 的 loss→TV 界特化到固定源/目标边缘的 OT-GFN，配合 OT 专题建议的 Kantorovich potential 作 critic 监控 primal–dual gap，有望得到"balance 残差 →（边缘 TV，传输代价 gap）"证书；这是 OT 与保真度两条线的交汇开放点。

## 5. "保证类型"对比表

| 保证 | 假设强度 | 界的形式 | 可计算性 | 是否需真实分布 |
|---|---|---|---|---|
| Thm 3.5 Eq.7（TB→TV） | 强（**所有**轨迹 \(\mathcal L_{TB}\le c^2\)） | 确定性，\(1-e^{-2c}\)，长度无关 | 直接验证不可行（全轨迹一致界） | 否（界本身不需，验证前提难） |
| Thm 3.5 Eq.8（DB/FM→TV） | 强（所有 transition \(\le c^2\)） | 确定性，\(1-e^{-2Lc}\)，随 \(L\) 退化 | 同上，且依赖最大长度 \(L\) | 否 |
| Thm 3.6（全局概率证书） | 中（双向采样 + 有限 \(m,n\)） | 高概率 \(1-2\alpha\)，\((e^{2c}-1)+\log(1/\alpha)(\tfrac1m+\tfrac1n)\) | 可算，但需从 \(\hat\pi\) 采样 | **是**（从 \(\pi_{target}\) 采 \(m\) 条） |
| Cor 3.7 + Alg.1（子图/top-K 证书） | 中（子图主导奖励质量） | 高概率，\(\mathcal X_{sub}\) 上 TV | 可算（top-\(K\) buffer） | 子图内需目标采样 |
| Thm 3.10（reference flow 折中） | 中（\(\mathcal L_{aug}\le c^2\)） | 确定性上界 ×\((1+\Delta/Z^\star)\) | 依赖 \(\Delta,Z^\star\) | 需 \(Z^\star\)（或估计） |
| Thm 3.11 \(\mathcal M_{TV}\)（监控） | 弱（后向采样若干条） | 无形式保证，经验相关 \(>0.9\) | 廉价，训练期可算 | 需后向采样，不需精确 \(\pi_{target}\) 概率 |
| KL→TV（VI+Pinsker）【本节推断】 | 中（on-policy 期望梯度） | 确定性，\(\sqrt{\mathrm{KL}/2}\) | 需估 KL（方差大） | 需目标以定义 KL |

## 6. 开放问题

1. **large space 下证书的可扩展性**：Eq (9) 依赖从 \(\hat\pi\)（含 \(\pi_{target}\)）采样；大组合空间只能退到 top-\(K\) 子图证书，如何在**不先解原采样问题**的前提下上界全局 TV，仍是 guide §8.9(3) 的核心开放点。
2. **非渐近 / 更紧的界**：\(\mathcal B_{TV}\) 因最坏依赖 reference flow 而偏保守（论文 conclusion 自陈）；求随 \(m,n\) 更快收缩、随 \(\Delta\) 不易空洞的非渐近界。
3. **连续与非无环空间**：Theorem 3.6 的双向采样在连续设定下难做（论文列为 future work），需与 guide §7 连续/非无环理论对接。
4. **f-TB 的 loss→TV**：一般 \(f\)-divergence 损失是否有 Theorem 3.5 式的闭式 loss→TV 界（§3 开放点）。
5. **OT-GFN 的残差 → 误差证书**：把 balance 残差映射到"边缘 TV + 传输代价 gap"（§4 开放点）。

> 一句话总结：**Stable GFlowNets 把"低 loss 是否等于分布准"从口号变成不对称定理——分布准不保证 loss 小（Prop 3.3/3.4），但逐轨迹 loss 小可显式认证 TV 小（Thm 3.5），代价是要么假设全轨迹一致界、要么需从目标分布采样构造概率证书（Thm 3.6/Cor 3.7）。** 截至 2026-08-14，主线仍是 2026 预印本，宜作理论诊断工具而非成熟训练配方。

