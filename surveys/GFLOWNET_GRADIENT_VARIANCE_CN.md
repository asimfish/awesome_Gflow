# 训练目标的梯度结构与方差

> 本文扩展理论指南 §3（训练目标）。
> 来源：GFlowNet 调研 2026-08 审查扩充（E06）。核心索引见 [README](README.md)。

---

> 定位：guide §3 比较了各训练目标"约束了什么"，§3.8 的对照表停留在约束单位层面。本节下沉一层，比较**随机梯度本身**的结构与方差，解释"TB 高方差、SubTB 折中、DB 局部"这句社区常识里哪些部分有证明、哪些只有实验、哪些只是推断。记号沿用 §2–§2.4 与 §3 开头的约定（终止对象 \(x\) 的唯一子节点为 \(s_f\)，故 \(F(x)=R(x)\)；真配分函数 \(Z=\sum_x R(x)\)，学习参数为 \(Z_\theta\)）。声明纪律：每个关键论断标注**论文证明 / 论文实验 / 预印本 / 本节推断 / 社区经验**。

## 3.9 训练目标的梯度结构与方差：为什么 TB 高方差、SubTB 折中、DB 局部

### 3.9.1 同一族平方 log-残差，三种梯度结构

对部分轨迹 \(\tau_{m:n}=(s_m\to\cdots\to s_n)\) 定义统一残差

\[
\delta_{m:n}
=\log\frac{F_\theta(s_m)\prod_{t=m+1}^{n}P_{F,\theta}(s_t|s_{t-1})}{F_\theta(s_n)\prod_{t=m+1}^{n}P_{B,\theta}(s_{t-1}|s_t)},
\qquad F_\theta(s_0)=Z_\theta,\quad F_\theta(x)=R(x),
\]

则 FM/DB/TB/SubTB 的 loss 都是 \(\delta^2\) 的加权和，差别在于 \(\delta\) 的支撑长度与权重。

**TB：一个全局标量误差广播给整条轨迹。** 对完整轨迹 \(\tau=(s_0\to\cdots\to s_n=x)\)，记 \(\delta_{\mathrm{TB}}(\tau)=\delta_{0:n}\)，

\[
\nabla_\theta\mathcal L_{\mathrm{TB}}(\tau)
=2\,\delta_{\mathrm{TB}}(\tau)\Big(\nabla_\theta\log Z_\theta
+\sum_{t}\nabla_\theta\log P_{F,\theta}(s_t|s_{t-1})
-\sum_{t}\nabla_\theta\log P_{B,\theta}(s_{t-1}|s_t)\Big).
\]

每个动作收到同一份误差信号：终端奖励无自举地直达 \(s_0\)（信用直接），但单个标量 \(\delta\) 的噪声也乘遍全轨迹 score（噪声同样直达）。且 \(\partial\mathcal L_{\mathrm{TB}}/\partial\log Z_\theta=2\delta\)，故 \(\log Z_\theta\) 的 SGD 更新等价于对逐 batch 残差做加权滑动平均（[TB 论文](https://arxiv.org/abs/2201.13259)脚注 3，论文陈述）。

**DB：逐边残差，靠 \(F_\theta\) 自举串联。**

\[
\nabla_\theta\mathcal L_{\mathrm{DB}}(s,s')
=2\,\delta_{s:s'}\,\nabla_\theta\big(\log F_\theta(s)+\log P_{F,\theta}(s'|s)-\log F_\theta(s')-\log P_{B,\theta}(s|s')\big).
\]

单项只涉及一条边与两端状态流，幅度受局部量控制；终端信息要经 \(F_\theta\) 沿图逐步回传（TD 式自举，见 §3.2 的"深层终端信号传播慢"）。

**SubTB(\(\lambda\))：\(\lambda\) 加权的多尺度组合。**

\[
\nabla_\theta\mathcal L_{\mathrm{SubTB}(\lambda)}(\tau)
=\frac{\sum_{0\le m<n}\lambda^{\,n-m}\;2\,\delta_{m:n}\,\nabla_\theta(\text{对应 log 项})}{\sum_{0\le m<n}\lambda^{\,n-m}},
\]

\(\lambda\to0^+\) 时退化为轨迹内 DB 平均，\(\lambda\to\infty\) 时退化为 TB（[Madan et al. 2023](https://proceedings.mlr.press/v202/madan23a.html)，论文证明）。关键机制：\(\log F_\theta(s)\) 出现在所有以 \(s\) 为端点的子轨迹残差中，等于**把 TB 残差中以 \(s\) 截断的随机片段替换为其期望的可学习估计**。原文明确类比 actor-critic 的 value baseline，并指出"替换必引入相对 TB 梯度的偏差、预期降低方差"是交给实验检验的假设（论文原文定性 + 实验显示，见 3.9.2）。

**on-policy 期望梯度的身份（论文证明）。** 当 \(\tau\sim P_F\) 且只看 \(P_F\) 的参数时，对任意 \(Z_\theta\) 取值，

\[
\tfrac12\,\mathbb E_{\tau\sim P_F}\big[\nabla_\theta\mathcal L_{\mathrm{TB}}(\tau)\big]
=\nabla_\theta D_{\mathrm{KL}}\big(P_F(\tau)\,\big\|\,R(x)P_B(\tau|x)/Z\big),
\]

因为 \(\log Z_\theta\) 乘的是零均值 score 项：它不改变期望梯度，只作为 baseline（控制变量）改变方差，其更新规则恰与 REINFORCE 的 running-average 全局 baseline 一致（[TB 论文附录 A.3](https://arxiv.org/abs/2201.13259)；[GFlowNets and VI](https://arxiv.org/abs/2210.00580) Proposition 1，后者附录 C 把等价扩展到 SubTB/DB 与 nested VI）。A.3 还证明：在最优点邻域（固定 \(P_B\)、括号项 \(1+2\log\frac{R P_B}{P_F}>0\) 的条件下），TB 估计的方差**低于**保留零均值项的朴素 REINFORCE 估计，并给出方差差的显式表达式（论文证明，相对性结论）。

**VarGrad 恒等式。** 把 \(\log Z_\theta\) 在每个 batch 内先闭式最优化，剩下的 TB loss 恰是 batch 内 log-ratio 的经验方差——即 VI 文献的 log-variance / [VarGrad](https://arxiv.org/abs/2010.10436) loss（Richter et al. 2020，论文证明其在特定条件下比 score-function 估计低方差；GFlowNet 语境的等价表述见 [Sendera et al. 2024](https://arxiv.org/abs/2402.05098)）。这解释了为什么条件任务中可以不学 \(Z_\theta\) 而用组内估计（见 3.9.4）。

### 3.9.2 方差来源：轨迹长度、奖励稀疏性、off-policy 程度

**轨迹长度 \(n\)。** \(\delta_{\mathrm{TB}}\) 是 \(n\) 项逐步 log 比值误差与 \(\log R\) 之和，score 因子也是 \(n\) 项之和；若各步误差弱相关，残差波动按 \(\sqrt n\) 量级增长、score 范数按 \(n\) 量级增长，两者相乘使单样本梯度方差随 \(n\) 超线性增长——**本节推断**（结构启发式，同 RL 中 Monte Carlo return 对 TD 的关系），目前**没有任何论文给出 \(\operatorname{Var}[\nabla\mathcal L_{\mathrm{TB}}]\) 关于 \(n\) 的已证公式**。有论文陈述的定性版本：TB 论文结论明确说"依赖长轨迹采样、随机梯度方差可能更高，是 TB 在困难环境的可能局限"（论文原文，经验层面）。

**奖励稀疏性。** \(\log R(x)\) 直接进入 \(\delta_{\mathrm{TB}}\)：奖励动态范围大、背景奖励趋零时，残差跨轨迹重尾化，对应 REINFORCE 回报重尾问题；\(R_{\min}\) 也出现在已知鲁棒性界的分母上（见下）。实验证据：SubTB 论文的稀疏 hypergrid（背景奖励 \(10^{-4}\)）中，TB 在大于 \(8\times8\) 的网格上无法发现全部模式，SubTB(\(\lambda\)) 保持强劲（论文实验）。

**off-policy 程度。** balance 族损失的已证优势：off-policy 采样**不需要重要性权重**，方差不会像 IS-VI 那样随 proposal 与目标错配增长（[GFlowNets and VI](https://arxiv.org/abs/2210.00580)，论文证明+实验）。代价是语义变化：训练分布只是逐轨迹多目标问题的 scalarization 权重（同文原表述）；off-policy 时期望梯度不再对应任何 \(f\)-divergence 的梯度，只保留同一全局零点（[\(f\)-TB](https://arxiv.org/abs/2605.15417)，论文证明）。因此"方差不爆炸"不等于"学得好"：replay 与探索分布决定**哪些轨迹的残差被压**，覆盖不足直接触发 §4.1 的支持条件问题。

**正式方差理论现状（截至 2026-08 联网检索）：**

1. **已证・相对性结论**：TB 附录 A.3 的近优邻域方差比较；VarGrad 对 score-function 估计的方差比较（特定条件）；divergence 训练中基于 REINFORCE leave-one-out 与 score-matching 的控制变量"可证降方差"（[On Divergence Measures for Training GFlowNets](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8aaae73117c9266b35eb977e70fcf95f-Abstract-Conference.html)，NeurIPS 2024；该文同时实验显示 divergence 直训失败的原因正是梯度方差过大）。
2. **已证・速率层面**：2025 年单作者预印本 [Secrets of GFlowNets' Learning Behavior](https://arxiv.org/abs/2505.02035) 给出 FM \(O(1/\sqrt T)\) 对 DB \(O(1/T^{1/3})\) 的收敛率（后者归因于比值型梯度的高方差）、\(O(|S|\,L\log(|S|/\delta)/\epsilon^2)\) 样本复杂度（轨迹长度 \(L\) 线性进入）、误差沿轨迹累积的命题，以及奖励噪声下以 \(R_{\min}^{-2}\)、\(R_{\min}^{-4}\) 为分母的鲁棒性界（即上文所指）。注意：该文的 TB/DB 使用非对数残差形式、假设固定 \(P_B\) 与指数参数化，且未见同行评审记录——引用需谨慎（预印本）。
3. **不存在**：TB/DB/SubTB 梯度方差关于轨迹长度或奖励稀疏度的显式解析界，以及 SubTB(\(\lambda\)) 方差随 \(\lambda\) 单调的定理。"DB 低方差、SubTB 居中、TB 高方差"排序目前最强证据是 Madan et al. 2023 的 tabular \(8\times8\) hypergrid 实验：小批与大批梯度的余弦自相似度 DB > SubTB(0.8) > TB；且小批 SubTB 梯度比小批 TB 梯度**更接近大批 TB 梯度**——有偏但低方差的估计反而更接近真梯度（论文实验）。

### 3.9.3 信用分配视角：一张对比表

| 目标 | 信号粒度 | 传播距离 | 参数化需求 | 单项梯度方差 | 适用场景 |
|---|---|---|---|---|---|
| FM | 状态级守恒残差 | 一步（自举） | 边流 \(F_\theta(s\to s')\) | 低（推断，同 DB 类比） | 父边可枚举、短轨迹 |
| DB | 边级残差 | 一步（自举） | \(F_\theta(s),P_F,P_B\) | 低（论文实验） | 局部/中间信号可用、深层信号可等 |
| SubTB(\(\lambda\)) | 全部子轨迹加权 | \(1\ldots n\) 连续可调 | \(F_\theta(s),P_F,P_B\)+调 \(\lambda\) | 中（论文实验） | 长轨迹、稀疏奖励 |
| TB | 整轨迹残差 | 直达 \(s_0\)（无自举） | \(Z_\theta,P_F,P_B\) | 高（论文实验） | 中短轨迹、稠密可达奖励 |

表注：「传播距离」指一次更新能把终端误差信息推多远；「方差」列的实验依据均为上述 tabular hypergrid 梯度实验，跨环境的普适排序没有定理保证。GTB 与 \(f\)-TB 的梯度支撑与 TB 相同（整轨迹），差别在残差的加权几何（§3.6–3.7）。

### 3.9.4 实践指导：什么任务选什么目标

- **长轨迹 + 稀疏奖励 → SubTB(\(\lambda\)) 起手**。论文实验：bit-sequence 全部长度/词表设置下 reward 相关性最高、发现模式更快；长度 237 的蛋白质任务显著超 TB（作者自己标注"优势随长度增长"是 speculation）；稀疏 hypergrid 见 3.9.2（[Madan et al. 2023](https://proceedings.mlr.press/v202/madan23a.html)）。有中间能量/部分奖励可用时，考虑 FL 型局部信用（[Pan et al. 2023](https://proceedings.mlr.press/v202/pan23c.html)，论文实验）。
- **中短轨迹、奖励非病态 → TB 默认**。TB 论文四域实验 + 结论句"default choice"（论文实验/原文）；后续文献普遍以 TB 为默认 baseline（社区经验）。
- **条件生成 / 每个条件一个 \(Z(y)\) → 考虑不学 \(Z\)**。条件 \(\log Z_\theta(y)\) 网络常成瓶颈：VAE 后验采样中 VarGrad 式组内估计超过学 \(Z\) 的 TB（[Sendera et al. 2024](https://arxiv.org/abs/2402.05098)，论文实验）；[GFlowRL](https://arxiv.org/abs/2607.13394) 在 LLM 上用 rollout-group 内 MC 估计替代 \(Z\) 网络是同一设计模式（2026 预印本）。
- **反例提醒：折中不是普适定律**。连续 diffusion sampler 基准上，FL-SubTB 相对 TB/VarGrad 没有一致优势（Sendera et al. 2024，论文实验）——SubTB 的收益依赖离散长轨迹+可泛化状态流的环境结构。
- **超参**：\(\log Z_\theta\) 用比策略高 1–2 个数量级的学习率（TB 论文附录 B.1 用 0.1 对 \(10^{-3}\)，论文设置）；\(\lambda\) 需按任务扫，梯度实验用 0.8（论文设置），常用范围 0.8–0.99（社区经验）。系统性控制重尾梯度可改 loss 形状：[Beyond Squared Error](https://proceedings.iclr.cc/paper_files/paper/2025/hash/353ec686503cd7020460d2829578ee4e-Abstract-Conference.html)（ICLR 2025）与 \(f\)-TB（ICML 2026）把"残差如何加权"从超参提升为 divergence 设计（论文提出）。
- **off-policy 与 replay**：不需要 IS 权重（论文证明），但 buffer 组成即 scalarization 权重——高奖励优先重放（[Shen et al. 2023](https://proceedings.mlr.press/v202/shen23a.html) PRT）与前缀信用/submodular replay（[RapTB](https://arxiv.org/abs/2603.00454)，ICML 2026）都可读作对该权重的干预（论文实验）。

### 3.9.5 与 §4 的衔接：梯度性质如何影响"损失小 ≠ 分布准"

§4 的链条是：采样期望 loss 小 \(\xrightarrow{(1)}\) 所有需要的轨迹上残差一致小 \(\xrightarrow{(2)}\) 终止分布 TV 界（§4.4）。梯度层面的性质分别攻击两个环节：

- **方差攻击 (1) 的"能否压下去"**：高梯度方差限制有效学习率与批量效率，属于 §4.5 的优化误差层；loss 曲线噪声大时，更不能把训练曲线当分布质量证书（对应 §4.4 "loss spike 不必然代表分布差"，反方向同样不成立）。
- **off-policy 覆盖攻击 (1) 的"在哪些轨迹上压"**：期望 loss 只约束训练分布支撑内的残差，正是 §4.1 支持条件的梯度层面读法。SubTB/DB 的自举还会把未见区域的 \(F_\theta\) 误差静默传导给已见边的目标值（本节推断）。
- **\(Z_\theta\) 的双重身份污染 loss 读数**：记 \(\tilde\delta(\tau)=\log\frac{P_F(\tau)}{R(x)P_B(\tau|x)}\)，则对任意训练分布 \(\pi\) 有代数恒等式 \(\mathbb E_\pi[\mathcal L_{\mathrm{TB}}]=\operatorname{Var}_\pi(\tilde\delta)+\big(\log Z_\theta+\mathbb E_\pi\tilde\delta\big)^2\)。第一项才是逐轨迹错配的散布；第二项在精确解处退化为 \((\log Z_\theta-\log Z)^2\)，一般情形混合"\(Z\) 校准误差"与"错配均值"。TB loss 下降可能主要是第二项在校准 \(Z_\theta\) 而策略分布未动（本节推断的读法），呼应 §3.3 "\(Z_\theta\) 只有拟合良好才可解释"与 §4.5 三种误差不可互换。
- **低方差的代价是认证力折扣**：DB/FM 的局部残差一致小到 \(c\) 时，全局 TV 界按轨迹长度退化（§4.4 的 \(1-\exp(-2Lc)\) 推断式；上引 2025 预印本的误差累积命题同向）。也就是说，低方差目标更容易"把 loss 压小"，但每单位 loss 对终止分布的约束力更弱——**方差与认证强度构成第二重折中**（本节推断）。

一句话总结：梯度方差决定"能把哪个 loss、在哪些轨迹上、压到多小"；§4 的定理决定"压小之后能保证什么"。TB 用优化噪声换认证强度，DB 用认证的长度折扣换优化平稳，SubTB(\(\lambda\)) 是在这两条汇率之间连续定价。

### 本节证据等级速查

| 论断 | 等级 | 出处 |
|---|---|---|
| on-policy TB 期望梯度 = KL 梯度；\(\log Z_\theta\) 是 baseline | 论文证明 | TB 附录 A.3；GFN-VI Prop 1；[Zimmermann et al. TMLR 2023](https://openreview.net/forum?id=AZ4GobeSLq) 独立给出 |
| 近优邻域 TB 估计比朴素 REINFORCE 低方差 | 论文证明（条件性） | TB 附录 A.3 |
| off-policy 无需 IS 权重、方差不随错配爆炸 | 论文证明+实验 | GFN-VI |
| DB<SubTB<TB 的梯度方差排序 | 论文实验（tabular grid） | Madan et al. 2023 |
| SubTB 长轨迹/稀疏奖励优势 | 论文实验 | Madan et al. 2023 |
| divergence 直训失败源于梯度方差；控制变量可证降方差 | 论文实验+证明 | da Silva et al. NeurIPS 2024 |
| 方差进入收敛率与样本复杂度 | 预印本（假设强） | Yu 2025 |
| 方差随轨迹长度/稀疏度的显式解析界 | **不存在**（截至 2026-08） | 本节检索结论 |

