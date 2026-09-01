# T17 · 用 f-散度直接训练 GFlowNet：问题不在散度而在梯度方差

> **On Divergence Measures for Training GFlowNets**
> 作者：Tiago da Silva, Eliezer de Souza da Silva, Diego Mesquita（Getulio Vargas Foundation, 巴西）· NeurIPS 2024 · [arXiv](https://arxiv.org/abs/2410.09355) · 代码：未开源（论文承诺接收后公开，截至 2026-09 未检索到官方仓库）

## 一句话

把 forward/reverse KL、Renyi-α、Tsallis-α 四类散度重新拉回 GFlowNet 训练目标的候选清单：此前「散度训练不如 TB」的结论（Malkin et al., ICLR 2023，即 T08）源于 REINFORCE 梯度估计的高方差与偏差，配上恰当的控制变量（control variates, CV）后，散度最小化经常比 Trajectory Balance（TB）收敛更快。

## 1. 要解决的问题

GFlowNet 与层次变分推断（hierarchical variational inference, HVI）在离散分布上等价（T08），照理说可以直接最小化 \(D(P_F \| P_B)\) 之类的散度来训练。但 T08 的实验结论是：直接最小化 reverse/forward KL 不如最小化 log-squared 差的 TB loss，尤其在稀疏奖励下。这留下两个悬而未决的问题：

1. 散度训练失败是散度目标本身的缺陷，还是其随机梯度估计器的缺陷？
2. GFlowNet–HVI 等价性此前只在有限支撑（离散）分布上成立，连续与混合空间上是否仍然成立？

本文对两个问题都给出回答：失败根源是梯度方差（以及 T08 使用的带偏 baseline 和 importance-weighted 聚合引入的偏差）；等价性可以推广到任意可测拓扑空间。

## 2. 核心方法

**背景符号。** 沿用 Lahlou et al.（T12）的 measurable pointed DAG 框架：状态空间 \((\mathcal{S}, \mathcal{T})\) 为拓扑空间，配前向/后向转移核 \(\kappa_f, \kappa_b\)、初始态 \(s_o\)、终结态 \(s_f\)、参考测度 \(\nu\)。\(p_{F_\theta}\) 是前向策略 \(P_F\) 相对 \(\kappa_f\) 的密度（神经网络参数 \(\theta\)），\(p_B\) 是后向策略密度（通常固定），\(r = dR/d\mu\) 是目标奖励密度，\(Z\) 是配分函数。轨迹 \(\tau = (s_o, s_1, \dots, s_n, s_f)\)，其终态记 \(x\)。目标分布在轨迹空间上定义为 \(\bar p_B(\tau) = \frac{r(x)}{Z}\, p_B(\tau \mid x)\)。

**四类散度目标（论文 Definition 5、6）。** 训练即求 \(\theta^* = \arg\min_\theta D(P_F, P_B)\)：

- Renyi-α：\(\;R_\alpha(P_F \| P_B) = \frac{1}{\alpha-1} \log \int p_{F_\theta}(\tau\mid s_o)^\alpha\, \bar p_B(\tau)^{1-\alpha}\, \kappa_f(s_o, d\tau)\)
- Tsallis-α：\(\;T_\alpha(P_F \| P_B) = \frac{1}{\alpha-1} \left[ \int p_{F_\theta}(\tau\mid s_o)^\alpha\, \bar p_B(\tau)^{1-\alpha}\, \kappa_f(s_o, d\tau) - 1 \right]\)
- reverse KL：\(\;D_{KL}[P_F \| P_B] = \mathbb{E}_{\tau \sim P_F}\!\left[\log \frac{p_{F_\theta}(\tau|s_o)}{\bar p_B(\tau)}\right]\)
- forward KL：\(\;D_{KL}[P_B \| P_F] = \mathbb{E}_{\tau \sim P_B}\!\left[\log \frac{\bar p_B(\tau)}{p_{F_\theta}(\tau|s_o)}\right]\)（因 \(P_B\) 不可采样，用 \(P_F\) 做重要性采样）

α 从 \(-\infty\) 到 \(\infty\) 对应从 mass-covering 到 mode-seeking 的连续谱，为控制探索—利用提供旋钮；实验统一固定 α = 0.5。

**梯度估计（Lemma 1、2）。** 记 \(g(\tau,\theta) = \left(\frac{p_B(\tau|x)\, r(x)}{p_{F_\theta}(\tau|s_o)}\right)^{1-\alpha}\)，用 REINFORCE（score function）估计器：

\[\nabla_\theta R_\alpha = \frac{\mathbb{E}\left[\nabla_\theta g(\tau,\theta) + g(\tau,\theta)\, \nabla_\theta \log p_{F_\theta}(\tau|s_o)\right]}{(\alpha-1)\, \mathbb{E}[g(\tau,\theta)]}, \qquad \nabla_\theta T_\alpha \overset{C}{=} \frac{\mathbb{E}\left[\nabla_\theta g + g\, \nabla_\theta \log p_{F_\theta}\right]}{\alpha-1}\]

期望在 \(P_F\) 下计算，\(\overset{C}{=}\) 表示相差正常数倍（Adam/RMSProp 等对常数缩放不变）。关键性质：所有梯度都不含 \(Z\)，绕过了 TB loss 必须联合学习 \(\log Z_\theta\) 的负担。

**控制变量（第 4 节，本文的技术核心）。** REINFORCE 估计器方差大，本文叠加两层无偏方差削减：

1. **score function 作 CV + 近似最优 baseline（Proposition 2）**：利用 \(\mathbb{E}_{P_F}[\nabla_\theta \log p_{F_\theta}] = 0\)，把 \(\nabla_\theta \log p_{F_\theta}\) 作为零均值控制变量；最优标量 baseline 为 \(a^* = \frac{\mathbb{E}[g(\tau)^T (f(\tau) - \mathbb{E} f)]}{\mathbb{E}[g(\tau)^T g(\tau)]}\)。因 \(a^*\) 对样本级梯度非线性、无法用 vector-Jacobian product 高效计算，改用 delta method 的线性化批估计（论文式 (6)）。
2. **REINFORCE leave-one-out（RLOO）**：对 \(\mathbb{E}[f(\tau) \nabla_\theta \log p_{F_\theta}(\tau)]\)，用第 \(i\) 条轨迹之外样本的均值 \(a(\tau_i) = \frac{1}{N-1}\sum_{n \neq i} f(\tau_n)\) 做逐样本 baseline，无偏（依赖样本独立性），且可写成一次 stop-gradient 内积（论文式 (7)），自动微分框架下几乎零开销。

**与 T08 做法的差别。** Malkin et al. 用批均值 baseline 加 importance-weighted 聚合修正 off-policy 采样，这两步都引入偏差、放弃了优化保证；本文认为这正是「TB 优于散度」这一结论的来源。

## 3. 理论结果

**Proposition 1（TB–KL 梯度等价，任意拓扑空间）。** 设 \(\mathcal{L}_{TB}(\tau;\theta) = \left(\log \frac{Z\, p_{F_\theta}(\tau|s_o)}{r(x)\, p_B(\tau|x)}\right)^2\)，则

\[\nabla_\theta\, \mathbb{E}_{\tau \sim P_F(s_o,\cdot)}[\mathcal{L}_{TB}(\tau;\theta)] = 2\, \nabla_\theta D_{KL}[P_F \| P_B].\]

成立条件：measurable pointed DAG 的五条公理（终结性、可达性、一致性、连续性、有限吸收），on-policy 采样，\(Z\) 视为已知常数参与 \(\mathcal{L}_{TB}\)（实际训练中 \(Z_\theta\) 是学出来的，此处是理想化对齐）。该命题把 T08 的离散结果推广到连续与混合空间，意味着 on-policy TB 与 reverse KL 在期望梯度意义下同速，而 KL 不需要估计 \(Z\)。

**Lemma 1 / Lemma 2**：如上，给出四类散度的 REINFORCE 型无偏梯度表达式（forward KL 的重要性加权版本无偏至正常数倍）。

**Proposition 2**：向量值控制变量下最小化协方差矩阵迹的最优 baseline 闭式解。

论文未给出散度训练的收敛速率或有限样本保证；「provably correct」指的是全局最优点与目标分布一致（散度为零当且仅当 \(P_F = P_B\)），不是优化过程的收敛定理。

## 4. 实验与证据

五个基准：set generation（|D|=32, N=16）、自回归序列生成（|D|=8, N=6）、Bayesian phylogenetic inference（BPI，7 物种、JC69 突变模型）、9 分量高斯混合（连续）、banana 分布（连续，HMC 采样作 ground truth）。离散任务用 \(L_1\) 距离评估，连续任务用 Jensen-Shannon 散度。3 个随机种子。

- **收敛速度（Figure 3）**：散度目标经常快于 TB；没有单一最优散度，与 T08「TB 全面占优」的结论相反。
- **最终精度（Table 1，\(L_1\)/JSD）**：Sets 上 TB 0.07±0.00 vs 全部散度 0.03±0.00；Sequences 上 TB 0.28±0.06 vs reverse KL 0.16±0.06；GMs 上 forward KL 0.09±0.10 vs TB 0.31±0.08；BPI 上各目标无统计显著差异（均约 0.21–0.22）。
- **banana 分布（Figure 4）**：Tsallis-α、Renyi-α、forward KL 学出的分布优于 TB 和 reverse KL；后两者行为相似——正是 Proposition 1 的预测。
- **CV 消融（Figure 2、5）**：CV 使梯度协方差迹下降若干个数量级，且直接转化为训练加速与稳定；无 CV 时 reverse KL 几乎不可用。
- **forward KL 补充实验（Figure 7）**：其欠佳表现来自重要性采样方差，批量加大到 1024 后差距消失。

证据边界：任务规模都偏小（最大动作空间 32），没有分子生成、LLM 等大规模任务；off-policy 场景未测试（散度的 REINFORCE 估计天然要求 on-policy 或重要性加权）。

## 5. 在 GFlowNet 版图中的位置

- **直接对话 T08（GFlowNets and Variational Inference）**：T08 建立离散等价性但实验否定散度训练；本文把等价性推广到任意拓扑空间（依托 T12 的连续理论），并用 CV 翻案实验结论。T09（variational perspective, Zimmermann et al.）是另一条被本文引用与扩展的 VI 路线。
- **与 T03（TB）的关系**：不推翻 TB，而是指出 on-policy TB ≈ reverse KL（Proposition 1），因此 TB 的低方差优势（不含 score function 项，见论文 Figure 6）与 KL 的免 \(Z\) 优势各有适用面。
- **被 T49（f-Trajectory Balance, ICML 2026）系统化**：T49 建立 translation-invariant 轨迹损失与 f-散度的一一对应，把本文「逐个散度设计 REINFORCE 估计器 + CV」的做法升级为「直接构造平方型代理损失，其 on-policy 梯度自动等于 f-散度梯度且天然支持 off-policy」，可视为对本文路线的收编与超越。
- **与 T34（Beyond Squared Error）互补**：T34 改回归损失的形状，本文改目标本身；两者共同构成 2024–2025 年「loss 设计」分线。
- **方差削减谱系**：与 VarGrad（Richter et al. 2020，GFlowNet 中广泛用作免 \(Z\) 的 TB 变体）、N029（log-variance loss）同属一条「低方差梯度估计」线；本文的 RLOO 技术后来也出现在 LLM RLHF 的 GFlowNet 式微调中。
- **仓库内上下文**：`surveys/GFLOWNET_GRADIENT_VARIANCE_CN.md` 把本文作为方差问题的关键证据链一环。

## 6. 局限与批判

1. **α 只测了 0.5**：论文自己承认不同任务可能有更优 α，但没有给出选择准则；mode-seeking/mass-covering 旋钮的实用价值停留在示意图层面（Figure 1）。
2. **on-policy 局限**：所有散度梯度都在 \(P_F\) 下取期望，与 GFlowNet 社区赖以对抗 mode collapse 的 off-policy 训练（replay buffer、local search、tempering）不兼容；forward KL 的重要性加权在提议分布与目标差距大时方差爆炸（论文 Figure 7 自证）。这是 T49 后来主攻的缺口。
3. **评估任务偏小且偏「密集奖励」**：T08 声称散度在稀疏奖励下失效，本文没有在真正稀疏奖励的大空间任务（如分子生成）上正面回应，两边结论的适用域交叠不清。
4. **Proposition 1 的理想化**：等价性把 \(Z\) 当作 oracle 常数；实际 TB 训练中 \(\log Z_\theta\) 与策略参数联合优化，动力学并不等同 reverse KL。
5. **代码未开源**，CV 实现细节（式 (6) 的 ε、baseline 更新时机）只能靠正文复现。

## 7. 对后续研究的启示

1. **「目标 vs 估计器」要分开归因**：一个训练目标被实验否定时，先检查梯度估计的方差与偏差，再下结论。这一方法论教训在 GFlowNet 文献中被反复引用。
2. **散度选择是探索—利用旋钮**：α 参数为按任务调节 mode-seeking 程度提供了原则化接口，直接启发 T49 的 α 家族与 T46 的 α-GFN。
3. **免 \(Z\) 训练路线**：KL/α 散度梯度不依赖配分函数，对 \(Z\) 难学的任务（长轨迹、宽奖励动态范围）是 TB 的现实替代。
4. **CV 是低垂果实**：RLOO 与 score-function CV 几乎零成本，任何用 REINFORCE 估计梯度的 GFlowNet 变体（含 LLM 微调）都应默认配备。
5. **开放组合**：论文结尾指出可将 χ-散度、Ruiz–Titsias 型 MCMC-VI 混合散度、IWAE 式目标引入 GFlowNet——其中 f-散度族的系统化已由 T49 完成，其余仍是空位。
