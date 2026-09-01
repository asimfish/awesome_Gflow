# N032 · Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models with Memoryless Stochastic Optimal Control

> 发表：ICLR 2025 Spotlight · 链接：https://openreview.net/forum?id=xQBRrtQM8u

## 一句话
本文提出了一种新的算法——Adjoint Matching，用于通过无记忆随机最优控制（Memoryless Stochastic Optimal Control）来微调流匹配和去噪扩散模型，以实现更好的文本到样本一致性，并保持良好的多样性。

## 问题与动机
流匹配和去噪扩散模型在生成建模中广泛应用，但这些模型通常需要改进以提高样本质量。现有方法通常忽略了基础模型的分布，而专注于奖励模型。这导致了生成模型在微调时出现偏差，无法生成期望的倾斜分布。因此，本文旨在开发一种简单且理论上有效的微调方法，以解决这一问题。

## 方法核心
本文提出了一个基于随机最优控制（SOC）的微调框架，并证明了一个特定的无记忆噪声调度必须在微调过程中使用，以消除噪声变量与生成样本之间的依赖关系。具体步骤如下：

1. **定义参考流**：
   - 给定初始分布 $\bar{X}_0 \sim p_0 = \mathcal{N}(0, I)$ 和数据分布 $\bar{X}_1$，定义参考流 $\bar{X} = (\bar{X}_t)_{t \in [0,1]}$：
     \[
     \bar{X}_t = \beta_t \bar{X}_0 + \alpha_t \bar{X}_1,
     \]
     其中 $(\alpha_t)_{t \in [0,1]}$ 和 $(\beta_t)_{t \in [0,1]}$ 是函数，满足 $\alpha_0 = \beta_1 = 0$ 和 $\alpha_1 = \beta_0 = 1$。

2. **流匹配模型**：
   - 流匹配模型的生成马尔可夫过程是一个常微分方程（ODE）：
     \[
     dX_t = v(X_t, t) dt, \quad X_0 \sim \mathcal{N}(0, I),
     \]
     其中 $v(X_t, t)$ 是优化参数，以匹配参考流的导数。

3. **去噪扩散模型**：
   - 去噪扩散模型的采样方案可以表示为：
     \[
     dX_t = \left(\frac{\dot{\bar{\alpha}}_t}{2\bar{\alpha}_t} X_t - \frac{\dot{\bar{\alpha}}_t}{2\bar{\alpha}_t} + \frac{\sigma(t)^2}{2}\right) \epsilon_{\text{base}}(X_t, t) \sqrt{1-\bar{\alpha}_t} dt + \sigma(t) dB_t, \quad X_0 \sim \mathcal{N}(0, I).
     \]

4. **统一框架**：
   - 将流匹配和去噪扩散模型统一为一个通用框架：
     \[
     dX_t = b(X_t, t) dt + \sigma(t) dB_t, \quad X_0 \sim \mathcal{N}(0, I),
     \]
     其中 $b(x, t) = \kappa_t x + \left(\frac{\sigma(t)^2}{2} + \eta_t\right) s(x, t)$，$\kappa_t = \frac{\dot{\alpha}_t}{\alpha_t}$，$\eta_t = \beta_t \left(\frac{\dot{\alpha}_t}{\alpha_t \beta_t} - \dot{\beta}_t\right)$。

5. **随机最优控制问题**：
   - 定义随机最优控制问题：
     \[
     \min_{u \in U} \mathbb{E} \left[\int_0^1 \left(\frac{1}{2} \|u(X_t, t)\|^2 + f(X_t, t)\right) dt + g(X_1)\right],
     \]
     其中 $dX_t = \left(b(X_t, t) + \sigma(t) u(X_t, t)\right) dt + \sigma(t) dB_t$，$X_0 \sim p_0$。

6. **价值函数偏移问题**：
   - 证明了初始价值函数偏移问题，即直接添加KL正则化会导致偏移分布而不是期望的倾斜分布。

7. **Adjoint Matching算法**：
   - 提出Adjoint Matching算法，通过最小化以下损失函数来解决上述问题：
     \[
     \hat{L}_{\text{Adj-Match}}(\theta) = \sum_{t \in K} \min\left(L_{\text{CT}}, \left\|\frac{2}{\sigma(t)} \left(v_{\text{finetune}}^\theta(X_t, t) - v_{\text{base}}(X_t, t)\right) + \sigma(t) \tilde{a}_t\right\|^2\right),
     \]
     其中 $L_{\text{CT}}$ 是损失剪辑阈值，$K$ 是随机时间步子集。

## 理论结果
本文证明了无记忆噪声调度在微调过程中能够消除噪声变量与生成样本之间的依赖关系，从而确保生成模型收敛到期望的倾斜分布。此外，Adjoint Matching算法通过将SOC问题转化为回归问题，提高了算法的效率和稳定性。

## 实验与证据
本文进行了广泛的实验对比，包括现实性、一致性和多样性等多个方面。实验结果显示，Adjoint Matching方法在未见过的人类偏好奖励模型上具有较好的泛化能力，同时保持了良好的文本到样本一致性以及多样性。具体实验环境和基线包括：

- 使用40个时间步进行微调和推理。
- 在多个时间步（10, 20, 40, 100, 200）下进行实验，结果显示在100和200时间步下的指标与40时间步相似，但在10和20时间步下表现较差。

## 局限与批判
1. **计算成本**：Adjoint Matching算法的计算成本较高，尤其是在大规模数据集上。虽然其计算时间与离散伴随损失相似，但连续伴随损失的计算时间更长。
2. **超参数选择**：Adjoint Matching算法中的超参数（如损失剪辑阈值$L_{\text{CT}}$）需要仔细调整，否则可能导致性能下降。
3. **泛化能力**：尽管Adjoint Matching在未见过的人类偏好奖励模型上表现出较好的泛化能力，但在不同任务上的泛化能力仍需进一步验证。
4. **多样性保持**：虽然Adjoint Matching在保持多样性方面表现出色，但在某些情况下仍可能出现多样性不足的问题。

## 与谁对话
- **T02**（GFlowNet Foundations）：本文提出的Adjoint Matching方法在理论基础上扩展了GFlowNet的统一数学框架，提供了更精细的微调策略。
- **N010**（改进扩散采样器的off-policy训练）：本文通过引入无记忆随机最优控制来解决扩散模型的微调问题，是对N010中off-policy训练方法的一种补充和提升。
- **O01**（最优传输讲义）：本文利用无记忆随机最优控制的概念，与最优传输理论中的熵正则化OT和Schrödinger桥建立了联系，进一步丰富了流匹配和扩散模型的理论基础。

## 对后续研究的启示
1. **超参数优化**：进一步研究如何自动优化Adjoint Matching算法中的超参数，以减少人工干预。
2. **泛化能力提升**：探索如何进一步提升Adjoint Matching在不同任务上的泛化能力，特别是在未见过的数据集上。
3. **多样性增强**：研究如何在保持文本到样本一致性的同时进一步增强生成样本的多样性。