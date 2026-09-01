# N007 · Distributional GFlowNets with Quantile Flows

> 发表：TMLR 2024 · 链接：https://openreview.net/forum?id=vFSsRYGpjW

## 一句话
本文提出了一种新的基于量化的分布型GFlowNet训练算法，能够处理随机奖励函数，并在确定性基准测试中优于现有方法。

## 问题与动机
传统的GFlowNet框架只能从确定性的奖励函数学习，这在现实场景中过于严格。实际环境中往往存在随机性，需要建模不确定性。为此，本文提出采用概率方法来建模流函数，以应对这种随机性。

## 方法核心
### 分布型GFlowNet
本文提出将边缘流视为随机变量，并参数化其分位数函数。通过分位数回归训练GFlowNet模型，基于类似时间差分的流约束。具体步骤如下：
1. **定义边缘流和状态流**：
   - 边缘流 $F(s \rightarrow s')$ 是轨迹经过该边的概率。
   - 状态流 $F(s)$ 是所有包含状态 $s$ 的轨迹的概率总和。
2. **流匹配约束**：
   - 对于任意状态 $s'$，流入该状态的流量等于流出该状态的流量：
     $$
     \sum_{s:(s \rightarrow s') \in A} F(s \rightarrow s') = \sum_{s'':(s' \rightarrow s'') \in A} F(s' \rightarrow s'').
     $$
3. **参数化边缘流的分位数函数**：
   - 使用神经网络参数化边缘流的分位数函数 $Z^\text{log}_\beta(s \rightarrow s'; \theta)$。
4. **构建时间差分误差**：
   - 构建时间差分误差 $\delta_\beta, \tilde{\beta}(s'; \theta)$：
     $$
     \delta_\beta, \tilde{\beta}(s'; \theta) = \log \left( \sum_{(s' \rightarrow s'') \in A} \exp Z^\text{log}_{\tilde{\beta}}(s' \rightarrow s''; \theta) \right) - \log \left( \sum_{(s \rightarrow s') \in A} \exp Z^\text{log}_\beta(s \rightarrow s'; \theta) \right).
     $$
5. **分位数回归损失**：
   - 使用分位数回归最小化pinball误差 $\rho_\beta(\delta)$：
     $$
     L_\text{QM}(s; \theta) = \frac{1}{\tilde{N}} \sum_{i=1}^N \sum_{j=1}^{\tilde{N}} \rho_{\beta_i}(\delta_{\beta_i, \tilde{\beta_j}}(s; \theta)).
     $$
6. **推理阶段**：
   - 在生成阶段，前向策略通过数值积分估计：
     $$
     P_F(s' | s) \propto E[Z(s \rightarrow s')] \approx \frac{1}{N} \sum_{i=1}^N \exp(Z^\text{log}_{\beta_i}(s \rightarrow s'; \theta)).
     $$

### 风险敏感流
为了应对现实世界的不确定性，引入了扭曲风险测度的概念。具体来说，使用扭曲风险测度来计算期望值：
$$
E_g[Z] = \int_0^1 Q_Z(g(\beta)) d\beta.
$$
其中 $g(\beta)$ 是一个单调递增函数，用于调整风险敏感性。

## 理论结果
本文证明了提出的量化匹配算法能够有效地处理随机奖励函数，并且在确定性基准测试中优于现有的GFlowNet方法。此外，通过引入扭曲风险测度，可以生成风险敏感的策略。

## 实验与证据
实验部分包括两个任务：序列生成和分子合成。
1. **序列生成**：
   - 基准测试使用Levenshtein距离定义奖励函数。
   - 结果显示，量化匹配算法在最短时间内达到稳定状态。
2. **分子合成**：
   - 使用预测归一化负结合能作为奖励。
   - 结果表明，量化匹配算法在发现新分子方面表现优异。

## 局限与批判
1. **超参数选择**：
   - 超参数的选择（如 $N$ 和 $\tilde{N}$）对性能有较大影响，需要仔细调优。
2. **计算复杂度**：
   - 量化匹配算法涉及多次计算分位数函数，增加了计算复杂度。
3. **泛化能力**：
   - 尽管在确定性基准测试中表现良好，但在更复杂的随机环境中泛化能力有待验证。
4. **量化函数的选择**：
   - 不同的量化函数实现方式（显式 vs 隐式）可能会影响最终效果，需要进一步探讨。

## 与谁对话
- **T01**（GFlowNet 原始论文）：本文扩展了原始GFlowNet框架，使其能够处理随机奖励函数，而不仅仅是确定性的奖励函数。
- **T07**（Towards Understanding GFN Training）：本文解决了GFN训练过程中存在的loss与分布误差之间的鸿沟，提出了新的训练方法来优化这一过程。
- **T12**（连续 GFlowNet 理论）：本文借鉴了连续GFlowNet理论中的流守恒概念，将其应用于离散环境下的随机奖励函数建模。

## 对后续研究的启示
1. **超参数优化**：
   - 进一步研究超参数（如 $N$ 和 $\tilde{N}$）的选择方法，以提高算法的鲁棒性和效率。
2. **复杂环境下的泛化能力**：
   - 在更复杂的随机环境中进行实验，验证量化匹配算法的泛化能力。
3. **量化函数的选择**：
   - 探讨不同的量化函数实现方式（显式 vs 隐式），以找到最佳的实现方案。