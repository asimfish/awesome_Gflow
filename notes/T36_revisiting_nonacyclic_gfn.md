# T36 · Revisiting Non-Acyclic GFlowNets in Discrete Environments

> 发表：ICML 2025 · 链接：https://proceedings.mlr.press/v267/morozov25a.html

## 一句话
本文重新审视了在离散环境中非无环GFlowNets的理论，并提供了一种更简单的方法来构建其理论框架，同时探讨了固定反向策略下的损失稳定性以及状态流正则化的重要性。

## 问题与动机
GFlowNets是一种生成模型，通过在适当构造的有向无环图环境中采样轨迹来学习从给定概率分布中采样对象。然而，这种设置的一个主要限制是需要图的无环性。本文旨在放松这一假设，探讨在包含循环的离散环境中如何应用GFlowNets，并提供新的理论和方法论见解。

## 方法核心
### 环境定义
- **状态空间** $S$ 和边集 $E \subseteq S \times S$。
- 特殊初始状态 $s_0$ 和特殊终止状态 $s_f$。
- 轨迹 $\tau = (s_0 \rightarrow s_1 \rightarrow \ldots \rightarrow s_{n_\tau} \rightarrow s_f)$，其中 $n_\tau$ 是轨迹长度。

### 向前和向后策略
- **向前策略** $P_F(s' | s)$ 定义为从状态 $s$ 到状态 $s'$ 的转移概率。
- **向后策略** $P_B(s | s')$ 定义为从状态 $s'$ 回到状态 $s$ 的转移概率。

### 状态和边流
- **状态流** $F(s)$ 定义为状态 $s$ 的期望访问次数。
- **边流** $F(s \rightarrow s')$ 定义为从状态 $s$ 到状态 $s'$ 的期望访问次数。

#### 公式推导
1. **状态流**：
   \[
   F(s) = F(s_f) \cdot \mathbb{E}_{\tau \sim P(\tau)} \left[ \sum_{t=0}^{n_\tau+1} I\{s_t = s\} \right]
   \]
   其中 $F(s_f)$ 是最终状态的流值，$\mathbb{E}$ 表示期望值。

2. **边流**：
   \[
   F(s \rightarrow s') = F(s_f) \cdot \mathbb{E}_{\tau \sim P(\tau)} \left[ \sum_{t=0}^{n_\tau} I\{s_t = s, s_{t+1} = s'\} \right]
   \]

3. **详细平衡条件**：
   \[
   F(s \rightarrow s') = F(s) P_F(s' | s) = F(s') P_B(s | s')
   \]

4. **流匹配条件**：
   \[
   F(s) = \sum_{s' \in \text{out}(s)} F(s \rightarrow s') = \sum_{s'' \in \text{in}(s)} F(s'' \rightarrow s)
   \]

### 损失函数
- **Detailed Balance Loss**：
   \[
   L_{DB}(s \rightarrow s') = \left( \log \frac{F_\theta(s) P_F(s' | s, \theta)}{F_\theta(s') P_B(s | s', \theta)} \right)^2
   \]

- **Stable Detailed Balance Loss**：
   \[
   L_{SDB}(s \rightarrow s') = \log \left( 1 + \epsilon \Delta^2(s, s', \theta) \right) (1 + \eta F_\theta(s))
   \]
   其中 $\Delta(s, s', \theta) = F_\theta(s) P_F(s' | s, \theta) - F_\theta(s') P_B(s | s', \theta)$。

### 状态流正则化
- 引入状态流正则化项以控制总流：
   \[
   \lambda \sum_{s \in S \setminus \{s_0, s_f\}} F(s)
   \]

## 理论结果
1. 提供了一个简单的理论框架来构建非无环GFlowNets。
2. 当反向策略固定时，损失稳定性不影响优化结果。
3. 当反向策略也进行训练时，最小化预期轨迹长度等价于最小化总流。
4. 提出状态流正则化作为解决优化问题的一种方法。
5. 证明了在非无环环境下，GFlowNets与熵正则化强化学习之间的等价关系。

## 实验与证据
### 环境与基线
- **Hypergrid 20x20x20x20**：一个四维超网格环境。
- **Permutations of length 4**：长度为4的排列环境。

### 结果
- **L1误差**：衡量经验分布与目标分布之间的差异。
- **平均轨迹长度**：衡量采样的轨迹长度。

具体数值：
- 在Hypergrid环境中，使用Detailed Balance Loss和Stable Detailed Balance Loss，随着训练轨迹数量增加，L1误差逐渐减小。
- 使用不同强度的状态流正则化参数 $\lambda$，较大的 $\lambda$ 值会导致较小的平均轨迹长度，但过大的 $\lambda$ 会使得到的向前策略显著偏置。

## 局限与批判
1. **理论复杂性**：虽然本文简化了非无环GFlowNets的理论框架，但仍存在一定的复杂性，特别是在处理状态流和边流时。
2. **损失稳定性**：尽管引入了稳定损失，但在某些情况下，不稳定损失仍然可能导致训练过程中的问题。
3. **状态流正则化**：状态流正则化的引入有助于控制总流，但选择合适的正则化参数 $\lambda$ 仍是一个挑战。
4. **实验环境有限**：本文的实验主要集中在一些特定的离散环境中，对于更复杂的实际应用场景，还需要进一步验证。

## 与谁对话
- T01：原始论文讨论了GFlowNets的基本概念和理论基础。
- T09：从变分视角讨论了GFlowNets与强化学习的关系。
- T14：证明了GFlowNets与熵正则化强化学习之间的等价关系。
- T19：讨论了非无环GFlowNets的理论基础。
- T32：探讨了GFlowNets何时能够学习正确的分布。

## 对后续研究的启示
1. **扩展到连续环境**：将本文提出的理论和方法扩展到连续环境中，以应对更广泛的应用场景。
2. **改进损失函数**：探索更多类型的损失函数，以提高训练的稳定性和效率。
3. **自动选择正则化参数**：开发一种自动选择状态流正则化参数 $\lambda$ 的方法，以减少人为干预的需求。