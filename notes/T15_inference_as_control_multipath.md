# T15 · Discrete Probabilistic Inference as Control in Multi-path Environments

> 发表：UAI 2024 · 链接：https://proceedings.mlr.press/v244/deleu24a.html

## 一句话
该论文探讨了通过最大熵强化学习（MaxEnt RL）和生成流网络（GFlowNets）解决离散结构化样本空间中的概率推理问题，并证明了这些方法之间的等价关系。

## 问题与动机
在离散和高度结构化的样本空间中进行概率推理是一个挑战，因为传统的重参数化技巧变得难以应用。该论文提出将采样视为一个顺序决策问题，利用最大熵强化学习（MaxEnt RL）和生成流网络（GFlowNets）来解决这一问题。通过引入修正后的奖励函数，可以确保最优策略诱导的边缘分布与目标分布一致。

## 方法核心
### 最大熵强化学习（MaxEnt RL）
考虑一个有限时间步长的马尔科夫决策过程（MDP），定义如下：
- 状态空间 \( S \) 和动作空间 \( A \) 是离散且有限的。
- 转移函数 \( T: S \times A \rightarrow \bar{S} \) 是确定性的，其中 \(\bar{S} = S \cup \{s_f\}\)，\(s_f\) 是终止状态。
- 初始状态 \( s_0 \in S \)。
- 奖励函数 \( r(s, s') \) 定义为：
  \[
  r(s, s') = 
  \begin{cases}
  -E(s_T) & \text{if } s' = s_f \\
  \alpha \log PB(s | s') & \text{otherwise}
  \end{cases}
  \]
  其中 \( E(s_T) \) 是终止状态的能量函数，\( PB(s | s') \) 是从状态 \( s' \) 回到状态 \( s \) 的后向转移概率。

最优策略 \(\pi^*_{\text{MaxEnt}}\) 满足：
\[
\pi^*_{\text{MaxEnt}} = \arg \max_\pi \mathbb{E}_\tau \left[ \sum_{t=0}^{T} r(s_t, s_{t+1}) + \alpha H(\pi(\cdot | s_t)) \right]
\]

### 生成流网络（GFlowNets）
GFlowNets 寻找一个满足流量匹配条件的流函数 \( F(s \rightarrow s') \)：
\[
\sum_{s \in \text{Pa}(s')} F(s \rightarrow s') = \sum_{s'' \in \text{Ch}(s')} F(s' \rightarrow s'')
\]
边界条件为：
\[
F(x \rightarrow s_f) = \exp(-E(x)/\alpha)
\]
从流函数 \( F \) 可以定义一个策略：
\[
P_F(s_{t+1} | s_t) \propto F(s_t \rightarrow s_{t+1})
\]

### 等价关系
通过修正奖励函数，可以建立 MaxEnt RL 和 GFlowNets 之间的等价关系。例如，路径一致性学习（PCL）和子轨迹平衡（SubTB）算法之间的等价关系：
\[
L_{\text{PCL}}(\theta, \phi) = \frac{1}{2} \mathbb{E}_{\pi_b}[\Delta^2_{\text{PCL}}(\tau; \theta, \phi)]
\]
\[
L_{\text{SubTB}}(\theta, \phi) = \frac{1}{2} \mathbb{E}_{\pi_b}[\Delta^2_{\text{SubTB}}(\tau; \theta, \phi)]
\]
其中 \(\Delta_{\text{PCL}}\) 和 \(\Delta_{\text{SubTB}}\) 分别是残差项。

## 理论结果
该论文证明了通过修正奖励函数，MaxEnt RL 和 GFlowNets 之间的等价关系。具体而言，当使用修正后的奖励函数时，最优策略 \(\pi^*_{\text{MaxEnt}}\) 诱导的边缘分布与目标分布一致。

## 实验与证据
实验部分展示了多个算法在不同任务上的性能，包括生成进化树的任务。具体数据集和统计信息如表 2 所示。实验结果显示，不同算法在不同数据集上的表现具有较高的相关性。

## 局限与批判
1. **修正奖励函数的复杂性**：修正奖励函数需要引入后向转移概率 \( PB(s | s') \)，这增加了算法的复杂性和计算成本。
2. **理论假设的限制**：理论证明依赖于特定的假设条件，例如 MDP 结构为有向无环图（DAG），这限制了其在更一般情况下的适用性。
3. **实际应用中的性能差异**：尽管理论证明了等价关系，但在实际应用中，不同算法的表现可能会有所不同，需要进一步研究优化。
4. **计算资源需求**：修正奖励函数和流量匹配条件的实现需要较大的计算资源，这可能限制其在大规模问题中的应用。

## 与谁对话
- **T01**（GFlowNet 原始论文）：本文基于原始 GFlowNet 提出的方法，进一步将其应用于离散概率推理问题，展示了其在多路径环境中的有效性。
- **T14**（GFN 等价于熵正则 RL）：本文扩展了 T14 中关于 GFlowNets 与熵正则化强化学习的关系，具体应用于离散结构化样本空间中的概率推理任务。
- **T02**（GFlowNet Foundations）：本文继承了 GFlowNet 的统一数学框架，提出了新的方法来解决离散概率推理问题，进一步丰富了 GFlowNet 的理论基础。

## 对后续研究的启示
1. **改进修正奖励函数**：探索更高效的修正奖励函数，减少计算复杂性和资源需求。
2. **扩展理论假设**：研究在更一般 MDP 结构下，MaxEnt RL 和 GFlowNets 之间的等价关系。
3. **优化实际应用性能**：针对具体应用场景，优化算法性能，提高实际应用效果。