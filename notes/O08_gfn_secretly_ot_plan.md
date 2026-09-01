# O08 · Your GFlowNet Secretly Learns an Optimal Transport Plan

> 发表：ICML 2026 SPIGM Workshop · 链接：https://arxiv.org/abs/2606.06272

## 一句话
本文揭示了非无环GFlowNet和最优传输之间的联系，并证明了在固定初始流分布的情况下，最小流目标可以转化为Kantorovich最优传输问题。

## 问题与动机
生成流网络（GFlowNet）是一种通过随机轨迹在有向图中采样结构化对象的方法。尽管最初设计用于无环图环境，但后来的研究将其扩展到了非无环图环境。本文旨在建立非无环GFlowNet和最优传输（OT）之间的理论联系，从而将GFlowNet框架应用于大规模图上的OT问题。

## 方法核心
### 最小流GFlowNet
考虑一个非无环GFlowNet环境图 \( G = (S, E) \)，其中 \( S \) 是有限状态空间，\( E \subseteq S \times S \) 是有限边集。定义状态流 \( F(s) \) 和边流 \( F(s \rightarrow s') \) 如下：
\[ F(s) = Z \cdot E_{\tau \sim P}\left[\sum_{t=0}^{n_\tau + 1} I\{s_t = s\}\right], \]
\[ F(s \rightarrow s') = Z \cdot E_{\tau \sim P}\left[\sum_{t=0}^{n_\tau} I\{s_t = s, s_{t+1} = s'\}\right]. \]

这些流函数满足以下条件：
\[ \sum_{s' \in out(s)} F(s \rightarrow s') = \sum_{s'' \in in(s)} F(s'' \rightarrow s), \]
\[ F(s_0) = F(s_f) = Z. \]

### 最小流优化问题
最小流GFlowNet的目标是找到使得总流最小化的策略。定义内部状态集合 \( I := S \setminus \{s_0, s_f\} \)，则最小流优化问题可以表示为：
\[ \min_{F, P_F, P_B} \sum_{s \in I} F(s) \]
约束条件包括：
\[ F(s)P_F(s' | s) = F(s')P_B(s | s'), \quad \forall (s, s') \in E, \]
\[ F(s_f)P_B(x | s_f) = R(x), \quad x \in X. \]

### 等价于Kantorovich OT问题
当固定初始边流分布时，最小流目标可以转化为Kantorovich OT问题。定义成本函数 \( d(u, x) \) 为从 \( u \) 到 \( x \) 的最短路径长度，则Kantorovich OT问题可以表示为：
\[ \min_{\Pi \geq 0} \sum_{i,j} d(u_i, x_j) \Pi_{i,j} \]
约束条件包括：
\[ \sum_i \Pi_{i,j} = R(x_j), \]
\[ \sum_j \Pi_{i,j} = L(u_i). \]

### 等价性证明
定理3.2证明了在特定假设下，最小流GFlowNet问题等价于Kantorovich OT问题。具体来说，如果 \( P^* \) 是最小流GFlowNet问题的解，则它诱导了一个最优耦合 \( \Pi^*_{u,x} \)。

## 理论结果
本文证明了最小流GFlowNet问题在固定初始流分布的情况下等价于Kantorovich OT问题。这意味着GFlowNet可以通过学习最优路径来实现最优传输计划。

## 实验与证据
实验部分展示了GFlowNet在不同分布下的性能。具体环境包括月形分布、角形分布和球形分布。实验结果显示，GFlowNet能够准确地恢复精确的OT解决方案，并且随着组合空间的增长，能够有效地近似解决方案。

## 局限与批判
1. **理论假设限制**：本文的理论结果依赖于特定的假设条件，例如存在从初始状态到终端状态的有限长度路径。这些假设在实际应用中可能无法完全满足。
2. **计算复杂度**：虽然GFlowNet提供了一种有效的近似方法，但在大规模图上的计算复杂度仍然较高，特别是在需要处理大量状态和边的情况下。
3. **初始化影响**：初始流分布的选择对最终结果有很大影响。如何选择合适的初始流分布是一个开放的问题。
4. **泛化能力**：虽然GFlowNet在特定任务上表现良好，但其泛化能力仍有待进一步验证，尤其是在面对未见过的数据分布时。

## 与谁对话
1. T01：GFlowNet原始论文(NeurIPS2021)
2. T02：GFlowNet Foundations(JMLR2023)
3. T05：SubTB
4. T14：GFN=熵正则RL
5. T17：散度训练
6. T19：非无环理论
7. T32：何时学对分布

## 对后续研究的启示
1. **探索更广泛的假设条件**：未来研究可以尝试放宽本文中的假设条件，以使其适用于更广泛的应用场景。
2. **开发高效的初始化策略**：研究如何选择合适的初始流分布，以提高GFlowNet的学习效率和准确性。
3. **评估泛化能力**：进一步研究GFlowNet在未见过数据分布上的表现，以验证其泛化能力。