# N041 · Unsupervised Learning for Optimal Transport plan prediction between unbalanced graphs (ULOT)

> 发表：NeurIPS 2025 主会 · arXiv:2506.12025 · 链接：https://papers.nips.cc/paper_files/paper/2025/hash/873fd89b3e4db1f6242c2333673e104d-Abstract-Conference.html

## 一句话
本文提出了一种新的深度学习方法——无监督学习最优传输计划预测（ULOT），用于预测两个图之间的最优传输计划，特别是在不平衡图的情况下。该方法通过最小化融合不平衡Gromov-Wasserstein（FUGW）损失来训练神经网络，从而实现快速且准确的传输计划预测。

## 问题与动机
在许多图数据应用中，如对象检测、图编辑距离计算和形状匹配等，需要将两个图中的节点进行对齐或匹配。然而，当图是不平衡的，即节点数量不同或某些节点具有噪声特征时，这个问题变得更加困难。传统的最优传输（OT）方法虽然强大，但计算复杂度高，难以应用于大规模图。因此，本文提出了一种基于深度学习的方法，旨在加速最优传输计划的预测，并适应不平衡图的情况。

## 方法核心
### FUGW损失函数
考虑两个图 \( G_k = (F_k, D_k, \omega_k) \)（\( k \in \{1, 2\} \)），分别有 \( n_1 \) 和 \( n_2 \) 个节点。对于 \( k \in \{1, 2\} \)，它们由节点特征 \( F_k \in \mathbb{R}^{n_k \times d} \)、连接矩阵 \( D_k \in \mathbb{R}^{n_k \times n_k} \) 和节点权重 \( \omega_k \in \Delta_{n_k} \) 描述。FUGW损失函数定义如下：
\[ L_{\alpha, \rho}(G_1, G_2, P) = (1 - \alpha) \sum_{i=1}^{n_1} \sum_{j=1}^{n_2} \| (F_1)_i - (F_2)_j \|_2^2 P_{i,j} \]
\[ + \alpha \sum_{i=1}^{n_1} \sum_{j=1}^{n_2} \sum_{k=1}^{n_1} \sum_{l=1}^{n_2} | (D_1)_{i,k} - (D_2)_{j,l} |^2 P_{i,j} P_{k,l} \]
\[ + \rho \left( \text{KL}(P_{\#1} \otimes P_{\#1} \| \omega_1 \otimes \omega_1) + \text{KL}(P_{\#2} \otimes P_{\#2} \| \omega_2 \otimes \omega_2) \right) \]

其中，\( \alpha \in [0, 1] \) 是权衡参数，\( \rho \) 是边际惩罚项的权重。

### ULOT优化问题及架构
ULOT的目标是训练一个模型 \( P_{\theta}^{\rho, \alpha}(G_1, G_2) \)，给定两个图 \( G_1 \) 和 \( G_2 \) 以及参数 \( (\rho, \alpha) \)，能够预测一个FUGW传输计划。模型通过最小化期望FUGW损失进行训练：
\[ \min_{\theta} \mathbb{E}_{G_1, G_2 \sim D^2, \alpha, \rho \sim P} \left[ L_{\alpha, \rho}(G_1, G_2, P_{\theta}^{\rho, \alpha}(G_1, G_2)) \right] \]

其中，\( P \) 是参数 \( (\rho, \alpha) \) 的分布，\( P_{\rho} \) 是在 \( 10^{-7} \) 到 1 之间的对数均匀分布，\( P_{\alpha} \) 是Beta分布 \( \text{Beta}(0.5, 0.5) \)。

### 网络架构
ULOT架构包括两部分：
1. **节点嵌入层**：重复 \( N \) 次，包含跨图注意力和自节点更新（GCN）。
2. **最终传输计划预测层**：从学习到的节点特征和交互中预测传输计划。

#### 节点嵌入与跨注意力
- **自路径**：使用GCN处理每个图的独立节点特征。
\[ F_{\text{self}}^k = \text{GCN}(F_k) \]

- **跨路径**：计算节点特征之间的相似矩阵，并学习新的特征以描述其交互。
\[ F_{\text{cross}}^k = \text{MLP}(F_k, \rho, \hat{\alpha}) \]
\[ S_{i,j} = s((F_{\text{cross}}^1)_i, (F_{\text{cross}}^2)_j) \]
\[ S_1 = \text{softmax}_{\text{row}}(a_2 S) \]
\[ S_2 = \text{softmax}_{\text{column}}(a_2 S) \]
\[ F_{\text{cross}}^{1 \rightarrow 2} = \text{Linear}(F_{\text{cross}}^2 - S_2^T F_{\text{cross}}^1) \]
\[ F_{\text{cross}}^{2 \rightarrow 1} = \text{Linear}(F_{\text{cross}}^1 - S_1 F_{\text{cross}}^2) \]

- **合并路径**：
\[ F_{\text{final}}^k = \text{MLP}(\text{Linear}(F_k), \text{Linear}(F_{\text{self}}^k), \text{Linear}(F_{\text{match}}^{k' \rightarrow k}), \rho, \hat{\alpha}) \]

#### 传输计划预测
- **节点权重预测**：
\[ v_1 = \text{sigmoid}(\text{Linear}(F_{\text{final}}^1, \rho, \hat{\alpha})) \]
\[ v_2 = \text{sigmoid}(\text{Linear}(F_{\text{final}}^2, \rho, \hat{\alpha})) \]

- **传输计划预测**：
\[ P_{\theta}^{\rho, \alpha}(G_1, G_2) = \frac{1}{2} \left( \frac{1}{n_1} S_1 \text{diag}(v_1) + \frac{1}{n_2} \text{diag}(v_2) S_2 \right) \]

## 理论结果
ULOT通过最小化期望FUGW损失来训练模型，从而能够在较短的时间内预测出高质量的传输计划。此外，模型可以作为经典求解器的初始值，加速其收敛速度。

## 实验与证据
### 模拟数据集
- **数据集**：使用随机生成的Stochastic Block Model（SBM）图，包含不同数量的簇。
- **训练设置**：使用50000对模拟图进行训练，评估ULOT在新图上的性能。
- **结果**：ULOT预测的传输计划与经典求解器的结果高度相关，且计算速度快两个数量级。

### 实际数据集
- **数据集**：使用Individual Brain Charting（IBC）数据集，包含功能性MRI激活数据。
- **训练设置**：使用160k顶点的大脑皮层表面数据，构建1000节点的图。
- **结果**：ULOT能够有效地预测fMRI激活，保持了总体趋势的一致性。

## 局限与批判
1. **参数选择**：尽管ULOT可以在较大范围内预测传输计划，但参数 \( (\rho, \alpha) \) 的选择仍然依赖于经验，缺乏自动化的最佳参数选择机制。
2. **模型泛化能力**：尽管ULOT在模拟和实际数据集上表现良好，但在更复杂的图结构或更大规模的数据集上，模型的泛化能力有待进一步验证。
3. **计算复杂度**：尽管ULOT的计算复杂度为 \( O(n_1 n_2) \)，但对于非常大的图，仍可能存在计算瓶颈。
4. **理论基础**：尽管ULOT在实践中表现出色，但其理论基础仍有待进一步完善，特别是关于模型的稳定性和鲁棒性的分析。

## 与谁对话
- **T08**: 本文与T08的讨论有关，因为两者都探讨了GFlowNet与变分推断的关系，但本文更侧重于无监督学习框架下的最优传输计划预测。
- **O01**: 本文借鉴了O01中关于熵正则化的最优传输概念，特别是Sinkhorn算法和Schrödinger桥，这些概念在本文的FUGW损失函数设计中起到了关键作用。
- **N038**: 本文与N038在处理图上的最优传输问题上有直接联系，但本文专注于不平衡图情况下的无监督学习方法，而N038则更广泛地讨论了广义Schrödinger桥的概念。
- **N043**: 本文与N043在扩散过程和最优传输之间建立了联系，但本文特别关注于不平衡图的最优传输计划预测，而N043则更侧重于扩散Schrödinger桥的匹配问题。

## 对后续研究的启示
1. **自动化参数选择**：开发一种自动化机制来选择最优的 \( (\rho, \alpha) \) 参数，减少人工干预。
2. **模型泛化能力提升**：探索如何提高模型在更复杂图结构和更大规模数据集上的泛化能力。
3. **理论分析**：深入研究ULOT的理论基础，特别是关于模型稳定性和鲁棒性的分析。