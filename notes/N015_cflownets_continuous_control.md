# N015 · CFlowNets: Continuous Control with Generative Flow Networks

> 发表：ICLR 2023 · 链接：https://openreview.net/forum?id=yAYHho4fATa

## 一句话
CFlowNets 是一种用于连续控制任务的生成流网络，通过近似计算连续流入和流出，实现对连续状态和动作空间的有效探索。

## 问题与动机
传统的强化学习算法在连续控制任务中存在探索不足的问题，尤其是在稀疏奖励环境中。而 GFlowNets 能够生成与终止状态奖励成比例的分布，适用于探索任务。然而，GFlowNets 原本设计用于离散任务，如何将其扩展到连续任务是一个挑战。因此，提出 CFlowNets，旨在解决连续控制任务中的探索问题。

## 方法核心
### 理论基础
考虑一个连续任务 $(S, A)$，其中 $S$ 表示连续状态空间，$A$ 表示连续动作空间。定义轨迹 $\tau = (s_1, \ldots, s_n)$ 为从状态空间 $S$ 中采样的序列，使得每次转移 $s_t \rightarrow s_{t+1}$ 属于动作空间 $A$。进一步定义无环轨迹 $\tau = (s_1, \ldots, s_n)$ 满足无环约束：$\forall s_m, s_k \in \tau, m \neq k$，有 $s_m \neq s_k$。记 $s_0$ 和 $s_f$ 分别为初始状态和最终状态，定义完整轨迹为从 $(S, A)$ 开始于 $s_0$ 结束于 $s_f$ 的任何采样无环轨迹。相应地，转移到最终状态的转移称为终止转移，记为 $F(s \rightarrow s_f)$。

### 流量定义
- **连续状态流量**：
  $$F(s) = \int_{\tau: s \in \tau} F(\tau) d\tau.$$
  其中 $F(\tau)$ 是轨迹 $\tau$ 的流量，表示共享相同路径 $\tau$ 的粒子数。
  
- **连续流入**：
  $$\int_{s \in P(s_t)} F(s \rightarrow s_t) ds = \int_{s: T(s, a) = s_t} F(s, a) ds = F(s_t) = \int_{a: T(s, a) = s_t} F(s, a) da.$$
  其中 $P(s_t)$ 是状态 $s_t$ 的父集，包含所有能够直接转移到 $s_t$ 的直接父状态。
  
- **连续流出**：
  $$\int_{s \in C(s_t)} F(s_t \rightarrow s) ds = F(s_t) = \int_{a \in A} F(s_t, a) da.$$
  其中 $C(s_t)$ 是状态 $s_t$ 的子集，包含所有能够从 $s_t$ 直接转移的所有直接子状态。

### 过渡概率
- **前向过渡概率**：
  $$P_F(s_{t+1}|s_t) = \frac{F(s_t \rightarrow s_{t+1})}{F(s_t)}.$$
  
- **后向过渡概率**：
  $$P_B(s_t|s_{t+1}) = \frac{F(s_t \rightarrow s_{t+1})}{F(s_{t+1})}.$$

### 流匹配条件
对于任意非负函数 $\hat{F}(s, a)$，其满足连续流匹配条件：
$$\forall s' > s_0, \quad \hat{F}(s') = \int_{s \in P(s')} \hat{F}(s \rightarrow s') ds = \int_{s: T(s, a) = s'} \hat{F}(s, a: s \rightarrow s') ds,$$
$$\forall s' < s_f, \quad \hat{F}(s') = \int_{s'' \in C(s')} \hat{F}(s' \rightarrow s'') ds'' = \int_{a \in A} \hat{F}(s', a) da.$$

### 近似损失函数
基于采样轨迹近似连续损失函数：
$$L_\theta(\tau) = \sum_{s_t = s_1}^{s_f} \left[ \sum_{k=1}^K F_\theta(G_\phi(s_t, a_k), a_k) - \lambda R(s_t) - \sum_{k=1}^K F_\theta(s_t, a_k) \right]^2,$$
其中 $\theta$ 是流网络 $F(\cdot)$ 的参数，$\lambda = K/\mu(A)$，$\mu(A)$ 是动作空间 $A$ 的测度。

## 理论结果
### 定理 1
定理 1 扩展了 GFlowNets 的流匹配条件到连续场景，证明了只要非负函数满足流匹配条件，则唯一确定了一个马尔可夫流。

### 定理 2
定理 2 提供了样本流入/流出与实际流入/流出之间的误差界，表明随着样本数量 $K$ 的增加，误差呈指数级减小。

## 实验与证据
实验在多个连续控制任务上进行，包括 Point-Robot-Sparse、Reacher-Goal-Sparse 和 Swimmer-Sparse。CFlowNets 在这些任务上的表现优于多种强化学习算法，特别是在探索能力方面。具体环境、基线及具体数字如下：
- **Point-Robot-Sparse**：CFlowNets 较 DDPG、TD3、SAC 和 PPO 表现更好。
- **Reacher-Goal-Sparse**：CFlowNets 较 DDPG、TD3、SAC 和 PPO 表现更好。
- **Swimmer-Sparse**：CFlowNets 较 DDPG、TD3、SAC 和 PPO 表现更好。

## 局限与批判
1. **理论假设**：CFlowNets 基于若干假设，如动作是平移动作、状态空间和动作空间是连续的等。这些假设在某些复杂任务中可能不成立。
2. **计算复杂度**：虽然 CFlowNets 不需要像其他强化学习算法那样大的回放缓冲区，但其计算复杂度较高，特别是在高维连续任务中。
3. **参数敏感性**：CFlowNets 的性能对超参数设置较为敏感，特别是样本流的数量和动作概率缓冲区大小。
4. **泛化能力**：CFlowNets 在特定任务上的表现较好，但在不同任务间的泛化能力有待验证。

## 与谁对话
- T01：GFlowNet 原始论文，讨论了 GFlowNets 的基本原理和离散任务的应用。
- T12：连续 GFlowNets 理论，探讨了 GFlowNets 在连续空间中的应用。
- T19：非无环理论，讨论了 GFlowNets 在非无环图中的应用。
- T32：何时学对分布，讨论了 GFlowNets 学习正确分布的条件。

## 对后续研究的启示
1. **理论拓展**：进一步拓展 CFlowNets 的理论基础，使其适用于更广泛的连续任务。
2. **计算优化**：开发高效的计算方法，降低 CFlowNets 的计算复杂度，提高其在高维任务中的实用性。
3. **超参数调整**：研究 CFlowNets 的超参数调整策略，以提高其在不同任务中的适应性和鲁棒性。