# N046 · 范畴型薛定谔桥匹配：离散空间上的 D-IMF 理论与算法

> **Categorical Schrödinger Bridge Matching**
> 作者：Grigoriy Ksenofontov, Alexander Korotin（Skoltech / MIPT / AIRI）· ICML 2025 · [arXiv](https://arxiv.org/abs/2502.01416) · [代码](https://github.com/gregkseno/csbm)

## 一句话

CSBM 证明了离散时间迭代马尔可夫拟合（D-IMF）在有限离散空间 $\mathbb{S}^D$ 上收敛到薛定谔桥（Schrödinger Bridge, SB），并给出用 D3PM 式变分目标实现 Markovian 投影的实用算法，把 SB 的无配对翻译能力带进 VQ 码本、文本 token 等范畴型数据。

## 1. 要解决的问题

SB 问题：给定两个边缘分布 $p_0, p_1$ 与参考分布 $q^{\text{ref}}$，在所有传输方案（transport plans）$\Pi(p_0,p_1)$ 中找 KL 意义下最接近参考的耦合（原文式 (1)）：

$$q^*(x_0,x_1)=\mathop{\mathrm{argmin}}_{q\in\Pi(p_0,p_1)}\text{KL}(q(x_0,x_1)\,\|\,q^{\text{ref}}(x_0,x_1))$$

它等价于熵正则最优传输（Entropic Optimal Transport, EOT，原文式 (2)），其中代价 $c(x_0,x_1)\stackrel{\text{def}}{=}-\log q^{\text{ref}}(x_0,x_1)$。应用场景是生成式设定的无配对域翻译（unpaired domain translation）：只有两个域的 i.i.d. 样本，要学出保语义的翻译映射，且支持样本外推断。

现有 SB 求解器几乎全部假设连续空间 $\mathcal{X}=\mathbb{R}^D$：SDE 系方法（DSBM 等）无法直接推广到离散；基于 MCMC 或二次代价 EOT 理论的方法同样绑定 $\mathbb{R}^D$。但大量现实数据是离散的——VQ 自编码器码本、文本 token、分子中的原子类别。理论缺口具体是（原文 Table 1）：**离散空间 + 离散时间（$N<\infty$）** 这一格没有"SB 是唯一既 Markov 又 reciprocal 的过程"的刻画定理，因此 D-IMF 在此设定下没有收敛保证。最接近的工作 DDSBM（Kim et al., 2024）处理离散空间但假设连续时间，实操中离散化到大 $N$ 后同样落在无理论的格子里。

注意与"离散 EOT"（Sinkhorn 一系）的区别：那些方法把经验分布本身当作离散分布求双随机矩阵，不支持生成式设定的样本外估计；本文的"离散"指状态空间 $\mathcal{X}=\mathbb{S}^D$ 是范畴型的。

## 2. 核心方法

### 2.1 动态 SB 与 D-IMF 框架（背景，§2.4–2.5）

引入 $N$ 个中间时刻 $0=t_0<t_1<\dots<t_N<t_{N+1}=1$，动态 SB 为（原文式 (3)）：

$$\min_{q\in\Pi_N(p_0,p_1)}\text{KL}(q(x_0,x_{\text{in}},x_1)\,\|\,q^{\text{ref}}(x_0,x_{\text{in}},x_1))$$

其中 $x_{\text{in}}=(x_{t_1},\dots,x_{t_N})$，$\Pi_N$ 是首末边缘固定的离散时间过程集合。KL 分解（原文式 (4)）表明动态解在 $t=0,1$ 的联合分布就是静态 SB。D-IMF 从任意 $q^0\in\Pi_N(p_0,p_1)$ 出发，交替做两种投影（原文式 (6)）：

$$q^{2l+1}=\text{proj}_{\mathcal{R}^{\text{ref}}}(q^{2l}),\qquad q^{2l+2}=\text{proj}_{\mathcal{M}}(q^{2l+1})$$

- **reciprocal 投影**：$[\text{proj}_{\mathcal{R}^{\text{ref}}}(q)](x_0,x_{\text{in}},x_1)=q^{\text{ref}}(x_{\text{in}}|x_0,x_1)\,q(x_0,x_1)$——保持端点耦合，把中间轨迹换成参考桥；
- **Markovian 投影**（原文式 (5)）：把过程投到马尔可夫过程集合 $\mathcal{M}(\mathcal{X}^{N+2})$，保持相邻时刻联合分布 $\{q(x_{t_n},x_{t_{n-1}})\}$，一般会改变端点耦合。

只要"SB 是 $\Pi_N(p_0,p_1)$ 中唯一既 Markov 又 reciprocal 的过程"成立，Shi et al. (2023, Theorem 8) 的通用论证就给出 KL 收敛。本文的工作就是在离散空间补上这个刻画。

### 2.2 参考过程的选择（§3.2.2）

单维（$D=1$）两种参考链，对应不同的类别结构先验：

- **均匀参考 $q^{\text{unif}}$**（类别无序，如 token、码本，原文式 (7)）：以概率 $1-\alpha$ 停留原类别，$\frac{\alpha}{S-1}$ 均匀跳到其他类别；$\alpha\in[0,1]$ 是随机度参数。
- **高斯参考 $q^{\text{gauss}}$**（类别有序，如像素强度，原文式 (8)）：跳转概率 $\propto\exp\!\big(-\tfrac{4(x_{t_n}-x_{t_{n-1}})^2}{(\alpha\Delta)^2}\big)$，$\Delta=S-1$ 为最大类别距离，$\alpha$ 类似方差参数。

$D>1$ 时各维独立组合；桥分布 $q^{\text{ref}}(x_{\text{in}}|x_0,x_1)$ 靠 Markov 性 + 贝叶斯公式解析可采样。

### 2.3 可学习过程的参数化（§3.2.3）

转移矩阵是 $S^D\times S^D$ 的，显式建模不可行。采用两个标准技巧的组合——端点后验采样 + 维度因子化（原文式 (9)）：

$$q_\theta(x_{t_n}|x_{t_{n-1}})=\mathbb{E}_{\widetilde{q}_\theta(\widetilde{x}_1|x_{t_{n-1}})}\big[q^{\text{ref}}(x_{t_n}|x_{t_{n-1}},\widetilde{x}_1)\big],\qquad \widetilde{q}_\theta(\widetilde{x}_1|x_{t_{n-1}})\approx\prod_{d=1}^D \widetilde{q}_\theta(\widetilde{x}_1^d|x_{t_{n-1}})$$

即先用网络预测"终点" $\widetilde{x}_1$ 的逐维分布（一个 $D\times S$ 行随机矩阵），再从参考桥采样下一步。单个网络加时间步输入，避免为每个 $n$ 训练独立模型。这与 D3PM（Austin et al., 2021）的 $x_0$-参数化同构。

### 2.4 Markovian 投影 = 变分目标（Proposition 3.3）

给定 reciprocal 过程 $q$，其 Markovian 投影可通过在 $m\in\mathcal{M}(\mathcal{X}^{N+2})$ 上最小化以下目标获得（原文式 (10)）：

$$L(m)=\mathbb{E}_{q(x_0,x_1)}\Big[\sum_{n=1}^{N}\mathbb{E}_{q^{\text{ref}}(x_{t_{n-1}}|x_0,x_1)}\text{KL}\big(q^{\text{ref}}(x_{t_n}|x_{t_{n-1}},x_1)\,\|\,m(x_{t_n}|x_{t_{n-1}})\big)-\mathbb{E}_{q^{\text{ref}}(x_{t_N}|x_0,x_1)}\log m(x_1|x_{t_N})\Big]$$

与 D3PM 变分下界的唯一关键差别：$x_{t_{n-1}}$ 不是从加噪过程 $q^{\text{ref}}(x_{t_{n-1}}|x_1)$ 采，而是从参考桥 $q^{\text{ref}}(x_{t_{n-1}}|x_0,x_1)$ 采。于是离散扩散的全部训练基建可以直接复用。

### 2.5 CSBM 算法（Algorithm 1）

双向训练（单向 IMF 会引入误差，De Bortoli et al., 2024, Appendix I）：外层 $L$ 轮，每轮先固定后向模型 $q_\eta$ 生成 $(x_0,x_1)$ 耦合训练前向模型 $q_\theta$（最小化 $L_\theta$，原文式 (21)），再对称地训练后向模型（$L_\eta$，式 (22)）。中间点从参考桥采样。

## 3. 理论结果

**Theorem 3.1（离散空间上动态 SB 解的刻画）**：设 $\mathcal{X}$ 有限离散，$p_0,p_1$ 全支撑，$q^{\text{ref}}\in\mathcal{M}(\mathcal{X}^{N+2})$ 为全支撑 Markov 参考过程。若 $q^*$ 满足 ① 端点边缘为 $p_0,p_1$；② 既是 Markov 又是 reciprocal，则 $q^*$ 是动态 SB (3) 的唯一解。

**Corollary 3.2（D-IMF 收敛性）**：上述条件下 D-IMF 序列 $\{q^l\}$ 满足 $\lim_{l\to\infty}\text{KL}(q^l\|q^*)=0$。

两点值得强调：(i) 定理只要求参考过程是一般 Markov 链——不需要 Wiener 过程结构，比 Gushchin et al. (2024b, Theorem 3.6)（$\mathbb{R}^D$ + 二次代价）条件更弱；作者在脚注 1 指出证明论证适用于任意 $\mathcal{X}$，顺带把 ASBM 推广到一般 Markov 参考。(ii) 理论上 $N=1$ 就够——与 DDSBM 需要大 $N$ 逼近连续时间形成对照。证明在 Appendix B。

## 4. 实验与证据

- **D-IMF 解析收敛（§4.1）**：$S=50, D=1$ 时转移矩阵可显式计算，与 Sinkhorn 算出的真值 $q^*$ 比较，$\text{KL}(q^l\|q^*)$ 收敛曲线（Figure 1）显示对不同 $N$、$\alpha$、参考过程均快速收敛。
- **2D 高斯→Swiss Roll（§4.2）**：$S=50$，$|\mathcal{X}|=2500$，$N=10$。$q^{\text{gauss}}$ 的跳转集中于邻近类别，$q^{\text{unif}}$ 全类别跳转，$\alpha$ 越大跳越多（Figure 2），与构造一致。KL 损失换 MSE 结果相当（Appendix C.1）。
- **Colored MNIST "3"→"2"（§4.3）**：原始像素空间 $S=256$，$D=32\times32\times3$，$q^{\text{gauss}}$，$\alpha=0.01$。$N=2$ 已有不错视觉质量与颜色保持；因子化导致轻微像素化，随 $N$ 增大减轻；但相同梯度步数下大 $N$ 相似度下降，作者归因于欠拟合（转移概率个数正比于 $N$）。
- **CelebA 128×128 男→女（§4.4）**：VQ-GAN 潜空间 $S=1024, D=256$，$q^{\text{unif}}$，$N=100$。对比像素空间的 ASBM 与 DSBM（原文 Table 2）：低随机度下 FID 10.60 vs 16.86（ASBM）/24.06（DSBM）；高随机度下 14.68 vs 17.44/92.15；CMMD 与 LPIPS 同样全面占优，且背景保持明显更好。注意口径：CSBM 用 $N=100$，ASBM 只用 $N=3$。
- **文本情感迁移**（Amazon Reviews，Appendix C.4）：展示 token 空间可行性。

## 5. 在 GFlowNet 版图中的位置

本仓库收录它的理由不是 GFlowNet 方法本身，而是它补齐了**离散空间上概率路径测度匹配**的一块理论版图，与 GFlowNet 社区三处交汇：

- GFlowNet 把离散组合对象的采样表为 DAG 上的流，SB/EOT 把无配对翻译表为路径测度上的 KL 投影；两者都在"离散序列决策 + 边缘约束"框架下做分布匹配。CSBM 的 Markovian 投影损失（KL 于逐步转移）与 GFlowNet 的 DB/TB 一族"局部一致性条件转训练目标"的思路同构。
- GFlowNet 文献中的桥采样与扩散采样线（如 Sendera et al., 2024；Zhang et al., 2024 的 diffusion GFN 系列）主要在连续空间；CSBM 提供了离散侧的对应物与理论模板——尤其是"一般 Markov 参考过程即可保证唯一性"这一结论，对设计离散 GFlowNet 式桥算法直接有用。
- D3PM/离散流匹配（Gat et al., 2024）是 GFlowNet 之外离散生成的主流路线；CSBM 表明这套基建可以无缝承载 SB 目标（只改中间点采样分布），是"离散扩散 ↔ 最优传输"的桥梁工作。

谱系：IMF（Peluchetti, 2023; Shi et al., 2023 DSBM）→ D-IMF/ASBM（Gushchin et al., 2024b）→ 本文（离散空间 + 离散时间收敛理论）；平行分支 DDSBM（离散空间 + 连续时间，无离散化理论）被本文覆盖。

## 6. 局限与批判

- 因子化假设 $\widetilde{q}_\theta(\widetilde{x}_1|x_{t_{n-1}})\approx\prod_d\widetilde{q}_\theta(\widetilde{x}_1^d|\cdot)$ 忽略维度间耦合，是 MNIST 像素化伪影的直接来源（作者在 Appendix A 承认）；小 $N$ 时该近似的误差没有理论量化。
- 收敛定理是渐近的：没有收敛速率，也没有"神经网络近似投影 + 有限训练步"下的误差累积分析；实验里"大 $N$ 欠拟合"现象恰说明实践收敛对计算预算敏感。
- CelebA 对照不完全公平：CSBM 在 VQ-GAN 潜空间（作者论证了为何不在潜空间训 ASBM/DSBM，见 Appendix C.3），且 $N=100$ vs ASBM 的 $N=3$，FID 优势混杂了表示空间与步数两个因素。
- 全支撑假设（$p_0,p_1$ 与 $q^{\text{ref}}$ 均全支撑）在定理中不可去，但 VQ 码本上的真实分布高度稀疏，理论与实践的支撑条件有落差。
- 参考过程仅测试了 uniform/Gaussian 两种手工构造；SB 对 $q^{\text{ref}}$ 的选择本质敏感（它定义了传输代价），如何为特定域学习或设计参考链未触及。

## 7. 对后续研究的启示

- "一般 Markov 参考即可"的刻画为**结构化参考过程**打开空间：在分子图上用化学合理的编辑链、在文本上用掩码语言模型链作 $q^{\text{ref}}$，SB 解将继承相应归纳偏置。
- $N=1$ 理论可行提示做**少步离散翻译器**：训练一个单步端点预测器反复精化，与一致性模型（consistency models）的离散版对接。
- 对 GFlowNet 一线：CSBM 的双向 D-IMF 与 GFlowNet 的前向/后向策略对偶训练形式相似，可探索把 GFlowNet 的 off-policy 探索机制嫁接到 D-IMF 的耦合更新中，缓解模式坍缩。
- VQ 潜空间 + 离散 SB 的配方可推广到任何"连续数据 + 离散化表示"的翻译任务（音频、视频、3D），FID 证据表明潜空间离散建模可能优于像素空间连续 SB。
- 开放理论问题：D-IMF 在离散空间的收敛速率、神经近似误差的传播、以及支撑不满时的行为。
