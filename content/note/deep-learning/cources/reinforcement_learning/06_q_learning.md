# 06 Q 学习

## 引入

### 基础

回顾一下 Q function 的公式：
$$
Q^\pi(s_t,a_t)
=
\sum_{t'=t}^{T}
\mathbb{E}_{\pi}
[
r(s_{t'},a_{t'})
\mid s_t,a_t
]
$$
指的是如果我在当前状态 $s_t$ 先执行动作 $a_t$，然后之后一直按照旧策略 $\pi$ 行动，我最终能拿到多少总奖励。

考虑一个算法，也就是每次都使用贪心，让策略每次都选择能够让 Q function 最大的行为
$$
\pi_{\text{new}}(a_t|s_t)=
\begin{cases}
1, & \text{if } a_t=\arg\max_a \hat Q^\pi(s_t,a)\\
0, & \text{otherwise}
\end{cases}
$$
在 $Q^\pi(s,a)$ 足够准确的时候，新的策略不会比原来的策略差，一般会更好或者一样；但是这个新策略不一定是实际上的最优策略。

因为新策略只做了一次策略改进，它用的是旧策略的 $Q^\pi$，而不是自己的 $Q^{\pi_{\text{new}}}$，也不是最优 $Q^*$。

也就是说，$\pi_{\text{new}}$ 的决策逻辑是**当前这一步选一个最好的动作，但是假设之后仍然按照旧策略 $\pi$ 行动**，只保证“一步改进”，不保证全局最优。

### 移除梯度

根据上面的思想，可以跳过梯度下降的过程，直接定义一个新的策略估计 $Q^\pi$，再让策略变成对 $Q^\pi$ 贪心。

具体流程参见下图，也就是直接对 Q 做贪心的学习，而跳过梯度下降的过程。

![](http://oss.rainerseventeen.cn/blog/2026/202605151008797.png)

理论上这是可行的，但是有几个问题：

1. $\arg\max$ 对于连续的空间比较难处理，对于连续的取值范围 + 多个值 就会构成一个复杂的优化问题。
2. $Q^\pi$ 本身也是估计值，会有误差，$\arg\max$ 天然倾向于选择被高估的动作
3. 完全的贪心就会缺乏探索性，这也是强化学习的一个核心要素，需要让模型尝试自己探索而不是单纯的学习某种固定的行为。

## 方法

### 基础

原来的方法中，首先训练一个 Actor，然后训练一个 Critic；在 Q-learning 中只训练一个 $Q(s,a)$ 并直接用它的最大值作为策略。

基于 Actor Critic 方法改进后可以得到如下流程：

1. 执行动作 $a \sim \pi(a|s)$ 并从环境中获取到 $(s,a,s',r)$, 并把数据放进 Replay Buffer $\mathcal{R}$ 
2. 从 $\mathcal{R}$ 中采样一个 batch 的数据 $\{s_i,a_i,r_i,s_i'\}$ 
3. 最重要的一步，详细介绍一下：

定义一个标签，用这个 $y_i$ 作为目标，去训练当前的 $Q_\phi(s_i,a_i)$（该式可以参考 04 章节 中Bootstrapping 小节）
$$
y_i = r_i + \gamma \max_{a'} \hat{Q}_\phi(s_i',a')
$$

注意到这里后一项使用的不是期望而是确切的最大值$\max_{a'} Q(s',a')$，此时强化学习类似于监督学习，学习一个确定的标签。

这里的 $Q$ 实际上也是一个网络的估计值，但是 $Q$ 本身也是在反复优化的。利用 Bellman optimality equation 思想，最终会收敛到一个不动点上。
$$
Q_{\text{new}}(s,a)
\leftarrow
r + \gamma \max_{a'}Q_{\text{old}}(s',a')
$$
如果行为是离散的点（表格型），例如说上下左右等，那么基本是可以收敛的，因为不同的动作之间是相对独立的参数。

但如果是连续的函数型的行为，则不一定能够保证行为收敛，需要使用额外的技巧来稳定整个训练流程。

### 数据收集

观察更新更新目标中的 $\max_{a'} \hat Q_\phi(s_i', a')$ 表明更新时候需要知道下一状态 $s_i'$ 下很多动作的 $Q$ 值。

为了更好的估计 Q function 需要尽可能多地覆盖动作种类，所以需要更新一下数据的获取方式。

####  $\epsilon$-greedy

$$
\pi(a_t|s_t)
=
\begin{cases}
1-\epsilon, & \text{if } a_t = \arg\max_a Q_\phi(s_t,a) \\
\epsilon/(|\mathcal A|-1), & \text{otherwise}
\end{cases}
$$

以 $1-\epsilon$ 的概率选择当前 Q 最大的动作，以 $\epsilon$ 的概率探索其他动作。

一般 $\epsilon$ 会随着训练的推进逐渐变小。训练早期网络不准所以需要多次探索，在后期网络可靠的时候可以减少探索。

#### Boltzmann Exploration

Q 值越大的动作，被采样的概率越高，但 Q 值较低的动作仍然有机会被选中：
$$
\pi(a|s)
=
\frac{\exp(Q_\phi(s,a))}
{\sum_{a'} \exp(Q_\phi(s,a'))}
$$

### 综合流程

初始化 Q 网络 $Q_\phi$ 和 replay buffer $\mathcal{R}$，用当前策略和环境交互，例如 $\epsilon$-greedy，得到 transition：
$$
(s_i,a_i,s_i',r_i)
$$
然后把 transition 存入 replay buffer。从 replay buffer 采样 mini-batch。对每条数据构造 Bellman target：
$$
y_i = r_i + \gamma \max_{a'} Q_\phi(s_i',a')
$$
让 $Q_\phi(s_i,a_i)$ 逼近这个 target，重复收集数据和训练，最后使用：
$$
a = \arg\max_a Q_\phi(s,a)
$$
作为最终策略。























