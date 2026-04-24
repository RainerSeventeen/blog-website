---
paper:
  - "Proximal Policy Optimization Algorithms"
  - "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
---

# RL Basic

补一点基本的 RL 相关的内容，作为后文的前置知识

## RL 基础

和普通的监督学习不一样，RL 的行为逻辑是：

1. 智能体在环境里看到一个状态 $s_t$

2. 采取动作 $a_t$

3. 环境返回奖励 $r_t$

4. 同时转移到下一个状态 $s_{t+1}$

所以 RL 的本质是：**通过与环境不断交互，学到一个策略，使长期累计奖励最大。**

如果把目标定得更明确一点，这里并不是要系统学完整套强化学习，而是先把 PPO 在解决什么问题、公式里的符号分别表示什么、训练流程为什么这么设计补齐。

所以看 PPO 之前，前置知识其实不用很多，但要补得准。最核心的四层是：

1. MDP 与交互过程

2. 价值函数与优势函数

3. 策略梯度

4. 从 TRPO 到 PPO 的"为什么要限制更新幅度"

## MDP

PPO 论文默认把环境建模成 **马尔可夫决策过程（MDP, Markov Decision Process）**。一个 MDP 通常写成：

$$
(\mathcal S, \mathcal A, P, R, \gamma)
$$

分别表示：

- $\mathcal S$：状态空间
- $\mathcal A$：动作空间
- $P(s' \mid s, a)$：状态转移概率
- $R(s, a)$ 或 $R(s, a, s')$：奖励函数
- $\gamma \in (0, 1]$：折扣因子

这里最关键的假设是 **马尔可夫性**：

$$
P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \dots) = P(s_{t+1} \mid s_t, a_t)
$$

也就是：下一个状态只依赖当前状态和当前动作，不依赖更久之前的历史。这个假设很重要，因为后面很多价值函数和递推公式都建立在它上面。

一条交互轨迹通常记作：

$$
\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \dots)
$$

PPO 训练时会先用当前策略去环境里跑，收集很多这样的轨迹，这个过程常叫：

- sampling
- rollout
- collect trajectories

你在 PPO 代码里经常看到的 `obs`、`actions`、`rewards`、`dones`、`log_probs`、`values`，本质上就是一批 rollout 数据。

## Return、Policy、Value

这一部分是看 PPO 前最需要先吃透的内容。如果这里没建立起来，后面的 loss 会显得很抽象。

### Return

从时刻 $t$ 开始的累计回报定义为：

$$
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots = \sum_{k=0}^{\infty} \gamma^k r_{t+k}
$$

它表示：从当前开始，未来总共能拿到多少奖励。

之所以要乘折扣因子 $\gamma$，直觉上有三个原因：

- 更远处的奖励权重更小
- 数学上更稳定
- 表达"越近的收益越重要"

### Policy

策略就是"在一个状态下怎么选动作"的规则，通常记为：

$$
\pi(a \mid s)
$$

表示在状态 $s$ 下选择动作 $a$ 的概率。参数化之后通常写成：

$$
\pi_\theta(a \mid s)
$$

其中 $\theta$ 是策略网络参数。PPO 这类方法通常使用随机策略，因为它本质上属于策略梯度方法。

### State Value

状态价值函数定义为：

$$
V^\pi(s) = \mathbb E_\pi [G_t \mid s_t = s]
$$

意思是：如果当前在状态 $s$，之后都按照策略 $\pi$ 行动，期望能拿到多少累计回报。

它评估的是：**这个状态本身有多好。**

### Action Value

动作价值函数定义为：

$$
Q^\pi(s, a) = \mathbb E_\pi [G_t \mid s_t = s, a_t = a]
$$

意思是：在状态 $s$ 下先采取动作 $a$，后面继续按策略 $\pi$ 行动，期望总回报是多少。

它评估的是：**在这个状态下做这个动作有多好。**

### Advantage

优势函数定义为：

$$
A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)
$$

它表示：这个动作比该状态下的平均水平好多少。

拆开看就很好理解：

- $V^\pi(s)$：状态 $s$ 的平均水平
- $Q^\pi(s, a)$：在 $s$ 下选了动作 $a$ 的效果
- 两者相减：这个动作相对平均水平到底是赚了还是亏了

因此：

- $A(s, a) > 0$：这个动作比平均好，应该提高它的概率
- $A(s, a) < 0$：这个动作比平均差，应该降低它的概率

后面 PPO 目标函数里的 $\hat A_t$，本质上就在做这件事。

## Bellman 关系

虽然这里不展开完整推导，但至少要能看懂它在表达什么。

因为 return 满足：

$$
G_t = r_t + \gamma G_{t+1}
$$

所以价值函数会满足 Bellman expectation equation：

$$
V^\pi(s) = \mathbb E_\pi [r_t + \gamma V^\pi(s_{t+1}) \mid s_t = s]
$$

直觉上就是：

**当前状态的价值 = 当前奖励 + 下一个状态的折扣价值。**

类似地：

$$
Q^\pi(s, a) = \mathbb E [r_t + \gamma V^\pi(s_{t+1}) \mid s_t = s, a_t = a]
$$

这两个递推关系后面会直接连到 TD、GAE、critic。

## Actor-Critic 与 TD

如果直接用采样得到的 $G_t$ 去更新策略，方差通常会很大，训练不稳定。所以强化学习里常引入一个网络专门预测 $V(s)$，这就是 **critic**；而输出动作分布 $\pi_\theta(a \mid s)$ 的策略网络叫 **actor**。

于是就有：

- Actor：输出动作分布 $\pi_\theta(a \mid s)$
- Critic：输出状态价值 $V_\phi(s)$

PPO 本质上就是一个 actor-critic 方法。

进一步，如果每次都等整条 trajectory 结束，再去计算完整 return，效率低，方差也大。于是引入 **Temporal Difference, TD** 的思想，定义一步 TD 误差：

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

它表示的是：当前价值估计和"一步奖励 + 下一个状态价值"这个目标之间差了多少。

可以这样理解：

- $\delta_t > 0$：说明这个状态比原来估得更好
- $\delta_t < 0$：说明原来的价值估计偏高了

这个量后面会直接进入 GAE。

## GAE：PPO 里的 Advantage 是怎么来的

PPO 公式里经常出现 $\hat A_t$。很多实现里它并不是直接用 Monte Carlo return 算出来的，而是通过 **GAE（Generalized Advantage Estimation）** 来估计。

理论上：

$$
A^\pi(s_t, a_t) = Q^\pi(s_t, a_t) - V^\pi(s_t)
$$

但 $Q^\pi$ 往往拿不到，所以只能估计。

最简单的一步近似就是刚才的 TD 误差：

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

它本身就可以看成一种一步 advantage 估计。

GAE 则是把未来很多步的 TD 误差按权重累加起来：

$$
\hat A_t^{\text{GAE}(\gamma, \lambda)} =
\delta_t + (\gamma \lambda)\delta_{t+1} + (\gamma \lambda)^2\delta_{t+2} + \cdots
$$

其中：

- $\gamma$：折扣因子
- $\lambda$：控制 bias-variance tradeoff 的参数

最重要的直觉是：

- $\lambda$ 小：更依赖短期 TD，偏差大一点，但方差小
- $\lambda$ 大：更接近 Monte Carlo，偏差小，但方差大

PPO 常用 GAE，就是为了得到一个相对稳定的 advantage 估计。

## 策略梯度

这是 PPO 最直接的理论来源。

强化学习里我们希望最大化策略的期望回报：

$$
J(\theta) = \mathbb E_{\tau \sim \pi_\theta}[R(\tau)]
$$

其中 $R(\tau)$ 是整条轨迹的回报。

问题在于：轨迹是通过采样产生的，不能像普通监督学习那样直接对标签求导，所以需要策略梯度。

策略梯度最核心的形式可以写成：

$$
\nabla_\theta J(\theta)
=
\mathbb E_{\pi_\theta}\left[
\nabla_\theta \log \pi_\theta(a_t \mid s_t)\, A^\pi(s_t, a_t)
\right]
$$

这个公式不用一开始就完整推导，但一定要抓住它的直觉：

- $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$：告诉你怎样改参数，才能让这个动作更可能或者更不可能
- $A^\pi(s_t, a_t)$：告诉你这个动作到底值不值得鼓励

所以它表达的就是：

**如果某个动作的 advantage 为正，就提高它的概率；如果 advantage 为负，就降低它的概率。**

这里前面的 log 概率来自一个常见技巧：

$$
\nabla_\theta \pi_\theta(a \mid s)
=
\pi_\theta(a \mid s)\nabla_\theta \log \pi_\theta(a \mid s)
$$

这就是所谓的 log-derivative trick。你现在不需要把证明推完，但要知道后面 PPO 代码里的 `old_log_prob`、`new_log_prob`、`ratio` 都和这里直接相关。

## 从 REINFORCE 到 Actor-Critic

最原始的策略梯度方法例如 REINFORCE，通常直接使用 return：

$$
\nabla_\theta J(\theta) \approx \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) G_t
$$

问题是：

- 方差很大
- 训练不稳定
- 样本效率低

所以后来会减去一个 baseline，最常见的就是 $V(s_t)$，也就是用：

$$
G_t - V(s_t)
$$

这其实就是 advantage 的雏形。再往后就发展成 actor-critic：actor 学策略，critic 学价值，用来降低方差、提高训练稳定性。PPO 基本就是这条路线上的现代稳定版本。

## 为什么 PPO 里会出现 Old Policy、Ratio、Clip

这是读 PPO 论文前必须先建立起来的动机。

如果直接按普通策略梯度去更新，可能会发生：

- 一次更新太大
- 策略突然变化
- 性能崩掉
- 训练变得很不稳定

所以一个自然的问题是：**能不能在每次更新时，不要让新策略离旧策略太远？**

这就是 TRPO 和 PPO 的背景。

PPO 公式里会出现一个非常关键的比值：

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
$$

它表示：当前新策略对这个动作的概率，相比旧策略变化了多少。

如果：

- $r_t(\theta) > 1$：新策略更倾向这个动作
- $r_t(\theta) < 1$：新策略更不倾向这个动作

之所以需要它，是因为数据通常是由旧策略采样出来的，但你现在想评估新策略，这时就需要 importance sampling ratio 来修正分布差异。

如果只使用

$$
r_t(\theta)\hat A_t
$$

作为目标，那么当 $r_t$ 变得很大或很小时，策略更新可能会过于激进。

PPO-Clip 于是引入了下面这个目标：

$$
L^{\text{CLIP}}(\theta)
=
\mathbb E_t \left[
\min\left(
r_t(\theta)\hat A_t,\;
\text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon)\hat A_t
\right)
\right]
$$

它的关键直觉不是推导，而是"保守更新"：

- 当 $\hat A_t > 0$ 时，我们希望提高该动作概率，但不能提高得太夸张，所以把 $r_t$ 限制在 $1 + \epsilon$ 附近
- 当 $\hat A_t < 0$ 时，我们希望降低该动作概率，但也不能降得过猛，所以把 $r_t$ 限制在 $1 - \epsilon$ 附近
- `min` 的作用是取一个更保守的目标，避免策略更新过度

所以 PPO 用一句话概括就是：

**在策略梯度的基础上，用 clip 机制限制每次策略更新的幅度，提高训练稳定性。**

## 读 PPO 前最小公式清单

如果只想先把 PPO 论文读顺，下面这些公式基本够用：

回报：

$$
G_t = \sum_{k=0}^{\infty}\gamma^k r_{t+k}
$$

状态价值：

$$
V^\pi(s)=\mathbb E_\pi[G_t \mid s_t=s]
$$

动作价值：

$$
Q^\pi(s,a)=\mathbb E_\pi[G_t \mid s_t=s, a_t=a]
$$

优势函数：

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)
$$

一步 TD 误差：

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

GAE：

$$
\hat A_t = \delta_t + \gamma\lambda \delta_{t+1} + (\gamma\lambda)^2 \delta_{t+2} + \cdots
$$

策略梯度核心形式：

$$
\nabla_\theta J(\theta)
=
\mathbb E\left[\nabla_\theta\log \pi_\theta(a_t \mid s_t)\hat A_t\right]
$$

PPO ratio：

$$
r_t(\theta)=\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
$$

PPO clip 目标：

$$
L^{\text{CLIP}}(\theta)
=
\mathbb E_t \left[
\min\left(
r_t(\theta)\hat A_t,\;
\text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\right)
\right]
$$

到这里为止，后面的 PPO 章节里出现的大部分符号其实就都能对上了。





# PPO

> 论文全名：*Proximal Policy Optimization Algorithms*，2017，来自 OpenAI

## 简介

PPO 的全称是 *Proximal Policy Optimization*（近端策略优化），是一种通用的策略梯度强化学习算法，在 RLHF 中被广泛用于基于 reward model 的策略优化阶段。

在 PPO 之前，策略梯度方法存在两个主要问题：

1. **Vanilla Policy Gradient**：每采一批数据只能做一次梯度更新，样本效率低，而且如果步长太大，策略会发生灾难性的大幅变化

2. **TRPO（Trust Region Policy Optimization）**：通过 KL 散度硬约束限制更新幅度，保证单调改进，但实现复杂（需要共轭梯度法），不兼容 dropout 和参数共享架构

   PPO 的目标是：**保留 TRPO 的稳定性，同时像 vanilla PG 一样简单，只需要一阶优化（SGD/Adam）**。

## 策略梯度基础

标准策略梯度的梯度估计：

$$
\hat{g} = \hat{\mathbb{E}}_t \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \hat{A}_t \right]
$$

其中 $\hat{A}_t$ 是优势函数（advantage function）的估计——衡量"在状态 $s_t$ 选择动作 $a_t$，比平均水平好多少"。对应的损失函数（通过求梯度得到）：

$$
L^{PG}(\theta) = \hat{\mathbb{E}}_t \left[ \log \pi_\theta(a_t \mid s_t) \hat{A}_t \right]
$$

但问题是：在同一批数据上做多步优化这个损失，策略会更新过大导致崩溃。

## TRPO 的思路

TRPO 引入了**重要性采样比（importance sampling ratio）**，用旧策略采集的数据来优化新策略，并用 KL 散度作为硬约束：

$$
\max_\theta \hat{\mathbb{E}}_t \left[ \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} \hat{A}_t \right] \quad \text{s.t.} \quad \hat{\mathbb{E}}_t \left[ \text{KL}[\pi_{\theta_{\text{old}}}(\cdot \mid s_t), \pi_\theta(\cdot \mid s_t)] \right] \leq \delta
$$

理论上也可以改成 KL 惩罚项的无约束形式：

$$
\max_\theta \hat{\mathbb{E}}_t \left[ \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} \hat{A}_t - \beta \text{KL}[\pi_{\theta_{\text{old}}}, \pi_\theta] \right]
$$

但实验表明固定 $\beta$ 效果不好——不同问题、不同训练阶段需要不同的 $\beta$，所以 TRPO 选择了硬约束的实现方式，导致了实现复杂性。

## PPO-Clip：核心方法

PPO 提出了一个巧妙的替代方案。定义概率比 $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$，PPO 的 clipped surrogate objective 为：

$$
L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]
$$

其中 $\epsilon$ 是超参数，通常取 0.2。

这个公式的直觉解释：

- **$r_t(\theta) \hat{A}_t$**：就是标准的 importance sampling 目标（和 TRPO 一样）

- **$\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t$**：把概率比裁剪到 $[1-\epsilon, 1+\epsilon]$ 范围内，超过这个范围的变化不再提供梯度信号

- **取 min**：在两者之间取较小值，形成**悲观估计（下界）**

  分两种情况理解 clip 的效果：

1. **$\hat{A}_t > 0$（好动作）**：策略应该增大 $r_t$（提高这个动作的概率），但 clip 限制了 $r_t$ 最大只能到 $1+\epsilon$，**防止对好动作过度乐观**

2. **$\hat{A}_t < 0$（差动作）**：策略应该减小 $r_t$（降低这个动作的概率），但 clip 限制了 $r_t$ 最小只能到 $1-\epsilon$，**防止对差动作矫枉过正**

   本质上，clip 起到了类似 trust region 的作用——限制每次策略更新的幅度，但实现起来只需要一个 min 和 clip 操作，远比 TRPO 的共轭梯度法简单。

## PPO-Penalty：自适应 KL 惩罚

PPO 还提出了另一个变体，使用自适应 KL 惩罚系数：

$$
L^{\text{KLPEN}}(\theta) = \hat{\mathbb{E}}_t \left[ \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} \hat{A}_t - \beta \text{KL}[\pi_{\theta_{\text{old}}}(\cdot \mid s_t), \pi_\theta(\cdot \mid s_t)] \right]
$$

每次策略更新后，根据实际 KL 散度 $d$ 和目标值 $d_{\text{targ}}$ 动态调整 $\beta$：

- 如果 $d < d_{\text{targ}} / 1.5$，说明更新太保守，$\beta \leftarrow \beta / 2$

- 如果 $d > d_{\text{targ}} \times 1.5$，说明更新太激进，$\beta \leftarrow \beta \times 2$

  实验表明 PPO-Clip 效果优于 PPO-Penalty，因此实践中 PPO 通常指的是 PPO-Clip。

## 完整目标函数

在实际使用中（尤其是 actor-critic 架构），PPO 的完整损失函数还包括 value function 损失和熵正则：

$$
L^{\text{CLIP+VF+S}}_t(\theta) = \hat{\mathbb{E}}_t \left[ L^{\text{CLIP}}_t(\theta) - c_1 L^{VF}_t(\theta) + c_2 S[\pi_\theta](s_t) \right]
$$

其中：

- $L^{VF}_t = (V_\theta(s_t) - V^{\text{targ}}_t)^2$ 是 value function 的均方误差损失
- $S[\pi_\theta](s_t)$ 是策略的熵，鼓励探索
- $c_1, c_2$ 是系数

## 优势估计：GAE

PPO 使用 **Generalized Advantage Estimation (GAE)** 来估计优势函数：

$$
\hat{A}_t = \delta_t + (\gamma \lambda) \delta_{t+1} + \cdots + (\gamma \lambda)^{T-t+1} \delta_{T-1}
$$

其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ 是 TD 误差，$\gamma$ 是折扣因子，$\lambda$ 控制 bias-variance 的权衡（$\lambda = 1$ 时退化为 Monte Carlo 估计，$\lambda = 0$ 时退化为单步 TD）。

## 算法流程

```
for iteration = 1, 2, ... do
    for actor = 1, 2, ..., N do
        用当前策略 π_θ_old 在环境中跑 T 步，收集轨迹
    end for
    计算每一步的优势估计 Â_1, ..., Â_T
    在收集的 NT 条数据上优化 L^CLIP+VF+S，跑 K 个 epoch（minibatch SGD）
    θ_old ← θ
end for
```

关键特性：**同一批数据可以复用多次**（多个 epoch），这是 PPO 相比 vanilla PG 样本效率高的关键原因，也是 clip 机制的存在意义——保证多次更新不会偏离太远。

## PPO 在 RLHF 中的角色

在 RLHF 的语境下，PPO 的使用方式是：

- **状态 $s$**：prompt $x$ + 已生成的 token 序列

- **动作 $a$**：下一个生成的 token

- **策略 $\pi_\theta$**：语言模型

- **Reward**：只在序列生成完成后由 reward model 给出，中间步骤 reward 为 0

- **KL 惩罚**：通常在每个 token 级别加一个 KL 惩罚项 $-\beta \log \frac{\pi_\theta(y_t|x, y_{<t})}{\pi_{\text{ref}}(y_t|x, y_{<t})}$

- **Value Model**：需要额外训练一个 value head（通常在 LM 最后一层上加一个线性层）来估计 $V(s)$

  这也是为什么 RLHF 需要维护四个模型的原因：policy model、reference model、reward model、value model。

# DPO

> 论文全名：*Direct Preference Optimization: Your Language Model is Secretly a Reward Model*
>
> 发表于 NeurIPS 2023，作者来自 Stanford University

## 简介

DPO 的全称是 *Direct Preference Optimization*（直接偏好优化），是一种用人类偏好数据直接微调语言模型的方法，**不需要显式训练 reward model，也不需要强化学习**。

在 DPO 之前，让语言模型对齐人类偏好的主流方法是 RLHF（Reinforcement Learning from Human Feedback），其流程是：先用偏好数据训练一个 reward model，再用 PPO 等强化学习算法优化语言模型使其获得高 reward。这个过程复杂且不稳定，需要同时维护多个模型（policy、reward model、value model、reference model），超参数也很难调。

DPO 的核心发现是：**可以通过变量替换，把 RLHF 中的 RL 优化目标直接转化成一个基于策略模型本身的分类损失函数**，从而跳过 reward model 和 RL 训练，直接在偏好数据上用简单的交叉熵损失微调语言模型。

用论文的话说：你的语言模型本身就（隐式地）是一个 reward model。

## 背景：RLHF 流程回顾

标准 RLHF 包含三个阶段：

### 1. 监督微调（SFT）

在高质量数据上对预训练模型做监督微调，得到 $\pi^{\text{SFT}}$

### 2. Reward Model 训练

给定 prompt $x$，用 SFT 模型生成两个回答 $(y_1, y_2)$，让人类标注哪个更好（$y_w \succ y_l$），用 **Bradley-Terry 模型** 建模偏好概率：

$$
p^*(y_1 \succ y_2 \mid x) = \frac{\exp(r^*(x, y_1))}{\exp(r^*(x, y_1)) + \exp(r^*(x, y_2))}
$$

然后用极大似然训练 reward model $r_\phi$：

$$
\mathcal{L}_R(r_\phi, \mathcal{D}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right]
$$

其中 $\sigma$ 是 sigmoid 函数。直觉上就是让 reward model 给"好回答"打更高的分。

### 3. RL 微调（PPO）

用训好的 reward model 指导语言模型优化，同时加一个 KL 散度约束防止模型跑太远：

$$
\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y|x)} \left[ r_\phi(x, y) \right] - \beta \mathbb{D}_{\text{KL}} \left[ \pi_\theta(y \mid x) \| \pi_{\text{ref}}(y \mid x) \right]
$$

这里 $\pi_{\text{ref}}$ 就是 SFT 模型，$\beta$ 控制 KL 惩罚力度。这个目标的意思是：**最大化 reward 的同时不要和原模型偏离太多**。

在 RLHF 中，这一步通常使用 PPO（Proximal Policy Optimization）算法来完成。PPO 的核心是通过 clipped surrogate objective 限制每次策略更新的幅度，在保持训练稳定性的同时只需要一阶优化器（SGD/Adam），相比 TRPO 大幅简化了实现。详细的 PPO 原理参见下方 PPO 章节。

PPO 在 RLHF 语境下需要维护四个模型：policy model、reference model、reward model、value model，这个复杂性正是 DPO 试图解决的核心痛点。

## DPO 的核心推导

DPO 最精彩的地方在于其推导过程，通过一个巧妙的变量替换，把上面的 RL 问题变成了一个简单的分类问题。

### 第一步：写出 RL 目标的最优解

上面的 KL 约束 reward 最大化问题，其最优策略有闭式解：

$$
\pi_r(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)
$$

其中 $Z(x) = \sum_y \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)$ 是归一化常数（partition function）。

直觉理解：最优策略就是在参考模型的基础上，按 reward 的指数进行加权——reward 高的回答概率变大，reward 低的回答概率缩小。

### 第二步：反解出 reward

把上面的公式两边取对数，反解出 reward：

$$
r(x, y) = \beta \log \frac{\pi_r(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)
$$

**这是 DPO 最关键的一步**：reward 可以用最优策略和参考策略的对数概率比来表示。

### 第三步：代入 Bradley-Terry 模型

把上面的 reward 代入 Bradley-Terry 偏好模型，由于 $Z(x)$ 只和 $x$ 有关（和 $y$ 无关），在做差的时候会被消掉：

$$
p^*(y_1 \succ y_2 \mid x) = \frac{1}{1 + \exp\left(\beta \log \frac{\pi^*(y_2|x)}{\pi_{\text{ref}}(y_2|x)} - \beta \log \frac{\pi^*(y_1|x)}{\pi_{\text{ref}}(y_1|x)}\right)}
$$

也就是说，**人类偏好概率完全可以用最优策略和参考策略来表示，不需要显式的 reward model**。

### 第四步：得到 DPO 损失函数

现在把未知的最优策略 $\pi^*$ 换成我们要训练的参数化策略 $\pi_\theta$，用极大似然来优化，就得到了 **DPO 损失函数**：

$$
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]
$$

这个损失函数非常直观：
- 它衡量的是模型对"好回答"相对于"差回答"的**偏好程度**
- $\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)}$ 是好回答相对于参考模型的隐式 reward
- $\beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}$ 是差回答相对于参考模型的隐式 reward
- 损失要求好回答的隐式 reward 显著高于差回答的隐式 reward

## DPO 梯度的直觉理解

DPO 损失的梯度可以写成：

$$
\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta \mathbb{E} \left[ \underbrace{\sigma(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w))}_{\text{隐式 reward 排序错误时权重更大}} \left[ \underbrace{\nabla_\theta \log \pi_\theta(y_w \mid x)}_{\text{提高好回答概率}} - \underbrace{\nabla_\theta \log \pi_\theta(y_l \mid x)}_{\text{降低差回答概率}} \right] \right]
$$

其中 $\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ 是语言模型隐式定义的 reward。

梯度的含义是：
1. **提高好回答 $y_w$ 的概率，降低差回答 $y_l$ 的概率**
2. 前面的权重 $\sigma(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w))$ 起到**动态加权**的作用——当模型还没学好（隐式 reward 排序错误）时，这个权重大，梯度更强；当已经学得差不多了，权重小，梯度就弱了
3. 这个加权机制防止了模型退化，如果去掉这个权重只用朴素的概率比，模型会崩掉（论文实验验证了这一点）

## DPO 的理论保证

### 你的语言模型就是 Reward Model

论文证明了一个重要定理：在 Plackett-Luce（包括 Bradley-Terry）偏好框架下，**所有与之一致的 reward 等价类都可以用如下形式表示**：

$$
r(x, y) = \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}
$$

也就是说，DPO 的重参数化不会丢失任何表达能力。每训练一个策略 $\pi_\theta$，就等价于隐式地拟合了一个 reward model。

### 对 PPO 不稳定性的解释

论文还用 DPO 框架解释了 PPO 训练不稳定的原因：PPO 的目标中有一个归一化项 $\log \sum_y \pi_{\text{ref}}(y|x) \exp(\frac{1}{\beta} r_\phi(x,y))$，这个项需要用 value function 来估计或用基线来近似，估计不准就会导致高方差和不稳定。而 DPO 的重参数化天然消除了这个归一化项。

## DPO 流程总结

相比 RLHF 的三阶段流程，DPO 简化为两步：

| 步骤 | RLHF | DPO |
|------|------|-----|
| 1 | 偏好数据 → 训练 Reward Model | 偏好数据 → 直接优化策略 |
| 2 | Reward Model + PPO → 优化策略 | （不需要） |

具体来说：
1. **准备偏好数据**：收集 $(x, y_w, y_l)$ 三元组
2. **直接优化**：用 $\mathcal{L}_{\text{DPO}}$ 对语言模型做梯度下降，参考模型 $\pi_{\text{ref}}$ 固定不动

不需要训练 reward model，不需要采样，不需要 PPO 的各种 trick（value function、GAE、clipping 等）。

## 实验结果

论文在三个任务上验证了 DPO 的效果：

### 受控情感生成（IMDb）
- 使用 GPT-2-large，控制影评生成的情感倾向
- DPO 在所有 KL 预算下都取得了最高的 reward，**严格优于 PPO**，甚至优于用真实 reward function 训练的 PPO-GT

### 摘要生成（TL;DR）
- 使用 GPT-J 6B 模型
- DPO 在温度 0 时 win rate 约 61%，超过 PPO 最优温度时的 57%
- DPO 对采样温度的变化更鲁棒

### 单轮对话（Anthropic HH）
- 使用 Pythia 2.8B 模型
- DPO 是唯一一个在所有温度下都超过数据集首选回答的方法
- DPO 收敛速度快，训练稳定

### 分布外泛化
- 在 TL;DR 上训练的模型，直接在 CNN/DailyMail 新闻数据集上测试
- DPO 的泛化能力优于 PPO

## 实现细节

DPO 的 PyTorch 实现非常简洁：

```python
import torch.nn.functional as F

def dpo_loss(pi_logps, ref_logps, yw_idxs, yl_idxs, beta):
    pi_yw_logps, pi_yl_logps = pi_logps[yw_idxs], pi_logps[yl_idxs]
    ref_yw_logps, ref_yl_logps = ref_logps[yw_idxs], ref_logps[yl_idxs]

    pi_logratios = pi_yw_logps - pi_yl_logps
    ref_logratios = ref_yw_logps - ref_yl_logps

    losses = -F.logsigmoid(beta * (pi_logratios - ref_logratios))
    rewards = beta * (pi_logps - ref_logps).detach()
    return losses, rewards
```

默认超参数：$\beta = 0.1$，batch size 64，RMSprop 优化器，学习率 1e-6，warmup 150 步。

## 局限性与展望

1. **分布外泛化**：DPO 策略在分布外的泛化能力需要更多研究
2. **自标注**：能否用 DPO 策略自身的生成来做自标注训练？
3. **Reward 过优化**：训练后期 win rate 略微下降，可能存在过优化问题
4. **规模扩展**：论文实验最大到 6B 参数，更大规模的效果有待探索
5. **多模态**：DPO 的思路可以推广到其他生成模型的偏好优化

## 个人总结

DPO 是一篇非常优雅的工作，核心贡献是一个巧妙的数学推导——发现 RLHF 的 RL 优化目标可以通过变量替换，直接变成一个关于策略模型的分类损失。这个发现让偏好对齐从"RL 问题"变成了"监督学习问题"，大幅降低了实现和训练的复杂度。

DPO 的成功说明了一个深刻的道理：**有时候看似需要复杂流程的问题，换一个角度重新建模，可能有非常简洁的解法**。RLHF 需要四个模型（policy、reference、reward、value），而 DPO 只需要两个（policy 和 reference），效果还更好。

DPO 发表后迅速成为偏好对齐的主流方法之一，催生了大量后续工作（如 IPO、KTO、ORPO 等），也成为当前大模型训练 pipeline 中的标配组件。
