# 03 策略梯度

上一节中介绍了模仿学习， 模仿学习实际上永远无法超过 expert，今天的方法可以做到超过 expert 的表现。

##  在线强化学习

Online RL 的大致步骤是：

1. 初始化一个 policy，并运行这个 policy 收集到一些运行的结果
2. 根据这些结果对 policy 执行优化，并反复迭代这个过程

用具体的数学表达式可以表达强化学习中的策略优化目标，找到一组最优策略参数 $\theta^*$ ，使环境中交互得到的期望累计奖励最大。
$$
\theta^{*}
= \arg\max_{\theta} J(\theta),
$$
$$
J(\theta)
=
\mathbb{E}_{\tau \sim p_{\theta}(\tau)}
\left[
\sum_{t} r(s_t, a_t)
\right]
\approx
\frac{1}{N}
\sum_{i=1}^{N}
\sum_{t}
r(s_{i,t}, a_{i,t})
$$
$\theta$ 是策略的参数。第一个式子的意思就是找到让 $J(\theta)$ 最大的那个 $\theta ^*$

第二个式子$J(\theta)$ 代表在策略 $\pi_\theta$ 下，产生的 trajectory  累计奖励的期望值，也就是优化的目标，详细的关于 $s$ 与 $a$ 的说明可以参考之前章节。

## 梯度数学推导

> 下面推导的算法称为  Vanilla policy gradient

优化的式子已经有了，但是难点就在于计算  $J(\theta)$ 的梯度，因为这是一个期望，没有办法通过采样的方式来直接算梯度，所以下面要对这个式子操作一下。

首先根据 $\mathbb{E}_{\tau \sim p_\theta(\tau)} [f(\tau)] = \int p_\theta(\tau) f(\tau)d\tau$ 将期望展开成积分形式，并交换梯度与积分的运算顺序：
$$
J(\theta)
=
\int p_\theta(\tau) r(\tau) d\tau, \quad \mathbb{E}_{\tau \sim p_\theta(\tau)}
[f(\tau)]
=
\int p_\theta(\tau) f(\tau)d\tau
$$

$$
\nabla_\theta J(\theta)
=
\int \nabla_\theta p_\theta(\tau) r(\tau)d\tau
$$

注意这里默认奖励函数 $r(\tau)$ 本身不直接依赖 $\theta$。它依赖的是轨迹，而轨迹分布由策略参数 $\theta$ 决定。

这里用到一个计算技巧，因为 $\log$ 计算的导数就是本身的倒数，再结合链式法则有：
$$
\nabla_\theta p_\theta(\tau)
=
p_\theta(\tau)
\nabla_\theta \log p_\theta(\tau)
, \quad
\nabla_\theta \log p_\theta(\tau)
=
\frac{\nabla_\theta p_\theta(\tau)}
{p_\theta(\tau)}
$$

代入到最开始的式子中于是有：
$$
\nabla_\theta J(\theta)
=
\int
p_\theta(\tau)
\nabla_\theta \log p_\theta(\tau)
r(\tau)d\tau
$$

对 trajectory 展开得到一个连乘的形式，通过对数我们可以转化为求和的形式：
$$
p_\theta(\tau)
=
p(s_0)
\prod_{t=0}^{T}
\pi_\theta(a_t \mid s_t)
p(s_{t+1}\mid s_t,a_t)
$$

$$
\log p_\theta(\tau)
=
\log p(s_0)
+
\sum_{t=0}^{T}
\log \pi_\theta(a_t \mid s_t)
+
\sum_{t=0}^{T}
\log p(s_{t+1}\mid s_t,a_t)
$$

对 $\log p(s_0)$ 和 $\log p(s_{t+1}\mid s_t,a_t)$ 求关于 $\theta$ 的梯度实际上都是 $0$，将结果带回到计算式中有
$$
\nabla_\theta \log p_\theta(\tau)
=
\sum_{t=0}^{T}
\nabla_\theta \log \pi_\theta(a_t \mid s_t)
$$

$$
\nabla_\theta J(\theta)
=
\int
p_\theta(\tau)
\left(
\sum_{t=0}^{T}
\nabla_\theta \log \pi_\theta(a_t \mid s_t)
\right)
r(\tau)
d\tau
=
\mathbb{E}_{\tau \sim p_\theta(\tau)}
\left[
\left(
\sum_{t=0}^{T}
\nabla_\theta \log \pi_\theta(a_t \mid s_t)
\right)
r(\tau)
\right]
$$

将 $r(\tau)$ 记作离散的 trajectory 求和，于是就得到了最常见的 policy gradient 计算形式：
$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{\tau \sim p_\theta(\tau)}
\left[
\sum_{t=0}^{T}
\nabla_\theta \log \pi_\theta(a_t \mid s_t)
R(\tau)
\right]
, \quad
R(\tau)
=
\sum_{t=0}^{T} r(s_t,a_t)
$$
这个式子的实际意义就是，当一个 trajectory 的奖励很高的时候，更新参数会让这个 trajectory 的出现概率上升。

## 和模仿学习的关系

这个算法和 imitation learning 的最大区别就是，不需要一个 expert，可以自己去收集数据。

但是这两个也不是互斥的，可以先用 imitaition learning 学习到一个基准，然后用 policy gradient 进一步优化。

## 策略优化

### 时序因果

这个算法实际上有一点点小问题，在于 $R(\tau)=\sum_{t=0}^{T} r(s_t,a_t)$ 实际上会把过去的所有的行为都加入梯度优化的策略中。

举个例子，假如说使用强化学习让机器人学习走路，那么向后摔倒一步再向前走一小步，这个轨迹中，应该能够给出对向前一小步的单独奖励，而不应该因为向后退了而全盘否定这个轨迹。

因此，可以优化式子的因果性：
$$
\nabla_\theta J(\theta)
\approx
\frac{1}{N}
\sum_{i=1}^{N}
\sum_{t=1}^{T}
\nabla_\theta \log \pi_\theta(a_{i,t}\mid s_{i,t})
\left(
\sum_{t'=t}^{T}
r(s_{i,t'},a_{i,t'})
\right)
$$

### 偏置值

继续考虑机器场景，假如策略是向前的速度，那么可能会同时有多重噪声情况，例如训练机器人中的向前摔一跤、走一步、跑过去。

这些场景都有正向的 reward，他们的对梯度的影响都是正向的。也就是说在某些情况下，机器人可能会选择向前摔一跤而不是选择跑过去。

从深度学习的角度看，这个有点像局部最小值，这种情况下模型的收敛也会相对困难一些。解决这个办法也很简单，就是加上一个 baseline 的偏置：
$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{\tau \sim p_\theta(\tau)}
\left[
\nabla_\theta \log p_\theta(\tau)
\left(
r(\tau)-b
\right)
\right], \quad b=\frac{1}{N}\sum_{i=1}^{N}r(\tau_i)
$$
注意这个并不会改变原始算式的优化目标，因为可以证明  $\mathbb{E} \left[ \nabla_\theta \log p_\theta(\tau)b \right] = 0$ ，也就是不会改变期望的梯度。

简略的证明流程如下：
$$
\mathbb{E}
\left[
\nabla_\theta \log p_\theta(\tau)b
\right]
=
\int p_\theta(\tau)\nabla_\theta \log p_\theta(\tau)b\,d\tau
=
\int p_\theta(\tau)
\frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)}
b\,d\tau
$$
消掉 $p_\theta(\tau)$，提取常数 $b$，并交换积分与求导，即可证明
$$
原式 
=
b\int \nabla_\theta p_\theta(\tau)d\tau
=
=
b\nabla_\theta \int p_\theta(\tau)d\tau
=
b\nabla_\theta 1
=0
$$
这个操作并不会改变期望，是**无偏**（unbaised）的，并且能够**降低梯度的方差（variance）**，方差的证明省略。

## Importance Sampling

> 重要性重采样，是 off-policy policy gradient 的基础

### 问题

经过上节的两轮优化，最终的目标可以记作：
$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{\tau \sim p_\theta(\tau)}
\left[
\sum_{t=1}^{T}
\nabla_\theta \log \pi_\theta(a_t \mid s_t)
\left(
\sum_{t'=t}^{T}
r(s_{t'},a_{t'})
-b
\right)
\right]
$$
算法的总体流程就是：

1. 从 整体的策略中 $\pi _ \theta (a_t | s_t)$ 中采样一个轨迹点 $\tau ^ i$

2. 计算这个轨迹点的  $\nabla _ \theta J(\theta)$ 梯度
3. 梯度更新 $\theta = \theta + \alpha \nabla _ \theta J(\theta)$ 

在第 3 步中更新了 $\theta$ ，也就是 策略 $\pi _ \theta (a_t | s_t)$，那么在计算上面那个优化式子的时候，因为 $\tau \sim p_\theta(\tau)$ 所以整个概率分布全部都改变了，因此也就是说每一步的梯度更新都需要重新采样数据。

这里涉及到一个 online 和 offline 的区分，online 只会使用当前策略的数据来更新策略（策略更新后就需要重新采样）；offline 的更新可以复用其他或者自己过去的数据。我们这里讨论的 Vanilla Policy Gradient 一个 online 算法。

### 基础方案

如果我们没有办法从 $p(x)$ 采样，但可以从另一个分布 $q(x)$ 采样，那么可以用一个小技巧替换到 $q(x)$ 分布下（假设 $q(x) \ne 0$ ）：

$$
\int p(x)f(x)dx
=
\int q(x)\frac{p(x)}{q(x)}f(x)dx
$$

$$
\mathbb{E}_{x\sim p(x)}[f(x)]
=
\mathbb{E}_{x\sim q(x)}
\left[
\frac{p(x)}{q(x)}f(x)
\right]
$$

将这个算法代入到实际的学习目标中可以得到：
$$
J(\theta)
=
\mathbb{E}_{\tau\sim \bar{p}(\tau)}
\left[
\frac{p_\theta(\tau)}{\bar{p}(\tau)}
r(\tau)
\right]
$$
这里 $\frac{p_\theta(\tau)}{\bar{p}(\tau)}$ 表示这条轨迹在当前策略下出现的概率，相对于在旧策略下出现的概率有多大，比值越大表示这个轨迹对当前的策略更重要，否则就更加偏向于旧策略的产物。

关于这个比值的计算，因为可以稍微化简一下，因为初始值的状态都是一致的。
$$
\frac{p_\theta(\tau)}{\bar{p}(\tau)}
=
\frac{
p(s_1)
\prod_{t=1}^{T}
\pi_\theta(a_t|s_t)
p(s_{t+1}|s_t,a_t)
}{
p(s_1)
\prod_{t=1}^{T}
\bar{\pi}(a_t|s_t)
p(s_{t+1}|s_t,a_t)
}
=
\prod_{t=1}^{T}
\frac{
\pi_\theta(a_t|s_t)
}{
\bar{\pi}(a_t|s_t)
}
$$
这里有个隐患，这个权重是非常多的项的连乘，这个值的数值会非常不稳定，导致**分布的方差变大**，在训练中就表现为训练不稳定。

### 对应改进

> 这一节的算法已经很接近 PPO 了

把方案代入可以得到：
$$
\nabla_{\theta'} J(\theta')
=
\mathbb{E}_{\tau\sim p_\theta(\tau)}
\left[
\prod_{t=1}^{T}
\frac{
\pi_{\theta'}(a_t|s_t)
}{
\pi_\theta(a_t|s_t)
}
\left(
\sum_{t=1}^{T}
\nabla_{\theta'}\log \pi_{\theta'}(a_t|s_t)
\left(
\left(
\sum_{t'=t}^{T} r(s_{t'},a_{t'})
\right)-b
\right)
\right)
\right]
$$
上文中也提到了这个连乘的权重数值很不稳定。改进思路就是不再给整条轨迹乘一个统一的轨迹级权重，而是把策略梯度拆成每个时间步的贡献，然后每个时间步只用对应动作的概率比值修正。
$$
\nabla_{\theta'} J(\theta')
\approx
\frac{1}{N}
\sum_{i=1}^{N}
\sum_{t=1}^{T}
\frac{
\pi_{\theta'}(a_{i,t}|s_{i,t})
}{
\pi_\theta(a_{i,t}|s_{i,t})
}
\nabla_{\theta'}
\log \pi_{\theta'}(a_{i,t}|s_{i,t})
\left(
\left(
\sum_{t'=t}^{T}
r(s_{i,t'},a_{i,t'})
\right)-b
\right)
$$
其中 $i$ 就是 第 $i$ 条轨迹，$t$ 是轨迹中的第 $t$ 个 step。

严格上这个式子并不完全严谨，具体来说就是
$$
\frac{
p_{\theta'}(s_t,a_t)
}{
p_\theta(s_t,a_t)
}
=
\frac{
p_{\theta'}(s_t)
}{
p_\theta(s_t)
}
\frac{
\pi_{\theta'}(a_t|s_t)
}{
\pi_\theta(a_t|s_t)
}
$$
这个式子中的 $\frac{ p_{\theta'}(s_t) }{ p_\theta(s_t) }$ 被当成了 1，也就是只考虑动作概率比值，忽略状态分布比值的变化情况。
