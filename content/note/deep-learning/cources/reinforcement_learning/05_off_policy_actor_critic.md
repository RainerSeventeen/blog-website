# 05 离策演员评论家

> 本章节详细介绍 PPO 和 SAC 这类实用算法，并介绍实用离线策略方法让他们更加高效

## 现有问题

基于我们上一章节的优化公式
$$
\nabla_\theta J(\theta)
\approx
\sum_{t,i}
\nabla_\theta
\log \pi_\theta(a_{t,i} \mid s_{t,i})
\hat A^{\pi_\theta}(s_{t,i},a_{t,i})
$$
如果使用 importance weight 来对更重要的样本进行加权，则有：
$$
\nabla_{\theta'}J(\theta')
\approx
\sum_{t,i}
\frac{\pi_{\theta'}(a_{i,t}|s_{i,t})}
{\pi_\theta(a_{i,t}|s_{i,t})}
\nabla_{\theta'}\log \pi_{\theta'}(a_{t,i}|s_{t,i})
\hat{A}^{\pi_\theta}(s_{t,i},a_{t,i})
$$


可以发现**优势函数** $\hat{A}^{\pi_\theta}$ **是基于旧策略** $\pi_\theta$ **和旧数据估计出来的**，如果对同一批的数据优化很多不，新策略会逐渐偏离旧策略，此时优势函数就会偏离实际的训练目标了，会导致过拟合的问题。

### KL 散度

对于这个问题，可以尽可能让新策略和旧策略更加接近，从而降低偏离的情况。

对同一个状态 $s$，旧策略和新策略的动作分别是 $\pi_\theta(\cdot|s)$ 和 $\pi_{\theta'}(\cdot|s)$ ，可以记 KL 散度为：
$$
D_{\mathrm{KL}}(\pi_\theta(\cdot|s) \| \pi_{\theta'}(\cdot|s))
=
\sum_a
\pi_\theta(a|s)
\log
\frac{\pi_\theta(a|s)}{\pi_{\theta'}(a|s)}
$$
实际表示意义为 **KL 越小，新旧策略在同一个状态下给出的动作概率分布越接近。**

在使用了 KL 散度后优化目标为只允许在就策略的小区域内进行更新。

在原始的 $J(\theta)$ 上减去一个 KL 散度的值就可以实现控制更新范围：
$$
J(\theta) - \beta D_{\mathrm{KL}}
$$
这是 PPO 的一个变体

### 权重裁剪

对 importance weight 进行裁剪，控制 $\frac{\pi_{\theta'}(a_{i,t}|s_{i,t})}
{\pi_\theta(a_{i,t}|s_{i,t})}$ 数值的大小到某个固定的区间内。

这也是 PPO 中的一个核心技巧，参考下文。

## Proximal Policy Optimization (PPO)

PPO 的核心目标函数就是从普通的 policy gradient 中优化出来的

原始的梯度策略目标的梯度为
$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{s_t,a_t\sim \pi_\theta}
\left[
\nabla_\theta \log \pi_\theta(a_t|s_t) A^{\pi_\theta}(s_t,a_t)
\right]
$$
最大的问题就是 优势函数是从旧的 数据算的，所以 PPO 的核心就是为了改进这个问题

PPO 中的代替目标为
$$
\tilde{J}(\theta')
\approx
\sum_{t,i}
\frac{\pi_{\theta'}(a_{i,t}|s_{i,t})}
{\pi_{\theta}(a_{i,t}|s_{i,t})}
\hat{A}^{\pi_\theta}(s_{t,i},a_{t,i})
$$
其中优势函数前面那个加权的因子称为 **importance weight** / **probability ratio**，用于衡量新策略对某个动作的概率改变量。
$$
r_t(\theta')
=
\frac{\pi_{\theta'}(a_t|s_t)}
{\pi_\theta(a_t|s_t)}
$$
这个加权因子虽然能够对让策略更新在旧的策略附近，但是乘法会引入数值的不稳定。

### Clip Importance Weight

对 importance weight 的值进行裁剪，用来控制训练时候的稳定性
$$
\tilde{J}(\theta')
\approx
\sum_{t,i}
\operatorname{clip}
\left(
\frac{\pi_{\theta'}(a_{i,t}|s_{i,t})}
{\pi_{\theta}(a_{i,t}|s_{i,t})},
1-\epsilon,
1+\epsilon
\right)
\hat{A}^{\pi_\theta}(s_{t,i},a_{t,i})
$$
例如设置一个区间 $[0.8, 1.2]$ ，限制新策略对某个动作相对于旧策略的概率比例在区间内。

但此时 clip 有可能会让目标函数变得比原来值更大，对于优化过程其实是错误的，所以还需要额外的改进。

注意实际计算中，**优势函数本身的值被当做一个常数（不参与梯度计算）**，所以一旦发生了裁剪则整个一项的梯度都变成了 0。

### Take Minimum

为了进一步控制梯度，PPO 算法额外再取一次 clip 后的值和原来的值之间更小的那个
$$
\tilde{J}(\theta')
\approx
\sum_{t,i}
\min
\left(
r_t(\theta')\hat{A}_t,
\operatorname{clip}(r_t(\theta'),1-\epsilon,1+\epsilon)\hat{A}_t
\right)
$$

1. 在 $\hat A_t>0$ 时，为了控制优化的策略不要太激进，所以使用上界 $(1+\epsilon)\hat A_t$ 来控制，此时最小值就是 $(1+\epsilon)\hat{A}_t$ 
2. 在 $\hat{A}_t < 0$ 时，例如 $r_t\hat A_t=1.5\times(-10)=-15$，把 因数 clip 到 $1.2$ 会得到 $-12$，反而让目标函数变大了，如果取最小值则还是那个 $-15$

在此情况下不再是一旦裁剪就梯度归零，而是还需要看最小值实际取的是哪个值。

### Generalized Advantage Estimation (GAE)

GAE 是用来估计 $\hat A_t$ 的的，就是上一章节中 演员评论家的三种方式，主要包含：

Monte Carlo, Bootstrapping, N step.

## 离策强化

### Replay Buffer

到此为止使用了两种 policy 方式，包括：

1. on-policy: 使用当前的 batch 的数据做一次梯度的 step
2. off-policy: 使用一个 batch 的数据做多个梯度 step

但是还有一些额外的方法可以进一步强化这个 off-policy，也就是从之前的多个 batch 来执行梯度更新，这样流程就会变成：

1. 从策略中收集 experience 并放进 replay buffer：${s_i, a_i} \sim \pi_\theta(a \mid s)$
2. 从 replay buffer 中采样一个 batch：${s_i, a_i, r_i, s_i’} \sim \mathcal{R}$
3. 使用 TD 目标更新价值函数：$y_i = r_i + \gamma \hat{V}\phi^\pi(s_i’)$，并用该目标更新 $\hat{V}\phi^\pi$
4. 计算优势函数估计：$\hat{A}^\pi(s_i, a_i) = r(s_i, a_i) + \gamma \hat{V}\phi^\pi(s_i’) - \hat{V}\phi^\pi(s_i)$
5. 估计策略目标函数的梯度：$\nabla_\theta J(\theta) \approx \frac{1}{N}\sum_i \nabla_\theta \log \pi_\theta(a_i \mid s_i)\hat{A}^\pi(s_i, a_i)$
6. 根据梯度更新策略参数：$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$

对第 3 步中的 $y_i = r_i + \gamma \hat{V}\phi^\pi(s_i’)$ 算法中如果使用这个作为 bootstrapping 算法来更新 $\hat{V}\phi^\pi$ ，那么这个估计实际上拟合的并不是某一个策略的 value function。

因此对此可以有这几个方法：

1. 加入权重，例如时间越近的数据则权重更大等
2. 只允许当前的 policy 用来作为 value function 的学习标签

3. 使用 Q function 而不是 value function （这是最常用的）

### 使用 Q-function

回顾一下 Q-function 的公式，在状态 $s$ 下，先执行动作 $a$，之后再按照当前策略 $\pi_\theta$ 行动，最终能获得多少累计回报：
$$
Q^{\pi_\theta}(s,a)
=
r(s,a)
+
\gamma
\mathbb{E}_{s' \sim p(\cdot|s,a), a' \sim \pi_\theta(\cdot|s')}
[
Q^{\pi_\theta}(s',a')
]
$$
这里第一个动作 $a$ 是从 旧策略中采样的，而新动作是当前策略采样的。

并且这个式子中，第二项中没有使用当前的 policy 的数据以外的信息，所以可以离策训练。

也就是说用 replay buffer 里的旧经验 $(s,a,r,s')$ 训练当前策略 $\pi_\theta$ 的 Q 函数时，可以把旧动作 $a$ 当作第一步动作，然后在下一状态 $s'$ 处接上当前策略采样的动作 $a'$，用 Bellman target 构造监督信号训练 critic。



















