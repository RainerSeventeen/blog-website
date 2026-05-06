# 04 演员评论家

> Actor-Critic 算法是 PPO 等语言训练算法的基础

## Value Function & Q-function

Value Function 指的是状态价值函数，计算的是**当前处在状态 s，如果之后一直按照策略 $\pi$ 行动，未来期望能拿到多少累计奖励？**

数学定义为：
$$
V^\pi(s)
=
\mathbb{E}_\pi
\left[
\sum_{t=0}^{\infty}
\gamma^t r_t
\mid s_0=s
\right]
$$
Value 更多的可以评价当前的状态优劣程度，因为计算的是后续的奖励的期望。

相对应的 Q-function 是动作价值函数，计算的是：

**当前处在状态 s，先采取动作 a，之后再按照策略 $\pi$ 行动，未来期望能拿到多少累计奖励？**
$$
Q^\pi(s,a)
=
\mathbb{E}_\pi
\left[
\sum_{t=0}^{\infty}
\gamma^t r_t
\mid s_0=s, a_0=a
\right]
$$
Q-function 主要可以用来在当前状态 $s$ 下采用动作 $a$ 的优劣程度，同时关心状态和动作本身。

##  优化策略梯度

### 使用 Q-function

上一节中我们提到了一个基础的梯度策略算法
$$
\nabla_\theta J(\theta)
\approx
\frac{1}{N}
\sum_{i=1}^{N}
\sum_{t=1}^{T}
\nabla_\theta \log \pi_\theta(a_{i,t}\mid s_{i,t})
\left(
\sum_{t'=t}^{T} r(s_{i,t'},a_{i,t'})
\right)
$$
这里计算的是 $s_t$ 采取了 $a_t$ 之后，这一条具体采样轨迹最终拿到了多少回报，这是一个具体的采样值，如果这里把他替换成 Q-function 则可以表示成：

**在状态 $s_t$ 下采取动作 $a_t$，之后继续按照策略 $\pi$ 行动，未来能获得的期望累计奖励是多少？**

原来的算法（reward-to-go）是一个采样，这个采样的方差很大，也就是对同一个状态可能有非常多的选择，而且轨迹会因为连乘而更加发散，如果使用了 Q-function 则相比一次采样稳定很多。

所以经过优化后可以得到新的梯度策略
$$
\nabla_\theta J(\theta)
\approx
\frac{1}{N}
\sum_{i=1}^{N}
\sum_{t=1}^{T}
\nabla_\theta \log \pi_\theta(a_{i,t}\mid s_{i,t})
Q^\pi(s_{i,t},a_{i,t})
$$

### Baseline 减值

上章节中还会有一个减值 $b$，可以在无偏的情况下优化策略梯度，这里也同样可以使用

一般来说可以用 $V^\pi(s_t)$ 来作为 baseline，表达的在当前状态下使用当前策略的平均 reward

可以定义 Advantage Function（优势函数）：
$$
A^\pi(s_t,a_t)= Q^\pi(s_t,a_t)-V^\pi(s_t)
$$
之前的减值代表的是全局的平均值，它对所有的状态都是同一个值 
$$
b = \text{average reward}
=
\frac{1}{N}
\sum_i Q(s_{i,t},a_{i,t})
$$
使用 $V^\pi(s_t)$  可以针对状态本身，进一步细分平均的收益，减少梯度的噪声。

注意，在实际计算中这个 $A^\pi(s_t,a_t)$ 并不好算。

## 计算期望

对于优势函数可以转化为仅关于 $V ^ \pi$ 的方程：

对于 Q-function 实际上就是第一步使用动作 $a_t$ 然后后面就是一个 value function：
$$
Q^\pi(s_t,a_t)
=
r(s_t,a_t)
+
\mathbb{E}_{s_{t+1}\sim P(\cdot|s_t,a_t)}
[
V^\pi(s_{t+1})
]
$$
所以可以转化优势函数：
$$
A^\pi(s_t, a_t)
\approx
r(s_t, a_t) + V^\pi(s_{t+1}) - V^\pi(s_t)
$$
经过这样的简化，最后的目标就是如何来计算这个 value function，对此有以下几种方法：

### Monte Carlo

训练一个神经网络来预测这个值，也就是训练一个网络来拟合 value function，用非常多的数据来对函数值进行预测拟合。

蒙特卡洛法直接使用完整的 trajectory 作为训练数据，对训练数据 $(s_{i,t}, y_{i,t})$ 使用 神经网络拟合 $\hat{V}^{\pi}_{\phi}(s_{i,t}) \approx y_{i,t}$ 

那么计算 Value 就需要整条轨迹跑完之后的所有数据，所有数据都来源于自当前策略 $\pi_{\theta_k}$

### Bootstrapping

> Bootstrapping learning 又称为 temporal difference learning

理想的用来训练的标签是：
$$
y_{i,t}
=
\sum_{t'=t}^{T}
\mathbb{E}_{\pi_\theta}
[
r(s_{t'},a_{t'})
\mid s_{i,t}
]=
V^\pi(s_{i,t})
$$
指的是从 $s_{i,t}$ （第 $i$ 条轨迹的第 $t$ 个 step）出发，按照当前策略 $\pi_\theta$，未来总回报的期望。

根据 Bellman 思想，当前状态的价值 = 当前一步奖励 + 下一状态的价值
$$
V^\pi(s_t)
=
\mathbb{E}_{a_t \sim \pi, s_{t+1}\sim P}
[
r(s_t,a_t) + V^\pi(s_{t+1})
]
$$
所以可以把理想标签近似为（近似是因为当前 step 的行为 $s_{i, t}$ 已经是一个确切的值了，但是 value function 的计算中实际上是一个期望）：
$$
y_{i,t}
\approx
r(s_{i,t},a_{i,t}) + V^\pi(s_{i,t+1})
$$
但真实的 $V^\pi(s_{i,t+1})$ 也不知道，于是用当前神经网络估计值代替：
$$
y_{i,t}
\approx
r(s_{i,t},a_{i,t}) + \hat{V}^{\pi}_{\phi}(s_{i,t+1})
$$
上式就是 Bootstrapping 的训练标签，其中 $r_t$ 来自于一次环境交互，是一个trajectory 上的 step 的结果

具体对对应一次状态转移 $(s_t,a_t,r_t,s_{t+1})$ 中，$a_t$ 是策略采样的结果，$r_t,s_{t+1}$ 是环境返回的值，使用这一个数据就可以构造一条标签（$\gamma$ 是一个控制大小的因数）
$$
y_t
=
r_t+\gamma \operatorname{stopgrad}(\hat V_{\phi}(s_{t+1}))
$$
这个标签值就是 value function 估计的训练标签，stopgrad 表示构造标签的那一侧不反向传播梯度。

对于 value function 估计网络，一次正向传播预测出来的就结果是 $\hat V_\phi(s_t)$ ，那么损失函数就是：
$$
\mathcal L(\phi)
=
\frac{1}{2}
\left(
\hat V_\phi(s_t)-y_t
\right)^2
$$
这就完成了一次梯度传播流程，对于这个构造反复重复，直到最后会收敛到一个固定的点。

这里做的好处就是，不需要整个轨迹完成后才能对 value funtion 的网络做一次参数更新，而是能够在每一步都执行一次更新。

### N-step 算法

将 Monte Carlo 和 Bootstrapped 方法结合一下，一段轨迹使用蒙特卡洛，一段使用引导学习，就是 N-step 算法。

![](http://oss.rainerseventeen.cn/blog/2026/202605061439968.png)

### Value Function 因子

对于一个 trajectory 非常长，或者理论上无限长的情况下，$V^\pi(s_t)$ 会趋向于无穷，因此可以对这项引入折扣因子$\gamma \in [0,1]$：
$$
y_{i,t}
\approx
r(s_{i,t}, a_{i,t})
+
\gamma \hat V^\pi_\phi(s_{i,t+1})
$$
由于 Bellman 方程本身是递归的，所以展开以后就是：
$$
V^\pi(s_t)
=
r_t
+
\gamma r_{t+1}
+
\gamma^2 r_{t+2}
+
\gamma^3 r_{t+3}
+
\cdots
$$
折扣因子的作用就是鼓励更早的收益，对于同 reward 越早则权重越大。

## 完整算法

![算法流程全图](http://oss.rainerseventeen.cn/blog/2026/202605061452436.png)

到此为止算法已经完整了，这就是 actor-critic algorithm 的整体流程：

1. 从当前的策略中收集一个 batch 的 trajectory 数据 $\{(s_{1,i},a_{1,i},\dots,s_{T,i},a_{T,i})\} \sim \pi_\theta$ 。这批次的数据是 on-policy 的，因为他来自目前正在使用的策略。
2. 训练 critic 模型，用来估计 value function 的值，详细的流程参考上文的 N-step 算法。
3. 使用拟合的网络估计某一个 step 的 value function 的值，并计算 TD error 来近似估计 advantage function

$$
\hat A^{\pi_\theta}(s_{t,i},a_{t,i})
=
r(s_{t,i},a_{t,i})
+
\gamma \hat V^{\pi_\theta}_\phi(s_{t+1,i})
-
\hat V^{\pi_\theta}_\phi(s_{t,i})
$$

4. 计算策略梯度，也就是 actor 更新的方向

$$
\nabla_\theta J(\theta)
\approx
\sum_{t,i}
\nabla_\theta
\log \pi_\theta(a_{t,i} \mid s_{t,i})
\hat A^{\pi_\theta}(s_{t,i},a_{t,i})
$$

5. 最后更新策略参数 $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$ ，反复从头开始循环这个流程。

总结这整个算法就是，**用当前策略采样轨迹，用奖励训练价值函数，用价值函数构造优势函数，再用优势函数加权 policy gradient 更新策略。**
