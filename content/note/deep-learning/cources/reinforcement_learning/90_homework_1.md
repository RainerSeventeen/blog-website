# 第一次作业

> 对于这次作业的代码，可以到 [GitHub 仓库](https://github.com/RainerSeventeen/deep-learning-playbook/tree/main/courses/reinforcement-learning) 查看具体实现

## Problem 1

实现一个最普通的 Behavior Clone，也就是训练一个网络去学习专家的行为。

具体的实现是一个 MSE + 一个 MLP 就可以完成。

这样做的问题就是当一种情况下有两种都合理的行为的时候会取平均值，所以会导致效果不佳。

如果运行

```python
python main.py --method bc_reg --env easy
```

不会有任何问题，通过下载视频发现能够很好地通过

对于有两个可以前进的空隙的时候就会出错

```python
python main.py --method bc_reg --env hard
```

这是原始的 log 截取一部分，可以看到会在半途撞墙。

```
Ep1: 252steps (CRASHED), Ep2: 386steps (CRASHED), Ep3: 252steps (CRASHED)
[eval] ep 10/50 (20.5s elapsed, avg_len=279, policy: 0.2s/286 calls = 0.6ms/call, render: 4.6s)
```

## Problem 2

这里用的是 Flow Matching 算法，一个从噪声逐渐导向 expert 行为的一个方法，假定从噪声到 expert 有一个给定的路线。

首先采样一个随机的噪声 $a_{t,0} \sim \mathcal{N}(0,I)$ ，然后通过差值方式给出这个噪声到专家的每一个步的位置，在作业中使用普通的线性插值：
$$
a_{t,\tau} = \tau a_t + (1-\tau)a_{t,0}
$$
也就是在 $\tau = 0$ 时候是纯噪声，在 $\tau = 1$  时候就是专家行为，逐步从噪声推导到专家行为。

训练的模型预测的是**速度**，也就是 $v_\theta(s_t, a_{t,\tau}, \tau)$​ ，我们已知的分别是当前状态 $s_t$，当前中间动作 $a_{t,\tau}$，当前 flow 时间 $\tau$。

在作业中差值模型的路径是一条直线，因为使用的是线性插值：
$$
a_{t,\tau} = a_{t,0} + \tau(a_t - a_{t,0})
$$
这个式对 $\tau$ 求导就是每一个步的速度
$$
\frac{d a_{t,\tau}}{d\tau} = a_t - a_{t,0}
$$
所以该作业中模型需要学习目标的是 $v_\theta(s_t, a_{t,\tau}, \tau) \approx a_t - a_{t,0}$ 

再回头看 Loss：
$$
\mathcal{L}_{FM}(\theta)
=
\frac{1}{|\mathcal{D}|}
\sum_{(s_t,a_t)\in \mathcal{D}}
\left\|
v_\theta(s_t,a_{t,\tau},\tau)
-
(a_t-a_{t,0})
\right\|_2^2
$$
本质上还是一个 MSE 的 Loss，监督和学习的都是速度。

在生成的时候，直接给定状态让模型更新 $n$ 次获取到的就是最终的结果，从 从 $\tau = 0$ 开始直到 $\tau = 1$
$$
a_{t,\tau+\frac{1}{n}}
=
a_{t,\tau}
+
\frac{1}{n}
v_\theta(s_t,a_{t,\tau},\tau)
$$

## Problem 3

### DAgger 算法

普通的 BC 算法会有分布偏移问题，也就是一旦出现了不在数据集中的情况，会逐步偏移到未知的错误位置，构成误差累计。

DAgger 的核心思想是让 Agent 在环境中按照自己的策略跑，把他遇到的状态收集起来，让专家给状态标注动作。

具体来说，首先利用专家数据 $D_0 = \{(s_i, a_i^*)\}$ 训练初始策略 $\pi_1 = \text{train}(D_0)$ 

然后利用这个策略与环境交互生成一串状态 $s_1, s_2, ..., s_T \sim d^{\pi_i}$，对这些状态去询问专家得到数据$D_i = \{(s_t, \pi^*(s_t))\}$ 。

最后把所有数据整合起来重新训练，不断重复这个过程。

## 数据整合

使用官方的已有脚本，直接运行

```bash
python main.py
```

获得所有方法的运行结果配置，然后再运行

```bash
python main.py --plot
```

获取实验最终的结果可以得到：

![](http://oss.rainerseventeen.cn/blog/2026/202605231013577.png)

观察图像可以发现，dagger 的效果应该是最好的，随后就是 bc_flow 算法。

bc_reg 在 hard 模式下（有两个通道）的时候会取平均值导致撞上，所以效果很差。



