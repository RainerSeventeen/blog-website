# 第二次作业

> 详细作业代码参见 [GitHub 仓库](https://github.com/RainerSeventeen/deep-learning-playbook/tree/main/courses/reinforcement-learning/91_homework_2)

## Problem 1

$$
Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha \left[ r_t + \gamma \max_{a'} Q(s_{t+1},a') - Q(s_t,a_t) \right]
$$

这是 Q leaning 的核心公式，其中右边括号里的是 TD error，表示新观察到的目标价值 和 旧估计值 之间的差距。

具体来说，$\alpha$代表学习率，$r_t$ 表示执行动作 $a_t$ 之后得到的奖励，$\gamma$决定未来奖励的重要程度，而 

$\max_{a'} Q(s_{t+1},a')$  表示到达下一个状态 $s_{t+1}$ 后，假设会选择价值最高的那个动作。

使用固定的表格来记录所有的 Q 值，这是运行的结果：

![](http://oss.rainerseventeen.cn/blog/2026/202606011127834.png)

在这个项目中分别对应三种场景：

- `scenario_1`：每走一步扣 1，但 goal 1 奖励更高，所以 agent 应该愿意多走几步去更远的 goal 1。
- `scenario_2`：每走一步扣 2，走路代价更大，所以虽然 goal 1 奖励高，agent 反而应该去更近的 goal 2。
- `scenario_3`：每走一步奖励 +1，两个 goal 只有 +1，进入 goal 会终止 episode，所以 agent 最优行为是拖着不进 goal，直到 timeout。

运行的结果也是符合这种行为的。

## Problem 2

本题目主要集中于部署 GAE 的 TE Error 算法以及 PPO 中的 clip 算法两大部分，也就是最核心的部分。

相关的理论部分可直接查看第 5 章离策演员评论家的内容，代码参照 GitHub 仓库，这里只放出结果。

这里是 W&B 训练的成功率截图，相对还是比较低的，后期大约在 20% 到 30% 的区间内，最好表现是 40% 左右

![](http://oss.rainerseventeen.cn/blog/2026/202606020906858.png)

## Problem 3

Problem 3 分成三个部分：BC、Actor 更新和 Critic 更新。

### Behavior Clone

首先是 BC 模块。这里的动作是多维连续向量，每个动作维度都有对应的概率密度，因此需要把各维度的 log probability 相加：

在 `off_policy.py` 的 BC 预训练中，常见实现是：

```python
dist = self.actor(obs)
loss = -dist.log_prob(action).sum(-1, keepdim=True).mean()
```

这里的核心目标是让 actor 在状态 $s$ 下给专家动作 $a$ 更高的概率，也就是最大化：

$$
\log \pi_\theta(a \mid s)
$$

训练时一般写成最小化负对数似然：

$$
L_{\mathrm{BC}}(\theta) = - \log \pi_\theta(a \mid s)
$$

如果动作是多维连续动作，例如：

$$
a = (a_1, a_2, \cdots, a_d)
$$

策略网络输出的是一个多维分布。通常实现里会假设各个动作维度在给定状态后条件独立，因此：

$$
\pi_\theta(a \mid s)
= \prod_{j=1}^{d} \pi_\theta(a_j \mid s)
$$

两边取对数后，乘法变成加法：

$$
\log \pi_\theta(a \mid s)
= \sum_{j=1}^{d} \log \pi_\theta(a_j \mid s)
$$

所以代码中的：

```python
dist.log_prob(action).sum(-1, keepdim=True)
```

就是把一个样本中每个动作维度的 log probability 加起来，得到整个动作向量的 log probability。

假设 batch size 是 $B$，动作维度是 $d$，那么维度变化是：

```python
dist.log_prob(action)          # [B, d]
.sum(-1, keepdim=True)         # [B, 1]
.mean()                        # scalar
```

其中：

- `dist.log_prob(action)` 给出每个样本、每个动作维度的 log probability。
- `.sum(-1, keepdim=True)` 沿最后一个维度，也就是动作维度求和，得到每个样本的整体 log probability。
- `.mean()` 对 batch 中的 $B$ 个样本取平均，得到一个标量 loss，方便反向传播。

因此完整的 batch loss 可以写成：

$$
L_{\mathrm{BC}}(\theta)
= -\frac{1}{B}
\sum_{i=1}^{B}
\sum_{j=1}^{d}
\log \pi_\theta(a_{i,j} \mid s_i)
$$

前面的负号表示：我们实际优化器是在最小化 loss，但目标效果是最大化专家动作在当前策略下的概率。

### Critic 更新

Critic 的作用是估计在状态 $s_t$ 下执行动作 $a_t$ 后能够获得的长期回报：

$$
Q_\phi(s_t,a_t)
$$

训练数据从 replay buffer 中采样，每个 batch 包含：

```python
obs, action, reward, discount, next_obs
```

这里使用的是 3-step return。`replay_buffer.py` 已经把连续三步的奖励累积到 `reward` 中，也把终止信号和 $\gamma^3$ 合并到了 `discount` 中，因此 Critic 中不需要再额外乘一次 $\gamma$，目标值直接写成：

$$
y_t
=
r_t^{(3)}
+
d_t^{(3)}Q_{\bar{\phi}}(s_{t+3},a_{t+3}')
$$

其中 $a_{t+3}'$ 从当前 actor 中采样：

```python
next_dist = self.actor(next_obs)
next_action = next_dist.sample(clip=self.stddev_clip)
```

`stddev_clip` 会限制采样噪声，避免 target 因动作扰动过大而产生较高方差。

代码没有只使用一个 Critic，而是维护了一个 Q 网络集合。计算 target 时，从 target critic ensemble 中随机选择两个 Q 值并取较小值：

```python
target_qs = self.critic_target(next_obs, next_action)
target_q1, target_q2 = random.sample(target_qs, 2)
next_q = torch.min(target_q1, target_q2)
target_q = reward + discount * next_q
```

对应公式为：

$$
y_t
=
r_t^{(3)}
+
d_t^{(3)}
\min
\left(
Q_{\bar{\phi}_i}(s_{t+3},a_{t+3}'),
Q_{\bar{\phi}_j}(s_{t+3},a_{t+3}')
\right)
$$

如果直接使用较大的 Q 值作为 bootstrap target，估计误差容易被不断放大，产生 Q 值过估计。取两个估计中的较小值是一种较保守的 clipped double Q 方法。

Target 的计算放在 `torch.no_grad()` 中，因为它只是监督 Critic 的固定标签，不需要更新 actor 和 target critic。

随后让所有 online critics 拟合同一个 target：

```python
current_qs = self.critic(obs, action)
critic_loss = sum(F.mse_loss(q, target_q) for q in current_qs)
```

如果共有 $N$ 个 Critic，则损失为：

$$
L_Q(\phi)
=
\sum_{k=1}^{N}
\frac{1}{B}
\sum_{b=1}^{B}
\left(
Q_{\phi_k}(s_b,a_b)-y_b
\right)^2
$$

随机抽取两个网络只发生在 target 的构造阶段，训练时仍然会更新全部 $N$ 个 online critics。这样既能利用双 Q 的保守目标，也能让整个 ensemble 都持续学习。

### Target Critic 的软更新

Critic 完成一次梯度更新后，target critic 不直接复制最新参数，而是进行软更新：

```python
utils.soft_update_params(
    self.critic,
    self.critic_target,
    self.critic_target_tau,
)
```

其公式为：

$$
\bar{\phi}
\leftarrow
\tau\phi+(1-\tau)\bar{\phi}
$$

本题中 $\tau=0.005$，所以 target critic 只缓慢跟随 online critic。这样 Bellman target 不会随着 Critic 的每次更新剧烈变化，能够缓解“用一个快速变化的网络监督自身”造成的训练不稳定。

### Actor 更新

Actor 更新不再使用 replay buffer 中记录的动作作为标签，而是让当前策略重新采样动作，并通过 Critic 判断这个动作的价值：

```python
dist = self.actor(obs)
action = dist.sample(clip=self.stddev_clip)
qs = self.critic(obs, action)
actor_q = torch.stack(qs, dim=0).mean()
actor_loss = -actor_q
```

对应目标为：

$$
L_\pi(\theta)
=
-
\frac{1}{N}
\sum_{k=1}^{N}
Q_{\phi_k}
\left(
s,\tilde a
\right),
\qquad
\tilde a \sim \pi_\theta(\cdot\mid s)
$$

优化器执行的是梯度下降，所以在平均 Q 值前添加负号；最小化 `actor_loss` 就等价于最大化 Critic 对策略动作的价值估计。

这里使用 `TruncatedNormal.sample()` 生成动作。该分布的采样保留了从动作到分布参数的可微路径，因此梯度可以按照下面的方向传播：

$$
Q(s,\tilde a)
\longrightarrow
\tilde a
\longrightarrow
\pi_\theta(s)
$$

Critic 在 Actor 更新中相当于一个可微的评价器。虽然前向与反向计算会经过 Critic，但代码只调用 `actor_opt.step()`，因此这一步只改变 Actor 参数。

使用全部 Critic 的平均值，而不是单个 Critic 的输出，可以减弱某一个 Q 网络估计噪声对策略更新方向的影响。

### 完整训练流程

`train_off_policy.py` 将 BC 和强化学习更新组合在一起：

1. 先用 demonstration 执行 2000 次 BC 更新，让策略在稀疏奖励任务中获得一个可用的初始行为。
2. 与环境交互，并把 transition 写入 replay buffer。
3. 每个环境 step 从 replay buffer 采样数据更新 Critic。
4. 超过 2000 个 warmup frame 后开始更新 Actor。
5. 默认每隔 2 个 step 再使用专家数据执行一次 BC，避免强化学习微调时策略过快偏离有效行为。

其中 `utd` 是 update-to-data ratio，表示每获得一个新的环境 step，要执行多少次 Critic 更新。默认配置为：

```text
num_critics = 2
utd = 1
```

高 UTD 实验则使用：

```text
num_critics = 10
utd = 5
```

更高的 UTD 会让同一批环境经验被重复利用，提高 sample efficiency；更多 Critic 也能提供更丰富的价值估计。但代价是每个环境 step 需要更多计算，而且 UTD 过高时也可能让 Critic 对 replay buffer 中的有限数据过拟合。

### 训练情况

![](http://oss.rainerseventeen.cn/blog/2026/202606130845093.png)

观察 `eval/episode_success` 曲线可以发现，成功率能够稳定在 90% 以上，并多次接近 100%。这说明 BC 提供的专家先验解决了稀疏奖励下的初始探索问题，而 replay buffer、Critic ensemble 和 target network 又能够继续利用环境交互数据改进策略。

与 Problem 2 中的 PPO 相比，当前 off-policy actor-critic 的最终成功率明显更高。PPO 是 on-policy 方法，主要依赖当前策略新采集的 rollout；本题的方法则可以重复采样 replay buffer 中的历史 transition，并通过 UTD 对每份环境数据进行多次更新，因此样本利用率更高。PPO 的 clipped objective 对策略更新限制得更保守，训练通常更加稳定，但在成功轨迹稀少的任务中也可能改进较慢；off-policy 方法依赖 Q 值估计，稳定性更敏感，不过 BC、双 Q target 和 target critic 共同降低了这种风险。
