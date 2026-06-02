# 08 奖励学习

前面的学习中，都假设面向与某种目标来学习，但是现实很多问题中并没有 reward 本身存在。

## 从目标学习

最简单的方法就是收集成功与失败的两种足够的数据，然后训练一个二元的分类器来学习分类是否成功，这个分类器可以作为一个最简单的 reward。

当然这有几个问题：

1. 分类器的结果是离散的，没有足够的学习激励
2. RL 会走火入魔，也就是为了欺骗 reward 而不是真正完成了任务（也就是分类器本身不能完整代表真正的 reward）

对此有个解决方法： example 中正确的情形以外的所有情况，全都标记为 negative，也就是只对见过的 state 的正向奖励负责，这可以防止 RL 学到错误的激励。

这样做的缺点就是不够 robust，并且 positive 和 negative 的数据不够平衡也会导致数据训练问题。

另一种就是 GAN 网络，同时训练分类器和生成器。这样的好处是除了最开始的数据集，可以不再需要人类的标注了。

## 从人类偏好学习

> 这里和 LLM 结合比较紧密，所谓的 RLHF 也就是面向人类偏好进行 RL

对于这种情况，一般会给出几个 trajectory 并让人类评价哪一个更加好。

人类给出偏好的那个轨迹应该有更高的 reward 总和，也就是如果人类偏好 $\tau_w \succ \tau_l$ 则应该有：
$$
\sum_{(s,a)\in \tau_w} r_\theta(s,a)
>
\sum_{(s,a)\in \tau_l} r_\theta(s,a)
$$
也可以吧 reward function 设置为一个神经网络，然后将 preference 作为数据集进行训练。

在实际的训练过程中，通常使用概率来设计这个目标，也就得到了最大似优化就是让下方的 Loss 最小：（其中 $\sigma$ 是 sigmoid 函数。）
$$
P_\theta(\tau_w \succ \tau_l)
=
\sigma(r_\theta(\tau_w)-r_\theta(\tau_l))
$$

$$
\mathcal{L}(\theta)
=
-\log \sigma(r_\theta(\tau_w)-r_\theta(\tau_l))
$$



























