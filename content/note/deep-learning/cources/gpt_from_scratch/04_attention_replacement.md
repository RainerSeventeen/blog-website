# 04 Attention 替代与 MOE

## Attention 优化

随着 Sequence Length 的增长，Attention 的计算量是平方级别增长的。

当上下文变长时，Attention 计算量会快速超过 FFN 的计算量。

### Linear Attention

实际上已经有一种投入使用的算法，将复杂度转为线性。

假设 $Q\in\mathbb R^{n\times d_k}, K\in\mathbb R^{n\times d_k}, V\in\mathbb R^{n\times d_v}$ ，并对他们计算普通的 Attention：
$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$
 在 $QK^T$ 阶段就会有 $O(n^2d_k)$ 的复杂度，在 $(QK^T)V$ 阶段则为 $O(n^2d_v)$，因此总体是$O(n^2d_k+n^2d_v)$ 的计算复杂度。

但是如果我们假设 softmax 的运算是单位的，因此可以拆开括号并应用乘法结合律$(AB)C=A(BC)$ 改成 $(QK^T)V=Q(K^TV)$

首先计算 $K^TV$ ，已经没有了 $n\times n$ 的矩阵乘法，复杂度降低到 $O(nd_kd_v)$

随后是 $Q(K^TV)$ ，同样也也只有 $O(nd_kd_v)$ ，因此整个算法复杂度降低到 $O(nd_kd_v)$ 的级别

这一点有点类似于 RNN 的网络，维护一个横跨时间递推的状态量，这个状态由当前输入以及之前的状态来过程。结合 FFN 的性质可以得到 $S_t=S_{t-1}+k_tv_t^T$ 以及 $y_t=q_t^T S_t$。

### Mamba

>  Nemotron 3 中使用了 Mamba 2 的结构，他是一个小规模的开源模型。

Mamba-2 中做了一个改进，加入一个门控控制哪些内容会被遗忘，而不是全部都塞进 state 中去。
$$
S_t=\gamma_tS_{t-1}+k_tv_t^T
$$

$$
y_t=q_t^TS_t
$$

在经历一系列推导后（展开递推式子，并代入变量）可以得到：
$$
y_t
=
\sum_{i\le t}
(q_t^Tk_i)
\left(
\prod_{j=i+1}^{t}\gamma_j
\right)v_i
$$
又回到了 Attention 的形式，称为 structured causal attention。

因此 Mamba 可以同时做到两种架构的优点：训练时候使用 Attention 的并行化，推理时候使用 RNN 的低存储消耗（不需要 KV cache 而是直接一个 state 即可）。

在 Mamba 的基础上还有 Gated Delta Net 的改进。

### Spare Attention

稀疏注意力，也称为 Deepseek Spare Attention，从全部的上下文中筛选出一部分来进行完整的 Attention。

实际做法中，训练模型只用普通的 Attention。在训练更长的上下文的时候，使用这个方法，让模型去适配更长的上下文。

## MOE

MOE 并没有本质上的改变，约等于一个更加高级的 MLP。将 FFN 改成更多的 FFN，但是每一次的 forward 或者 backward 都只激活其中一个。

MOE 让模型的参数量可以变得非常大，但是计算量（包括推理和训练）都不会有特别大的提升。
另外一个好处就是系统性的优化，MOE 自然而然地支持部署在多个机器上，这对大参数模型比较友好。

但是 MOE 的难点就在于训练，如何去让模型学会路由。同时并行部署等等都是问题，这也是为什么 MOE 在 2022 年被提出，但是 2024 年才流行起来。

### Routing Function

最核心的部分，如何将不同的 token 递交给不同的 expert ？

可以有 token 选择 expert，也可能让 expert 选择 token。实际上绝大多数都是用 token 来选择 expert。

1. Top-k：使用一个类似于 projector 的东西，计算和所有 expert 内积和距离，然后来选择 expert，这是最常见的用法。
2. Hashing：对 token 使用 hash 然后直接分配，实际部署中不常见。
3. RL 学习 route，可以用 bandit 算法等，不常用，因为会比较麻烦。
4. 线性分配算法，成本比较高，但是理论证明有效，尚未大规模应用。

关于最常见的 top-k 算法如下：

首先计算 token 和各个 expert 的分数，然后 softmax 算概率。
$$
s_{i,t}
=
\operatorname{Softmax}_i
\left(
{\mathbf u_t^l}^{T}\mathbf e_i^l
\right)
$$
使用 top-k 算法来选择只激活部分专家（不选择按照分数权重激活就是 MOE 减少运算量的关键）
$$
g_{i,t}
=
\begin{cases}
s_{i,t},
&
s_{i,t}\in
\operatorname{TopK}
(\{s_{j,t}\}_{j=1}^{N},K)
\\
0,
&
\text{otherwise}
\end{cases}
$$
最后就可以计算得到分数（实际上只会对 k 个求和，不会对剩下的 0 权重计算累加）
$$
\sum_{i=1}^{N}
g_{i,t}FFN_i(u_t)
$$
Deepseek 还提出了 shared experts，也就是对所有的 token 都应该存在的通用处理，这些 expert 用于会被激活。这也是很常见的一种想法

### Training

MOE 的训练会很困难，但是通过一系列的 trick 可以让训练稳定。

现实中主要使用 启发式的平衡 loss（heuristic balancing loss）

在训练过程中会有一种现象，如果使用普通的梯度下降的话，其中的更强的专家会被 更新为有更强的权重，强者愈强导致有的专家会 ”饿死“。因此也就需要一个负载均衡的方法。

在 Switch Transformer 中就使用了 balancing loss，目标是尽可能让不同的 expert 的 token 负载均衡：
$$
L_{\text{balance}}
=
\alpha N
\sum_{i=1}^{N}
f_i P_i
$$
其中 $f_i$ 表示专家收到的 token 数量，下式中 $T$ 是总量，右边是实际选择的 expert
$$
f_i
=
\frac{1}{T}
\sum_{x\in B}
\mathbf 1
\{
\arg\max p(x)=i
\}
$$
并使用 $P_i$ 表示 Router 实际上试图给 Expert 分配的概率。
$$
P_i
=
\frac{1}{T}
\sum_{x\in B}
p_i(x)
$$
注意到 $P_i$ 是没有取最大值的，也就是 softmax 本身的计算结果。

通过这个 loss 可以压制最热门的 expert，让 expert 的负载均衡；另一方面，也可以额外用这个思想套用在不同的设备上，让同一批训练中不同计算机的训练也更加均衡。

这样在选择专家这个信号上，可以在训练的时候绕过 top-k 这个不可微分的操作，使用 balance loss 来监督学习的方向。同时只有真正激活的 expert FFN 才会收到 token 的梯度，正常训练即可。