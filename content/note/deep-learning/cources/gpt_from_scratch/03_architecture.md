# 03 架构与超参数

## 1 Transformer 架构改进

原始的 Transformer 到现在已经发生了很多的变化，大体上区别可以在于：Layer Norm 的方法，激活函数，位置编码……以及超参数

发展流程：各种 Transformer -> Lamma 2 的各种微调 -> 各种各样的

### 1.1 Layer Norm 的位置

在 原始的 Attention 论文中，LN 是放在 残差连接后面的，这就是 Post Norm 。 

尽管现在模型的架构有非常多种，但是大部分的现代 LLM 都会把 LN 放到了残差连接的外面，比如说放在 多头注意力 的之前。

![残差连接的传输路线](http://oss.rainerseventeen.cn/blog/2026/202607311500647.png)

如图所示，将 LN 放在了残差连接的外面，可以保证整个残差路径上没有执行除了加法以外的任何操作，此时残差的梯度很好地传播。反过来如果在残差上加上 LN 会导致训练稳定性下降，随着模型层数的叠加容易出现梯度不稳定的情况。

另外如果遇到了训练稳定性问题，可以尝试加上 LN，事实表明这个对训练稳定性有帮助。

### 1.2 LN & RMSNorm

RMSNorm 也是一个非常常见的架构改进。

相对于 LN，RMS 并不会损失表达能力，两者的效果表现差不多，但是 RMSNorm 的计算效果更好。 RMS 的计算中移除了计算平均值这个操作，这个操作需要数据的搬运。

尽管 Norm 的算法的 FLOPs 占比很低，但是该操作的运算强度很低，所以这个阶段 GPU 实际功率就很低，Runtime 相应升高，所以实际运行中个优化一个 RMS 是一个很高收益的操作。

### 1.3 FFN 中的 Bias 偏置项

前馈网络的本质就是两层线性层，很多时候也会将那个偏置项 b 给丢掉，因为实验表明其的实际作用并不大。

### 1.4 Gated Acivation

最初只用 ReLU 来进行训练，在 GPT-3 中使用了 GeLU （使用了高斯改变了接近零点的斜率）

随后出现了 **Gated activation (*GLU)** **门控激活函数**，现代 LLM 大多使用 GLU 的变体。

在原始的 FFN 中运算式为（后面一般还会接一个 ReLU）：
$$
FF(x)=\max(0,xW_1)W_2
$$
GLU 增加一个门控，不再直接让激活函数控制信息的通过，而是额外设置一个可学习的门控参数来控制信息流：

一条路保持 ReLU 的算法 $\max(0,xW_1)$ ，另一条路使用 $xV$ ，随后对两者相乘得到：
$$
FF_{ReGLU}(x)
=
(\max(0,xW_1)\otimes xV)W_2
$$
基于这种思想有 GLU 的其他种类：
$$
FF_{GEGLU}(x)
=
(GELU(xW_1)\otimes xV)W_2
$$
还有 LLaMA 使用的（也是现代 LLM 最常用的）SwiGLU：
$$
FF_{SwiGLU}(x)
=
(SiLU(xW_1)\otimes xV)W_2
, \quad
SiLU(x)=x\sigma(x)
$$
显然使用 GLU **会让参数变多**，所以 FFN 在使用了 GLU 之后一般会调整维度让模型的总参数量不会增加太多。

### 1.5 Position Embedding

原始论文中使用的是 Sine Embedding，后面还有 绝对位置，相对位置编码等，但是在 2024 以后基本上都在用 RoPE，旋转位置编码。

旋转位置编码的本质是一个相对位置函数，利用了向量之间的点积只和两者之间的夹角有关的性质。在 Embedding 中将多维度的向量拆分成两两一组来实现，详细内容参见文章 [Position Encoding](https://note.rainerseventeen.cn/deep-learning/foundations/position-encoding/)
