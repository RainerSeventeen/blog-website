---
paper: "LoRA: Low-Rank Adaptation of Large Language Models"
---

# LoRA

> 论文全名：*LoRA: Low-Rank Adaptation of Large Language Models*

## 简介

LoRA 的全称是 *Low-Rank Adaptation*，低秩矩阵适配，是一种非常通用的大模型微调方案。

在 LoRA 之前，很多的微调工作只能用全参数微调，这对性能的需求是巨大的；要么就是额外引入一个 adapter，这会导致额外的性能损耗，但是 LoRA 同时做到了省计算资源的情况下，没有给正常推理引入延迟。

与此同时我，由于 LoRA 的特性，可以并行跑很多个 LoRA 的微调任务，因为他们共享同一个 pretrain 模型

## 问题建模

对于一个训练好的自回归语言模型 $$P_\Phi(y|x)$$ （给定输入和概率参数，预测输出的建模）

如果下游的数据集是 $\mathcal{Z} = \{(x_i, y_i)\}_{i=1,\dots,N}$ 这个形式的配对，可以得到 full fine-tuning 的目标（核心建模）：

$$
\max_{\Phi} \sum_{(x,y)\in \mathcal{Z}} \sum_{t=1}^{|y|} \log P_\Phi(y_t \mid x, y_{<t})
$$
详细解释一下这个公式：

- $P_\Phi(y_t \mid x, y_{<t})$ 表示在已经知道输入 $x$ 和前面已经生成的 token $y_{<t}$ 的情况下，模型生成当前 token $y_t$ 的概率，这就是自回归生成
- $ \log P_\Phi(y_t \mid x, y_{<t}) $ 就是取了对数，常见的最大似然优化手段，为了把 乘法降维到加法，方便数值优化，因此经过对数以后概率连乘变成了求和 $\log P(y|x) = \sum_{t=1}^{|y|} \log P(y_t \mid x, y_{<t})$
- $\sum_{(x,y)\in \mathcal{Z}} \sum_{t=1}^{|y|}$ 外层两层两层求和，内层就是某一个数据集的所有 token 的概率，最外层是整个数据集的所有样本，所以目标是让模型在整个训练集上，把所有正确输出 token 的概率尽量提上去。

因此整个微调的工作就是：$\Phi_0 + \Delta \Phi$  训练之后叠加一个新的参数上去，但是这个参数量是巨大的，但是从直觉上是没必要执行全量参数更新的，因此作者指出不要直接优化整个 $\Delta \Phi$，能不能用更小的一组参数去“编码”这个变化量，这个更小的参数就是 $\Theta$

最后得到优化目标变成：

$$
\max_{\Theta} \sum_{(x,y)\in \mathcal{Z}} \sum_{t=1}^{|y|} \log P_{\Phi_0 + \Delta\Phi(\Theta)}(y_t \mid x, y_{<t})
$$



## 方法原理

根据前文的推理，对于一个预训练权重$W_0 \in \mathbb{R}^{d \times k}$ ，微调中的 $\Delta W$ 实际上并不一定是满秩矩阵，可以是一个低秩

### 低秩分解

$$
\Delta W = BA
$$

其中，$B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, $r \ll \min(d,k)$，最后权重更新变成了 
$$
W_0 + \Delta W = W_0 + BA
$$
这一步的分解会让总参数量从 $d \times k$ 变成 $dr + rk = r(d+k)$ ，只要 $r$ 很小，就会远小于 $dk$

因此这一步之后，前向传播公式从 $h = W_0x$ 更新为 $h = W_0x + \Delta Wx = W_0x + BAx$

这也是为什么 LoRA 没有额外推理损耗的原因，因为他就是在原来的模型的推理上加了一个简单的偏移量

例如想推理的时候，这个值实际上可以在推理前直接更新到参数矩阵上，提前算好

$$
W = W_0 + BA
$$

也就是完全融合到原来的权重中去了，因此没有任何的延迟

### 初始化与缩放值

在初始化矩阵的时候，$A$ 用高斯随机初始化， $B$ 初始化为 0，因此 $\Delta W = BA = 0$

因此最开始训练模型的时候，和原模型是完全一样的，有点 cosine ramp 缓慢增强 loss 的感觉，目的是让模型先保持原始能力，后续训练逐步学会修正

此外 论文还说，会把 $\Delta Wx$ 再乘一个缩放因子：$\frac{\alpha}{r}$ （这里这 $r $ 是那个秩）
$$
h = W_0x + \frac{\alpha}{r}BAx
$$
有点类似与控制 Adam 中的 LR 一样，简单来说就是调整 $r $ 这个超参数的时候可以自动化更新值，少调整一个超参数（就是设置成和以第一个 $r $ 一样，后面不用调整了，可以自动缩放）

### 与全参微调的关系

LoRA 调整那个 $r $ 就实现了和 full finetune 之间的平整过度，如果 $r $ 足够大，那就跟 全参微调几乎一致

也就是说 LoRA 在 full fine-tuning 和 极小参数更新之间，提供了一个连续可调的中间地带。

### 应用于 Transformer

在 Transformer 中的 self attention 一般有 4 个 投影矩阵 $W_q$, $W_k$, $W_v$, $W_o$ ，而 MLP 中有两个 Fully Connected Layer，分别是输入升维和输出降维，这些都可以插入 LoRA，论文中为了简单高效只在 Attention 模块中处理

另外，虽然 Transformer 中有多个 head，但是都把他们视作一个整体的矩阵 $d_{\text{model}} \times d_{\text{model}}$  来添加 LoRA

