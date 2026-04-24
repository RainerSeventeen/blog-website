# Normalization

> 代码实现详见 [笔记仓库](https://github.com/RainerSeventeen/dive-into-deep-learning/tree/main/code/_common_interview_question)

## 简介

归一化方案，表面上是一个归一化的工作，实际上是一个**改善训练的模块**

让网络更容易优化， 允许更大的学习率，一定程度上起到了正则化的作用

## Batch Norm

### 简介

首先简单明确一下 Batch 的概念：一次性让模型计算 Batch size 个数据的梯度，并综合这所有的数据的梯度（一般是平均，也有求和）执行反向传播更新参数。

对于深度学习的训练中，随着网络层数加深，会遇到这些问题：

1. 如果某一个层的输出非常大，进入 SIGMOD 或 tanh 就会很容易饱和，于是梯度消失的问题
2. 还有就是不同层的输入分布不同，会不断地漂移

总体来说 BN 可以让：

1. 训练稳定，中间层输出更加稳定，梯度传播稳定
2. 对初始化值没那么敏感，也能使用更大的学习率
3. 有一定的正则化效果，BN 的统计量会引入噪声

### 算法

BN 主要应用目标是卷积层和全连接层，对“同一种特征”的一组激活值，利用当前 mini-batch 里的统计量做标准化。

对一个全连接层有输入 $x \in \mathbb{R}^{N \times D}$ ，其中$N$是 batch size，$D$ 是特征维度，沿着 batch 维度（纵向）求均值和方差：
$$
\mu_B = \frac{1}{m}\sum_{i=1}^{m} x_i
$$

$$
\sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m}(x_i - \mu_B)^2
$$

执行标准化计算，这里 $\epsilon$ 是一个小的常数，防止除以 0，提高数值稳定性：
$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$
缩放和平移，这里还有两个可以学习的参数 $\gamma$ 和 $\beta$ ，增加网络的表达能力：
$$
y_i = \gamma \hat{x}_i + \beta
$$
上式中在训练的时$\mu$ 和 $\sigma$ 都是可以实时计算的，因为训练的时候可以完整的看到一个 batch，但是推理的时候不一样，需要使用训练中累计的参数$\mu_{\text{running}}, \quad \sigma^2_{\text{running}}$来，类似于 momentum 的优化方法：

$$
\mu_{\text{running}} = (1-\text{momentum}) \cdot \mu_{\text{running}} + \text{momentum} \cdot \mu_B
$$
对于 CNN 算法中，BN 是对每一个 channel  执行单独的计算（因为每一个 channel 一般算一个独立的特征图）

详细的实现可以看代码，CNN 中 channel 中的所有数据视作为 同一种特征

## Layer Norm

相对于 BN 的同一批次的样本执行归一化，LN 是单个样本的归一化，对小 batch 相对友好

另外就是对 Transformer 中如果序列长度不固定的情况 下比较适用

对于 Transformer 的常见输入形状：$(B, T, C)$ 中 LN 一般是对最后一个维度 $C$ 做归一化

也允许对多个维度执行 LN，常见于 ViT 中

对于给定的一个输入序列 $x = [x_1, x_2, \dots, x_d]$ 计算其均值和方差：
$$
\mu = \frac{1}{d} \sum_{i=1}^{d} x_i
$$

$$
\sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2
$$

注意方差计算是  **biased variance**（有偏估计），除以的是 $d$，对应到代码中是

```python
x.var(dim=dims, keepdim=True, unbiased=False)
```

最后执行标准化并加上可以学习的缩放和平移：
$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

$$
y_i = \gamma_i \hat{x}_i + \beta_i
$$



## RMS Norm

现代 LLM 中更常用归一化方案，可以看做是 Layer Norm 的简化版，不减均值

对于输入向量$x = [x_1, x_2, \dots, x_d]$，计算其**均方根**（Root Mean Square, RMS）
$$
\mathrm{RMS}(x)=\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}
$$
执行归一化再乘以一个可以学习的参数得到（这里一般会对 $\gamma$ 执行广播，$\odot$ 是逐元素乘法）：
$$
y = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma
$$
因为 RMS 不做中心化，因此也没有对应的 $\beta$ 偏置项

### RMSNorm 的意义

> 为什么要用 RMSNorm 替换 Transformer 原文中的 LN 是个常见的问题

1. 计算更简单，RMSNorm 不用做均值的加减法，算的更快

2. Transformer 中有大量的残差结构，也就是 $x + \mathrm{Sublayer}(x)$ ，实验表明保留均值对语义表达更加自然，不用强行改变中心的位置
