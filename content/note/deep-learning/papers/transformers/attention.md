---
paper: "Attention Is All You Need"
---

# Attention

> 梦开始的地方：*Attention Is All You Need*
>
> 本文不再介绍背景等信息，而是重点关注具体算法与实现

## 注意力机制

### 一维注意力

注意力分数指的是：对于一个输入 Q，通过和 K 计算某种关系，得到 V 的权重，最后权重乘以 V 得到最后的分数

举个例子，对于这个公式，其中 $\alpha(x, x_i)$ 是注意权重，$-\frac{1}{2}(x - x_i)^2$ 表示的是注意力分数：
$$
f(x) = \sum_i \alpha(x, x_i) y_i 
= \sum_{i=1}^{n} \operatorname{softmax}\!\left(-\frac{1}{2}(x - x_i)^2\right) y_i
$$
上式是计算 $x$  的注意力分数的公式，表示输入 当前输入 $x$ 去和一组记忆位置 $x_i$ 做相似度比较，得到权重 
$\alpha(x,x_i)$，再用这些权重对对应的 $y_i$ 加权求和。

直观理解就是，如果 $x$ 和 $x_i$ 非常近，那么对应的权重就会上升，所以对应的输出就会更大

### 向量化

扩展到向量后，形式几乎不变，但是差平方替换为向量的平方范数 $\|x-x_i\|^2 = \sum_{m=1}^d (x_m-x_{i,m})^2$ ，也就是所有维度的差的平方求和：
$$
f(x)=\sum_{i=1}^n \alpha_i(x)\, y_i, \quad \alpha_i(x)=\frac{\exp\left(-\frac12 \|x-x_i\|^2\right)}
{\sum_{j=1}^n \exp\left(-\frac12 \|x-x_j\|^2\right)}
$$
上式就是带入了一个 Softmax 的结果

### Scaled Dot-Product Attention

最终的目标是缩放点积注意力公式
$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

#### 单个 Query 情况

1 个 query：$q \in \mathbb{R}^{d_k}$，$n$ 个 key：$k_1,\dots,k_n \in \mathbb{R}^{d_k}$，对应 $n$ 个 value：$v_1,\dots,v_n \in \mathbb{R}^{d_v}$

1. 计算 query 和每一个 key 的相似度，也就是 $s_i = q \cdot k_i = q^\top k_i$，也就是点积，越大表示向量方向越近
2. 缩放：$s = [s_1,s_2,\dots,s_n]$ 执行缩放后为 $\tilde{s}_i = \frac{s_i}{\sqrt{d_k}}$ ，也就是对维度缩放一下
3. Softmax：计算成概率权重，$\alpha_i = \frac{e^{\tilde{s}_i}}{\sum_{j=1}^n e^{\tilde{s}_j}}$ 得到一组权重（就是注意力分数）满足 $\alpha_1,\alpha_2,\dots,\alpha_n$，$\alpha_i \ge 0,\quad \sum_i \alpha_i = 1$
4. 加权求和：$o = \sum_{i=1}^n \alpha_i v_i$ 

#### 为什么使用点积？

1. 点积对于多个 query 和 key，可以用矩阵并行化一次运算完毕
2. 本身具有相似度的含义：$q^\top k = \|q\|\|k\|\cos\theta$

#### 为什么要缩放？

>  一个很常见的八股文问题，需要掌握数学推导

如果 $q$ 和 $k$ 的每个分量都大致是均值 0、方差 1 的随机变量，那么点积 $q^\top k = \sum_{m=1}^{d_k} q_m k_m$ 是 是 $d_k$ 项的求和

如果每一项的方差大致是 1，那么总方差会随维度增长，大约是：$\mathrm{Var}(q^\top k) \propto d_k$ 

关于高斯方差累计，对于独立的随机变量有 $\operatorname{Var}\left(\sum_{i=1}^{n} X_i\right)=\sum_{i=1}^{n}\operatorname{Var}(X_i)$ 

Softmax 对输入的尺度非常敏感，所以大方差会导致指数迅速拉开差距，方差大后几乎会退化成 one-hot 的形式

对 Softmax 计算梯度得到
$$
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}, \quad \frac{\partial p_i}{\partial z_k}=p_i(\delta_{ik}-p_k)
$$
其中 $\delta_{ik}$ 在 $i = k$ 等于 1，否则为 0，对于尖锐的 Softmax 有  $p_m \approx 1,\quad p_j \approx 0\ (j\neq m)$

代入得到：

- 对最大那个位置：$\frac{\partial p_m}{\partial z_m}=p_m(1-p_m)\approx 1\cdot 0=0$
- 对其他位置：$\frac{\partial p_j}{\partial z_j}=p_j(1-p_j)\approx 0$
- 交叉项：$\frac{\partial p_i}{\partial z_k}=-p_i p_k \approx 0$

整个 Softmax 的雅可比矩阵几乎全部都是 0，这也就是梯度消失

### 矩阵形式

> 尤其注意一下这里的维度变化

对于输入的 $Q \in \mathbb{R}^{n \times d_k}$，$K \in \mathbb{R}^{n \times d_k}$， $V \in \mathbb{R}^{n \times d_v}$ 三个矩阵计算 Attention：

1. 计算分数矩阵：$S = QK^\top$ ，其中$S \in \mathbb{R}^{n \times n}$， $S_{ij} = q_i^\top k_j$ （就是一维的值分布到了各个矩阵位置上）
2. 缩放：$\hat{S} = \frac{S}{\sqrt{d_k}}$
3. 有时候还会有掩码，也就是不允许看到未来的 token，负无穷到 softmax 分子是 0：

$$
\begin{array}{c} \hat{S}_{ij} =
\begin{cases}
\hat{S}_{ij}, & \text{允许关注}\\
-\infty, & \text{不允许关注}
\end{cases} \end{array}
$$

4.   Softmax：对**每一行**做 $A = \mathrm{softmax}(\hat{S})$ ，这里没有发生维度变化$A \in \mathbb{R}^{n \times n}$，只是改成了概率分布

5. 输出：$O = AV$，其中$O \in \mathbb{R}^{n \times d_v}$ ，$o_i = \sum_{j=1}^n A_{ij} v_j$ 表示第 $i$ 个位置从全序列汇总得到的新表示

实际上输入最基本的维度要求是这样的：

- $Q \in \mathbb{R}^{n_q \times d_k}$

- $K \in \mathbb{R}^{n_k \times d_k}$

- $V \in \mathbb{R}^{n_k \times d_v}$

主要有两点要求：

1. $Q$ 和 $K$ 的内积维 $d_k$ 必定相等，因为要做点积
2. $K$ 和 $V$ 序列长度 $n_k$ 必须一样，但是特征维度$d_k$ $d_v$可以不相同



#### 为什么 Softmax 对行做？

因为每一个行对应一个 query， $[S_{i1},S_{i2},...,S_{in}]$ 表示第 $i$ 个 query 对所有 key 的打分

### Self-Attention & Cross-Attention

自注意力指的是，所有的 QKV 都来自与一个 $X$ 

对于序列长度是 $n$，每个 token 的输入表示维度是 $d_{\text{model}}$，有输入矩阵$X \in \mathbb{R}^{n \times d_{\text{model}}}$

将输入经过三个投影矩阵计算后得到：

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

各自的维度是： $W_Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_K \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_V \in \mathbb{R}^{d_{\text{model}} \times d_v}$ 

后续的算法都是跟矩阵形式是一样的了

而 交叉注意力唯一的区别就是：

$$
Q=X_1W_Q,\quad K=X_2W_K,\quad V=X_2W_V
$$

## 多头注意力

### 简介

普通的单头注意力只有一套 $W_Q,W_K,W_V$，因此只能在一个子空间里做一次注意力匹配，一次只能学习到一次关系模式

但是在同一句话中模型需要同时关注多种关系，因此可以投射子空间来强化理解能力

### 实现

多头注意力的做法是把 $Q,K,V$ 分别投影到 **多个不同的子空间**，每个子空间各自做一次注意力，最后再拼接起来。

对于输入 $X \in \mathbb{R}^{n \times d_{\text{model}}}$，头数 $h$，每个头的维度 $d_k=d_v=d_{\text{model}}/h$（注意这里 Q 和 K 的最后一维必相同），第 $i$ 个头有：
$$
\text{head}_i = \mathrm{Attention}(Q_i, K_i, V_i) 
$$
其中：$W_Q^{(i)} \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_K^{(i)} \in \mathbb{R}^{d_{\text{model}} \times d_k}$ $W_V^{(i)} \in \mathbb{R}^{d_{\text{model}} \times d_v}$ 计算得到 
$$
Q_i=XW_Q^{(i)} \in \mathbb{R}^{n \times d_k}, K_i=XW_K^{(i)} \in \mathbb{R}^{n \times d_k}, V_i=XW_V^{(i)} \in \mathbb{R}^{n \times d_v}
$$
计算注意力权重：
$$
A_i=\mathrm{softmax}\left(\frac{Q_iK_i^T}{\sqrt{d_k}}\right), Q_iK_i^T \in \mathbb{R}^{n \times n}
$$
因此每一个头都有自己的注意力矩阵，每个头运算得到结果：
$$
\text{head}_i=A_iV_i \in \mathbb{R}^{n \times d_v}
$$
执行矩阵拼接得到 $H=\mathrm{Concat}(\text{head}_1,\dots,\text{head}_h)\in \mathbb{R}^{n \times (h d_v)}$ （也就是第二个维度执行左右拼接），最后乘以输出矩阵$W_O$

目的是把所有头拼接后的结果再映射回模型维度，得到 $Y \in \mathbb{R}^{n \times d_{\text{model}}}$
$$
Y=HW_O,\quad W_O\in\mathbb{R}^{h d_v \times d_{\text{model}}}
$$
这里 $W_O$ 类似于整合作用，把所有的信息混合到一个统一的表示

## 位置编码

Attention 架构中无法感知所有 token 之间的顺序，因此需要位置编码结合到 Embedding 中，让模型感知到 token 的位置

一般而言：对一个 token 的向量$x_i \in \mathbb{R}^{d_{\text{model}}}$，加入位置编码后成为 $z_i = x_i + PE(i)$

总体而言 PE 分为两大类，绝对位置编码和相对位置编码

绝对指的是，0123 这种绝对位置，Attention 论文中用的就是固定的正余弦位置编码；相对指的是表达两个 token 之间的相对位置

在论文中定义 PE 为下式，其中$pos$是位置，$i$是维度索引的一半，$d_{\text{model}}$是模型维度
$$
PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

$$
PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

#### 为什么用正余弦？

1. 最基本的：不同的位置可以得到不同的值，这可以区分位置
2. 编码具有连续性：位置相近则编码结果相似，距离远则位置更远
3. 可以学习**相对位置**：$\sin(a+b),\cos(a+b)$ 可以通过由 $\sin a,\cos a$ 与偏移量 $b$ 的关系表示（三角和公式）
4. 可以外推：因为是公式生成的，所以扩展到训练中没有见过的更长位置

## Transformer 架构

<img src="https://oss.rainerseventeen.cn/blog/2025/202510271427269.png" alt="Attention 架构" style="zoom: 33%;" />

### 输入层

指的是 Embedding + Position Encoding 向量模块，整体流程是：

1. tokenizer 将整个句子切分一下，常见的方式有 BPE 组合
2. Embedding，将每一个 token 映射为一个向量 $x_i \in \mathbb{R}^{d_{\text{model}}}$，隐藏维度 $d_{\text{model}}$，序列长度$n$
3. PE 加到 Embedding 结果中：$Z = X + PE$， $Z$ 是 Encoder 的输入

在论文中超参数 $d_{\text{model}} = 512$

### Encoder

整个 Encoder 包括以下这些结构，构成一个 Block，在论文中是堆叠了 $N = 6$ 次

1. Multi-Head Attention（是 Self-Attention）
2. Add & Norm
3. Position-wise Feed Forward Network
4. Add & Norm

Transformer 各个层的序列的**长度和维度全都不变，隐藏维度也保持不变**，所以很方便堆叠很多层

#### MHA

使用 Self-Attention 实现的 MHA，QKV 来自于同一个 $X$，也就是上文中的 $Z$

先执行线性映射，对输入映射到 $Q = XW^Q,\quad K = XW^K,\quad V = XW^V$

然后按照 MHA 流程切分到$h$个头（论文中超参数 $h = 8$），每一个头计算 $\text{head}_h = \text{softmax}\left(\frac{Q_hK_h^T}{\sqrt{d_k}}\right)V_h$

最后执行拼接 $\text{MultiHead}(X) = \text{Concat}(\text{head}_1,\dots,\text{head}_H)W^O$ ，输出维度仍然是：$\mathbb{R}^{n \times d_{\text{model}}}$ 

#### Add & Norm

残差连接（Residual Connection）+ 层归一化（LayerNorm）：
$$
\text{LayerNorm}(X + \text{Sublayer}(X))
$$
其中 $\text{Sublayer}(X)$ 是上一个 MHA 的变换后的输出结果

Layer Norm 算法指的是，对于 $x = [x_1, x_2, \dots, x_d]$ 序列计算均值和方差：$\mu = \frac{1}{d}\sum_{i=1}^{d} x_i$， $\sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i - \mu)^2$，执行归一化：
$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$
其中 $\epsilon$ 是一个很小的数，防止除零。之后再使用一个可学习的额缩放和平移：
$$
y_i = \gamma_i \hat{x}_i + \beta_i
$$
作用是先执行标准化，之后再让模型学习一个更加合适的分布

> 为什么用 Layer Norm 而不是 Batch Norm 也是个很常见的问题，这里省略

残差连接的功能是帮助梯度传播，减轻网络太深导致的退化问题。（为什么呢？）

#### Feed Forward Network + Add & Norm

前馈网络，指的是对每一个位置做一个相同的 MLP
$$
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$
原论文用的是两层线性层，中间 ReLU，第一层：从 $d_{\text{model}}$ 升到 $d_{ff}$；第二层：从 $d_{ff}$ 降回 $d_{\text{model}}$，就是一个多层感知机，论文中这个超参数 $d_{ff} =  2048$

Attention 的作用是 token 之间的信息交互，FFN 的作用是让 token 与对自己的做非线性变换

在 FFN 后接一个残差连接，承接作用

### Decoder

Decoder 也是堆叠 $N = 6$ 层，但每层比 Encoder 多一个注意力模块，包含（这里把 Add & Norm 结合到上一层了）后文省略 Add & Norm：

1. Masked Multi-Head Self-Attention + Add & Norm
2. Encoder-Decoder Attention（Cross-Attention）+ Add & Norm
3. Feed Forward + Add & Norm

因为 Decoder 不仅仅需要自己的生成信息，还需要输入句子的相关信息，SA 负责看已经生成的 token 的信息，CA 负责查看 Encoder 的内容

#### MHA

Decoder 的输入是当前输入的是将目标整体向右移动一位的输入，也就是：

对于目标：

```
<bos> 我 喜欢 学习 Transformer <eos>
```

Decoder 的输入是：

```
<bos> 我 喜欢 学习 Transformer
```

Decoder 的期望输出和监督目标是（`<bos>`是启动生成的编码）：

```
我 喜欢 学习 Transformer <eos>
```

因为模型不应该看到预测的目标，所以这里会出现一个 Mask 部分，实际上就是累加一个下三角矩阵 $M$ ：

$$
\begin{array}{c} M_{ij} =
\begin{cases}
0, & j \le i \\
-\infty, & j > i
\end{cases} \end{array}
$$

回顾一下 负无穷在 Softmax 的输出就是 0，这保证了 Decoder 的输出是自回归的，不依赖未来

#### Cross-Attention

Decoder 相对于 Encoder 最大的区别在这里，这里的 $Q$ 来自 Decoder 当前隐状态，但是 $K,V$ **来自 Encoder 输出**：
$$
Q = H_{\text{dec}} W_Q,\quad K = H_{\text{enc}} W_K,\quad V = H_{\text{enc}} W_V
$$
$H_{\text{enc}}$：Encoder 最后一层输出的整段序列表示，$H_{\text{dec}}$：Decoder 在进入 cross-attention 前的输入表示

注意这里$W_Q, W_K, W_V$是**这一层 cross-attention 自己学习的参数**，不是 Encoder 中缓存的 KV

CA 作用是让 Decoder 在生成的时候会额外考虑输入句子的相关内容，类似于Decoder 在边生成边对输入做检索

#### FFN

最后在输出的地方执行 FFN + Add & Norm 步骤，作用同样是 token 与自己交互

### 输出层

Decoder 最后一层的输出是 $Y \in \mathbb{R}^{n \times d_{\text{model}}}$ ，通过一个线性层，映射到词表中：
$$
\text{logits} = YW_{vocab} + b
$$
对于词表大小是 $V$，则有：$\text{logits} \in \mathbb{R}^{n \times V}$

最后对每一个位置执行 Softmax 可以计算出每一个词语的概率：
$$
P(y_t \mid y_{<t}, x) = \text{softmax}(\text{logits}_t)
$$


## KV Cache

> 论文中没有直接提出这个，但是这是一个非常常用的工程优化方案

### 简介

对于一个 Decoder 生成中过程，假设已经生成了： $y_1, y_2, y_3$ 现在开始预测 $y_4$

没有 Cache 的 Decoder，那么会把整个序列 $[y_1, y_2, y_3]$， 重新送进模型，再算一次 self-attention。

等要预测 $y_5$ 时，又把：$[y_1, y_2, y_3, y_4]$整个再算一遍，于是前面那些 token 的 $K,V$ 会被**重复计算很多次**。

KV Cache 的核心思想是：历史 token 的 Key 和 Value 一旦算出来，后续生成时就不变，缓存起来复用。

 （感觉有点像 DP 里面的记忆化搜索hhhh，简而言之就是缓存减少重复运算）

### 具体对象

对于某一个 Decoder 中的某一个 Attention 模块，有隐藏状态：$X \in \mathbb{R}^{T \times d_{\text{model}}}$， 经过线性映射得到：
$$
Q = XW^Q,\quad K = XW^K,\quad V = XW^V
$$
经过 MHA 以及多 Batch 得到：$Q, K, V \in \mathbb{R}^{B \times H \times T \times d_{\text{head}}}$ 

KV Cache 就是缓存每一层历史里的历史位置：$K_{\text{past}}, V_{\text{past}}$，

也就是： $K_{\text{cache}} \in \mathbb{R}^{B \times H \times T_{\text{past}} \times d_{\text{head}}}$ , $V_{\text{cache}} \in \mathbb{R}^{B \times H \times T_{\text{past}} \times d_{\text{head}}}$ 

### 计算例子

如果是在 **Transformer decoder 推理** 里做 KV cache，缓存空大小：
$$
\text{Cache bytes}
=
\text{N}
\times
(\text{需要缓存的 attention 模块数/层})
\times
2
\times
B
\times
n
\times
\text{bytes\_per\_elem}
$$

- 层数：Decoder Block 个数，经典值 $N = 6$
- Attention 个数：也就是每一个 block 有几个 Attention 模块
-  2 ：这个 2 指的是 KV 各要一份缓存空间，所以一个 Attention 模块是 2 份
- $B$：Batch Size 
- $n$ ：序列长度
- $\text{bytes\_per\_elem}$：每一个数据的结构大小

实际上严谨一些的话也不完全对，在 Decoder 架构中的 SA 和 CA 的长度并不一致：

SA 缓存的是 decoder 已生成序列，长度是 $n$；CA的 $K,V$ 来自 encoder 输出，长度应该是源序列长度，记为 $m$

更加严格的写法是：$(2nd + 2md) \times B \times N \times (\text{需要缓存的 attention 模块数/层})$ 
