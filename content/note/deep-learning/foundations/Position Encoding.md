# Position Encoding

## 简介

位置编码是 LLM 和 embedding 并排的一个模块，用来弥补 Transformer 结构不自带顺序偏置的问题
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$
Attention 的算法中只有 token 内容对应的向量，以及对应的相似度，而没有位置信息，本质是任意 token 之间相互查询的集合关系，对 token 没有顺序感知能力

这里有个容易混淆的点，经过 embedding 后虽然张量的索引会自动构建出位置信息，但是模型是不知道这个信息的，这只是数据结构上的位置信息：

对于输入 $\begin{array}{c} X =
\begin{bmatrix}
x_a \\
x_{and} \\
x_b
\end{bmatrix} \end{array}$ 和 $\begin{array}{c} X' =
\begin{bmatrix}
x_b \\
x_{and} \\
x_a
\end{bmatrix} \end{array}$ 的不同顺序，Attention 的算法只按照新的顺序重新排列一遍，但是语义信息的变化模型是不会意识到的，存放顺序并不是可学习的位置信号；因此需要位置编码告诉 “猫咬狗” 和 “狗咬猫” 这两个之间的不同

对于 PE 也有很多的分类：

1. 输入表示附加值：经典绝对位置编码，包括 Transformer 原文中的编码；使用 embedding 的可学习位置编码（GPT-2 的方案）
2. 位置信息附加到 attention 中：相对位置编码；以及大名鼎鼎的 RoPE（现代最常用）；轻量化方案 ALiBi
3. 多模态扩展方案：例如 Qwen2-VL 中的 M-RoPE，应用于多模态的方案

## 正余弦位置编码

这里摘取一些 Transformer 论文记录的一些笔记

Attention 架构中无法感知所有 token 之间的顺序，因此需要位置编码结合到 Embedding 中，让模型感知到 token 的位置

一般而言：对一个 token 的向量$x_i \in \mathbb{R}^{d_{\text{model}}}$，加入位置编码后成为 $z_i = x_i + PE(i)$

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

令 $\frac{1}{10000^{2i/d_{\text{model}}}}$  为 $ \omega $ 则有：
$$
\sin(\omega(pos+k))=\sin(\omega pos)\cos(\omega k)+\cos(\omega pos)\sin(\omega k)
$$

$$
\cos(\omega(pos+k))=\cos(\omega pos)\cos(\omega k)-\sin(\omega pos)\sin(\omega k)
$$



4. 可以外推：因为是公式生成的，所以扩展到训练中没有见过的更长位置

还有一个非常重要点，随着维度$i$ 的提升，$ \omega $ 也会越来越小；位置相近时，编码通常变化较平滑；而多种不同频率的正余弦组合，使不同位置能够被区分开。

## RoPE

旋转位置编码，主要是原来的位置编码有这个问题：距离向量直接加到了词向量上，混淆了本来的词语的意思，也就是有很大的噪声

###  旋转公式推导

RoPE 首先利用了简单的平面二维向量的旋转公式：
$$
\begin{pmatrix}
a'\\
b'
\end{pmatrix}
=
\begin{pmatrix}
\cos \phi & -\sin \phi\\
\sin \phi & \cos \phi
\end{pmatrix}
\begin{pmatrix}
a\\
b
\end{pmatrix}
$$
RoPE 就是把每对维度当成这样的二维向量，对第 $m$ 个位置使用角度 $\phi = m\theta_j$，所以第 $j$ 对维度的旋转可写成：
$$
\begin{pmatrix}
x_{2j}'\\
x_{2j+1}'
\end{pmatrix}
=
\begin{pmatrix}
\cos(m\theta_j) & -\sin(m\theta_j)\\
\sin(m\theta_j) & \cos(m\theta_j)
\end{pmatrix}
\begin{pmatrix}
x_{2j}\\
x_{2j+1}
\end{pmatrix}
$$
举个例子， 第 $m$ 个词语计算和 第 $n$ 个词语的注意力分数，执行旋转得到 $q' = R(m\theta)\,q$  和 $k' = R(m\theta)\,k$ 

这里是对 $q$ 向量转置，因为对于 $q,k \in \mathbb{R}^{d\times 1}$ 的情况计算向量点积就是 ${q_m'}^\top k_n$ ：
$$
{q_m'}^\top k_n'
=
\bigl(R(m\theta)q\bigr)^\top \bigl(R(n\theta)k\bigr)
$$
利用线性代数性质 $(AB)^T = B^T A^T$ 来计算点积运算有：
$$
{q_m'}^\top k_n'
=
q^\top R(m\theta)^\top R(n\theta)k
$$
利用旋转矩阵转置的性质 $R(\alpha)^\top = R(-\alpha)$ ，以及旋转矩阵相加性 $R(a)R(b)=R(a+b)$ 有：
$$
{q_m'}^\top k_n'
=
q^\top R(-m\theta)R(n\theta)k = q^\top R((n-m)\theta)k
$$
这表明，旋转之后的点积式子中包含了原始词语的**相对位置信息**

### 实际应用

现在把原始的二维应用扩展到多维度的向量空间中，对一个 embedding 空间执行两两配对。

构造一个旋转的分块对角矩阵：$R_m=
\mathrm{diag}\big(
R(m\theta_0),R(m\theta_1),\dots,R(m\theta_{d/2-1})
\big)$ 

其中每一个都是小旋转矩阵（每一个块的旋转角度不一样） $\begin{array}{c} R(m\theta_i)=
\begin{pmatrix}
\cos(m\theta_i) & -\sin(m\theta_i)\\
\sin(m\theta_i) & \cos(m\theta_i)
\end{pmatrix} \end{array}$ 

实际上这个 $\theta$ 和正余弦位置编码几乎一样：$\theta_i = 10000^{-2i/d_{\text{model}}}$， 维度越低转的越快

具体的算式是这样的，可以通过两两配对+旋转矩阵方式计算：
$$
\begin{array}{c} R_m x
=
\begin{pmatrix}
R(m\theta_0) & 0 & \cdots & 0\\
0 & R(m\theta_1) & \cdots & 0\\
\vdots & \vdots & \ddots & \vdots\\
0 & 0 & \cdots & R(m\theta_{d/2-1})
\end{pmatrix}
\begin{pmatrix}
x_0\\
x_1\\
x_2\\
x_3\\
\vdots\\
x_{d-2}\\
x_{d-1}
\end{pmatrix} \end{array}
$$
在实际的计算中

- 对位置 $m$ 的 query 向量 $q_m$，做   $ \tilde q_m = R_m q_m$ 

- 对位置 $n$ 的 key 向量 $k_n$，做 $ \tilde k_n = R_n k_n$

然后再计算 attention score：$\tilde q_m^\top \tilde k_n$ 

### RoPE 到底好在哪里？

先说结论：绝对位置编码中会混入没有实际意义的噪声。

给定 qeury 和 key 向量分别为 $q_m = W_q h_m = W_q(x_m + p_m)$， $k_n = W_k h_n = W_k(x_n + p_n)$

计算打分公式得到：
$$
s_{mn} = q_m^\top k_n = \bigl(W_q(x_m+p_m)\bigr)^\top \bigl(W_k(x_n+p_n)\bigr) = 
(x_m+p_m)^\top W_q^\top W_k (x_n+p_n)
$$
这里先忽略掉投影矩阵，假设他们是单位矩阵 $ W_q^\top W_k = I$ 有：
$$
s_{mn}
=
x_m^\top  x_n
+
x_m^\top  p_n
+
p_m^\top x_n
+
p_m^\top p_n
$$
注意看这四个项目中，有两个是噪声项：

- $x_m^\top x_n$  和 $p_m^\top p_n$ 分别是 token 之间的内容相关性，以及两个绝对位置之间的相关性
- $x_m^\top p_n$ 是 “当前位置 $m$ 的内容”与“位置 $n$ 的位置编码”之间的耦合，$p_m^\top  x_n$ 同理，这两个交叉项会增加模型的理解力负担

相比之下，RoPE 的算法非常的干净，直接将位置信息注入到了注意力算法中：
$$
{q_m'}^\top k_n'
=
q^\top R(-m\theta)R(n\theta)k = q^\top R((n-m)\theta)k
$$

## MRoPE

在 Qwen2-VL/Qwen3-VL 中引入了多模态的旋转位置编码，用来对视频图像多模态内容额外处理

### `head_dim` 三段切分

标准 RoPE 中，每个注意力头的 $q, k \in \mathbb{R}^{d_{head}}$ 被两两配对，全部按同一个 1D 位置 $m$ 旋转。

MRoPE 把 $d_{head}$ 按 `mrope_sections = (s_t, s_h, s_w)` 切成三段（三段之和 $= d_{head}$），分别对应时序 / 高度 / 宽度轴：

$$
\underbrace{x_{0\dots s_t-1}}_{\text{temporal 段}} \;\Vert\; \underbrace{x_{s_t \dots s_t+s_h-1}}_{\text{height 段}} \;\Vert\; \underbrace{x_{s_t+s_h \dots d_{head}-1}}_{\text{width 段}}
$$

对 temporal 段用 $m_t$ 旋转，对 height 段用 $m_h$ 旋转，对 width 段用 $m_w$ 旋转。拼回去就得到了融合了三个位置信息的完整旋转向量。

**注意：**实际上不是这么个连续的一段段切分，在 Qwen3-VL 中，为了让每一个维度都能拿到合理的频率，实际上是三种维度交错进行的

### Qwen3-VL实现

官方实现中，对于一个多头注意力 `d_head = 128` 的情况，首先两两组队构成 `64` 个旋转向量

 RoPE 的半维频率槽位 上，时间、高度、宽度三者分到的数量分别是 24、20、20

后面再拼成 cos/sin 对应到完整 head 维度，因此最终对应的旋转通道数可理解为：

- temporal: $24 \times 2 = 48$
- height: $20 \times 2 = 40$
- width: $20 \times 2 = 40$

频率方面，首先前 20 对是交错为 THWTHW 的格式，最后还会多出 4 个时间轴的向量，全部都放在最后，因此是`[THWTHW....TTTT]`

### `position_ids` 的形状与生成 

MRoPE 的 `position_ids` 形状是 $(3, B, T)$，三个维度分别存 temporal / height / width 坐标。

**纯文字序列**：三个轴全都赋同一个单调递增值，退化为标准 1D RoPE：

```
input:  [T T T T T]
time:   [0 1 2 3 4]
height: [0 1 2 3 4]
width:  [0 1 2 3 4]
```

**图文混合序列（以图像为例）**：假设图像 patch 网格大小为 $H \times W$（经过 `spatial_merge_size` 下采样后），图像前的文字正常编号，到图像区域时：

- **temporal 轴**：所有图像 token 都固定为同一个 anchor（等于图像段起始位置，表示"同一帧"）
- **height 轴**：每行的所有 patch 共享同一个行索引
- **width 轴**：每列的所有 patch 共享同一个列索引

具体例子（图像在序列中间，$H=2, W=2$）：

```
input:  [T T | V V V V | T T]
time:   [0 1 | 2 2 2 2 | 4 5]   ← 图像段全部 anchor=2
height: [0 1 | 2 2 3 3 | 4 5]   ← 两行各自递增
width:  [0 1 | 2 3 2 3 | 4 5]   ← 每行内列索引重复
```

图像后的文字从 $\max(\text{视觉 pos}) + 1$ 开始，三个轴相同（即 4、5…），保证文字位置连续。

### 代码实现：compute_mrope_position_ids

```python
# src/model/mrope_position.py（简化版）
position_ids = linear.view(1, 1, seq_len).expand(3, B, T)  # 先全部线性

for each sample in batch:
    span_start = 图像 token 在序列中的起始位置
    offsets = arange(grid_h * grid_w)
    # temporal 轴：全部固定为 span_start（anchor）
    position_ids[0, b, span_start:span_end] = span_start
    # height 轴：span_start + 行索引
    position_ids[1, b, span_start:span_end] = span_start + offsets // grid_w
    # width 轴：span_start + 列索引
    position_ids[2, b, span_start:span_end] = span_start + offsets % grid_w
```

注意 anchor 选择图像段起始位置而不是 0：这样后续文字的编号继续从最大值 +1 递增，不会与之前文字重叠。

### 代码实现：compute_mrope_cos_sin

```python
# src/model/rope.py（简化版）
for axis_i, section in enumerate(mrope_sections):  # 3 个轴
    # 用第 axis_i 个轴的 position_ids 做索引查频率表
    pos = position_ids[axis_i]          # (B, T)
    cos_parts[axis_i] = freqs_cos[pos, start:end]  # 取对应维度段
    sin_parts[axis_i] = freqs_sin[pos, start:end]

cos = concat(cos_parts, dim=-1)  # (B, T, head_dim)
sin = concat(sin_parts, dim=-1)
```

最终 `cos/sin` 形状 $(B, T, d_{head})$，送入 `apply_rotary_pos_emb` 与标准 RoPE 完全相同的公式完成旋转：

$$
q' = q \odot \cos + \mathrm{rotate\_half}(q) \odot \sin
$$

### Qwen2-VL vs Qwen3-VL 的视频时序编码差异

| 版本 | 时序编码方式 | 时序步长 |
|------|-------------|---------|
| Qwen2-VL | 每帧帧号直接作为 temporal 坐标，等间隔递增 | `t_index = 0, 1, 2, ...` |
| Qwen2.5-VL | 引入 `second_per_grid_t` 真实物理时间，步长为 `t * seconds * 2` | 非等间隔，反映真实帧率 |
| Qwen3-VL | 改用**时间戳 token** `<t1><t2>` 携带时序，每帧单独 1 个时间步，不再用 temporal 轴累计帧数 | `t_index` 始终为 0（每帧只有 1 层时间 patch） |

Qwen3-VL 之所以这样改，是因为时间戳方案把"帧间时间差"从位置编码移到了 token 语义层，LLM 可以直接从 token 内容读取时间信息，避免了 RoPE 时序轴随帧数线性膨胀导致的位置外推压力。

### spatial_merge_size 与实际分辨率

Qwen 视觉塔输出的视觉特征在送入 LLM 前会做 2×2 的空间合并（`spatial_merge_size=2`），所以 LLM 实际看到的网格是 $H/2 \times W/2$，MRoPE 的 height / width 坐标也相应使用合并后的尺寸：

```python
llm_grid_h = h // spatial_merge_size
llm_grid_w = w // spatial_merge_size
```

### mrope_position_deltas：KV Cache 推理时的位置连续性

在自回归生成阶段，每次只 forward 一个新 token。由于 MRoPE 图像段的 position_ids 不是简单线性的，"下一个 token 的位置"不等于序列长度，需要通过 `mrope_position_deltas` 来补偿：

$$
\text{mrope\_position\_deltas} = \max(\text{position\_ids}) + 1 - \text{seq\_len}
$$

新生成的文字 token 位置计算为：

$$
\text{new\_pos} = \text{seq\_len}_{\text{current}} + \text{mrope\_position\_deltas}
$$

这个偏移量在 prefill 阶段一次性计算好缓存，后续每步 decode 把三个轴都设为同一个单调递增的值（同标准 1D RoPE），保证文字区不出现位置回退或重叠。

### 与标准 RoPE 的关键异同

| 对比点 | 标准 RoPE | MRoPE |
|-------|----------|-------|
| position 轴数 | 1 | 3（t, h, w） |
| head_dim 使用 | 全部按同一位置旋转 | 按三段分别旋转 |
| 文字 token | 1D 单调递增 | 三轴相同值，退化为 1D |
| 图像 token | 拍平后 1D，位置不自然 | 2D 网格坐标，空间结构保留 |
| 旋转公式 | 不变 | 不变 |
| Q/K dot-product | 只含 1D 相对位置 | 含 2D/3D 相对位置 |

本质上 MRoPE 没有改变旋转的数学形式，只是把"给谁旋转多少"这件事从单轴扩展到了三轴，靠切分 head_dim 来实现三个轴的独立编码。
