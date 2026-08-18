# 第一次作业

> 关于详细的代码实现并不在这里展示，进摘选部分问题进行回答。
>
> 原作业中的部分题目也跳过，仅保留一部分内容。

## BPE Tokenizer

### unicode1

1.  What Unicode character does chr(0) return?

> `'\x00'`

2. How does this character’s string representation (`__repr__()`) differ from its printed representation?

>`repr()` 将它显示为 `'\x00'`，而直接打印时它通常不可见。

3. What happens when this character occurs in text? It may be helpful to play around with the following in your Python interpreter and see if it matches your expectations:

   ```python
   >>> chr(0)
   >>> print(chr(0)) 
   >>> "this is a test" + chr(0) + "string"
   >>> print("this is a test" + chr(0) + "string")
   ```

> 这是 Unicode 码点 U+0000（NUL）。它在 Python 字符串中是一个正常存在的字符，但直接打印时通常不可见；将其视为 C 字符串结尾是 C 语言及相关 API 的约定，并不意味着 Python 字符串会在这里截断。

### unicode2

1. What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings.

> 以字符串 `hello! こんにちは!` 为例：
>
> ```python
> >>> text = "hello! こんにちは!"
> >>> print(text.encode("utf-8"))
> b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'
> >>> print(text.encode("utf-16"))
> b'\xff\xfeh\x00e\x00l\x00l\x00o\x00!\x00 \x00S0\x930k0a0o0!\x00'
> ```
>
> UTF-8 兼容 ASCII：常见英文文本仍以单字节表示，适合从字节开始学习 BPE；UTF-16 在英文文本中会频繁出现 `\x00`，并且还涉及字节序与 BOM，容易让字节级统计混入编码层面的规律。UTF-32 的固定四字节表示则更浪费空间。UTF-8 同时能无歧义地表示全部 Unicode 字符，且无需依赖字节序。

2. Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results. 

```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
	return "".join([bytes([b]).decode("utf-8") for b in bytestring])
>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```

> ```python
> >>> decode_utf8_bytes_to_str_wrong("你好".encode("utf-8"))
> Traceback (most recent call last):
>   File "<stdin>", line 1, in <module>
>   File "<stdin>", line 2, in decode_utf8_bytes_to_str_wrong
>   File "<stdin>", line 2, in <listcomp>
> UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe4 in position 0: unexpected end of data
> ```
>
> 这个函数对每个字节逐一进行解码。UTF-8 的一个字符可能由多个字节组成；把多字节序列拆开后，单独的首字节或续字节不是合法的完整 UTF-8 字符，因此中文等多字节字符会报错。

3. Give a two-byte sequence that does not decode to any Unicode character(s).

>`\xe4\xbd` （"你"的前两个字节）

### train_bpe_tinystories

（关于 `train_bpe` 的实现直接参考代码即可）

1. Train a byte-level BPE tokenizer on the TinyStories dataset, using a maximum vocabulary size of 10,000. Make sure to add the TinyStories `<|endoftext|>` special token to the vocabulary. Serialize the resulting vocabulary and merges to disk for further inspection. How much time and memory did training take? What is the longest token in the vocabulary? Does it make sense?

>总耗时 550s，输入文件大小 2124.55 MiB，词表大小 10,000， RSS 内存占用 12GB
>
>按字节长度统计，最长的三个 token 为 `accomplishment`、`disappointment` 和 `responsibility`（对应词表 ID 分别为 7160、9143、9379）。这些都是 TinyStories 语料中较常见且可复用的词，因此合并为较长 token 是合理的。

2. Profile your code. What part of the tokenizer training process takes the most time?

>在统计并合并的过程中是最慢的，也就是 `bpe_merge()` 的过程

## Transformer

> 课程强烈推荐使用 `einops` 来简化矩阵的形状变化等操作，详细介绍可参见 [Einops 官方文档](https://einops.rocks/)。以及 [Einops 笔记](https://note.rainerseventeen.cn/code-algorithm/api/einops/)


### transformer_accounting

1. Consider a GPT-2 XL-sized model using our assignment architecture. How many trainable parameters would our model have? Assuming each parameter is represented using single-precision floating point, how much memory is required to just load this model? Suppose we constructed our model using this configuration:
```
vocab_size: 50,257
context_length: 1,024
num_layers: 48
d_model: 1,600
num_heads: 25
d_ff: 4,288 (the nearest multiple of 64 to 8/3 × 1,600)
```

>RMSNorm 包含一个 $d_{model}$ 维的可训练缩放参数: $1600$
>
>Transformer Block 中，MHA 的 Q、K、V 和输出投影共有 4 个矩阵，参数量为 $4\times1600\times1600$；SwiGLU 中 3 个权重矩阵的参数量为 $3 \times 1600 \times 4288$；两个 RMSNorm 的缩放参数量为 $2\times1600$。每个 Block 共计 $30,825,600$ 个参数，48 个 Block 共计 $48 \times 30,825,600 = 1,479,628,800$。
>
>Embedding 矩阵参数: $50257 \times 1600 = 80,411,200$
>
>最后 logit 线性层: $1600 \times 50257 = 80,411,200$
>
>全部加起来是 $1,479,628,800 + 80,411,200 + 1,600 + 80,411,200 = 1,640,452,800$，即约 $1.640B$ 参数。
>
>对于 FP32，每个参数占据 4 Bytes；仅加载权重需要 $1,640,452,800 \times 4 = 6,561,811,200$ Bytes，即约 $6.56$ GB（$6.11$ GiB）显存。

2. Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped model. How many FLOPs do these matrix multiplies require in total? Assume that our input sequence has context_length tokens.

>一次乘加（multiply-add）计为 2 FLOPs，一次 $A\in R^{a\times b},\quad B\in R^{b\times c}$ 的矩阵乘法算作为 ${2abc\text{ FLOPs}}$。设序列长度 $T=1024$，隐藏维度 $d=1600$，SwiGLU 中间维度 $d_{ff}=4288$。
>
>Embedding 是查表，RMSNorm、RoPE、SwiGLU 激活和 softmax 也都不是矩阵乘法，不计入本题要求的统计
>
>每个 Transformer Block 中：
>
>- Q、K、V 和输出投影共 4 次矩阵乘法$(T, d), (d, d)$，计算量为 $4\times2Td^2=8Td^2$
>- 注意力分数 $QK^\top$ 与注意力权重乘以 $V$ 各一次，计算量为 $2\times2T^2d=4T^2d$。
>- SwiGLU 的 3 个线性投影，计算量为 $3\times2Tdd_{ff}=6Tdd_{ff}$。
>
>因此每个 Block 的矩阵乘法计算量为：
>
>$$
>8\times1024\times1600^2
>+4\times1024^2\times1600
>+6\times1024\times1600\times4288
>=69,835,161,600\ \text{FLOPs}.
>$$
>
>48 个 Block 共需 $48\times69,835,161,600=3,352,087,756,800$ FLOPs。
>
>最后的 logits 线性层为一次 $(T,d)\times(d,\text{vocab\_size})$ 矩阵乘法，计算量为：
>
>$$
>2\times1024\times1600\times50257
>=164,682,137,600\ \text{FLOPs}.
>$$
>
>总计为：
>
>$$
>3,352,087,756,800+164,682,137,600
>=3,516,769,894,400\ \text{FLOPs}
>\approx3.52\ \text{TFLOPs}.
>$$
>
>

3. Based on your analysis above, which parts of the model require the most FLOPs?

> 在本题的 GPT-2 XL 配置（$T=1024$）下，SwiGLU 的 3 个线性投影约为每层 $42.15$B FLOPs，占 Block 矩阵乘法 FLOPs 的约 $60\%$，因此 MLP 是主要开销；Q/K/V/输出投影其次。注意力的 $QK^\top$ 与 $\operatorname{softmax}(QK^\top)V$ 合计约 $6.71$B FLOPs/层，但它随 $T^2$ 增长，在长上下文时会成为主导。

4. Repeat your analysis with GPT-2 small (12 layers, 768 `d_model`, 12 heads), GPT-2 medium (24 layers, 1024 `d_model`, 16 heads), and GPT-2 large (36 layers, 1280 `d_model`, 20 heads). As the model size increases, which parts of the Transformer LM take up proportionally more or less of the total FLOPs?

> 沿用 $d_{ff}$ 为最接近 $8d/3$ 的 64 的倍数，并取 $T=1024$、词表大小为 50,257，可得：
>
>| 模型 | $d_{ff}$ | 总矩阵乘法 FLOPs |
>| --- | ---: | ---: |
>| GPT-2 small | 2048 | $291,648,307,200\approx0.292$ TFLOPs |
>| GPT-2 medium | 2752 | $830,172,299,264\approx0.830$ TFLOPs |
>| GPT-2 large | 3392 | $1,768,530,903,040\approx1.769$ TFLOPs |
>| GPT-2 XL | 4288 | $3,516,769,894,400\approx3.517$ TFLOPs |
>
> head 数量本身不会改变总计算量：拆分或合并 head 后，投影维度总和仍为 $d$。随着层数和 $d$ 增大，Block 内的投影与 MLP 均按更高阶增长；由于 $d_{ff}\propto d$，MLP 的占比会上升。反之，固定 $T$ 时注意力的 $O(T^2d)$ 占比下降；最终 logits 投影的占比也因它不随层数增长而下降。

5. Take GPT-2 XL and increase the context length to 16,384. How does the total FLOPs for one forward pass change? How does the relative contribution of FLOPs of the model components change?

> 当 $T=16,384$ 时，GPT-2 XL 一次前向的矩阵乘法总计算量为
>
>$$
>133,577,729,638,400\ \text{FLOPs}\approx133.58\ \text{TFLOPs},
>$$
>
>是 $T=1024$ 时的约 $38.0$ 倍。此时各部分占比分别约为：Attention 矩阵乘法 $61.7\%$、MLP $24.2\%$、Q/K/V/输出投影 $12.1\%$、最终 logits 投影 $2.0\%$。原因是 Attention 的两次序列间矩阵乘法按 $T^2$ 增长，其余线性投影仅按 $T$ 增长。

## Train LM

### adamw_accounting

1. Assume we are using float32 for every tensor. How much peak memory does running AdamW require? Decompose your answer based on the memory usage of the parameters, activations, gradients, and optimizer state. Express your answer in terms of the batch_size and the model hyperparameters (`vocab_size`, `context_length`, `num_layers`, `d_model`, `num_heads`). Assume `d_ff = 8/3 * d_model`.
For simplicity, when calculating memory usage of activations, consider only the following components:

- Transformer block
    - RMSNorm(s)
    - Multi-head self-attention sublayer: $QKV$ projections, $QK^T$ matrix multiply, softmax, weighted sum of values, output projection.
    - Position-wise feed-forward (SwiGLU): w1, w2, SiLU on the gate branch, element-wise product, w3 
- final RMSNorm
- output embedding
- cross-entropy on logits

>记 $V=\texttt{vocab\_size}$、$B=\texttt{batch\_size}$、$T=\texttt{context\_length}$、$d=\texttt{d\_model}$、$L=\texttt{num\_layers}$、$h=\texttt{num\_heads}$，且 $d_{ff}=\frac{8}{3}d$。以下均以 **元素个数** 记账；每个 FP32 元素占 $4$ bytes。
>
>模型没有 bias，且输入、输出 embedding 不共享权重。因此可训练参数总数为
>$$
>P
>= \underbrace{Vd}_{\text{input embedding}}
>+ L\underbrace{\left(4d^2+3dd_{ff}+2d\right)}_{\text{attention、SwiGLU、2 RMSNorm}}
>+ \underbrace{d}_{\text{final RMSNorm}}
>+ \underbrace{Vd}_{\text{output embedding}}
>=2Vd+L(12d^2+2d)+d.
>$$
>
>AdamW 在一次更新中同时持有参数 $\theta$、梯度 $g$、一阶矩 $m$ 和二阶矩 $v$，故它们的显存分别为：参数 $4P$ bytes、梯度 $4P$ bytes、优化器状态 $8P$ bytes，合计 $16P$ bytes。
>
>再计算为反向传播保留的 activation。每个 Transformer block 中：两个 RMSNorm 各为 $BTd$；注意力部分的 $Q,K,V$、加权后的 values 与输出投影共 $5BTd$，$QK^T$ 和 softmax 各为 $BhT^2$；SwiGLU 的两条上投影、门分支 SiLU、逐元素乘积各为 $BTd_{ff}$，下投影为 $BTd$。因此单层为
>$$
>A_{\text{block}}
>=2BTd+(5BTd+2BhT^2)+(4BTd_{ff}+BTd)
>=\frac{56}{3}BTd+2BhT^2.
>$$
>
>最后的 RMSNorm 需要 $BTd$。输出 embedding 产生 logits $BTV$，cross-entropy 的 softmax/probability 中间量再计一个 $BTV$
>$$
>A=L\left(\frac{56}{3}BTd+2BhT^2\right)+BTd+2BTV
>=BT\left[L\left(\frac{56}{3}d+2hT\right)+d+2V\right].
>$$
>
>故峰值显存占用为：
>$$
>M_{\text{peak}}
>=4(4P+A)\ \text{bytes}
>=16\left[2Vd+L(12d^2+2d)+d\right]
>+4BT\left[L\left(\frac{56}{3}d+2hT\right)+d+2V\right]\ \text{bytes}.
>$$
>

2. Instantiate your answer for a GPT-2 XL-shaped model to get an expression that only depends on the batch_size. What is the maximum batch size you can use and still fit within 80GB memory?

>沿用前文 GPT-2 XL 配置 $V=50{,}257$、$T=1024$、$L=48$、$d=1600$、$h=25$，并使用配置中的 $d_{ff}=4288$。此时参数量为 $P=1{,}640{,}452{,}800$，因此参数、梯度和 AdamW 状态共占
>$$
>16P=26{,}247{,}244{,}800\ \text{bytes}.
>$$
>每增加一个 batch element，activation 占用
>$$
>4\left[L\left(8Td+4Td_{ff}+2hT^2\right)+Td+2TV\right]
>=16{,}373{,}391{,}360\ \text{bytes}.
>$$
>所以
>$$
>M_{\text{peak}}(B)
>=26{,}247{,}244{,}800+16{,}373{,}391{,}360B\ \text{bytes}.
>$$
>例如，$B=3$ 时约为 $75.37$ GB（$70.19$ GiB），而 $B=4$ 时约为 $91.74$ GB（$85.44$ GiB）。因此无论将 80GB 按十进制 GB 还是按 80 GiB 理解，最大 batch size 都是 $3$。

3. How many FLOPs does running one step of AdamW take?

>仅考虑 AdamW 更新时逐元素的运算，其中 $N$ 表示参数数量；忽略每个 parameter group 的标量 bias-correction 计算，并将平方根和除法各记为一次操作。
>
>1.  weight decay 需要 $2N$ 次逐元素操作（一次乘法、一次减法）
>2. 计算 m 需要 3N，计算 v 需要 4N
>3. 更新参数需要 5N 次
>
>综上，对每个可学习参数共需 $14$ 次操作，即一次 AdamW 更新需要 $14N$ FLOPs。对于本题的 GPT-2 XL 配置，$N=P=1{,}640{,}452{,}800$，故共约为
>$$
>14P=22{,}966{,}339{,}200\ \text{FLOPs}.
>$$

4. Model FLOPs utilization (MFU) is defined as the ratio of observed throughput (tokens per second) relative to the hardware’s theoretical peak FLOP throughput . An NVIDIA H100 GPU has a theoretical peak of 495 teraFLOP/s for “float32” (actually TensorFloat-32, which in reality is “bfloat19”) operations. Assuming you are able to get 50% MFU, how long would it take to train a GPT-2 XL for 400K steps and a batch size of 1024 on a single H100? Assume that the backward pass has twice the FLOPs of the forward pass.

>由前文可知，单个样本的一次前向传播需要 $3.51677$ TFLOPs。反向传播为前向的两倍，故一个 batch 的训练计算量为
>$$
>F_{\text{step}}
>=3\times 3.51677\times 1024
>\approx 10{,}803.5\ \text{TFLOPs}.
>$$
>50% MFU 时，有效吞吐为 $0.5\times495=247.5$ TFLOP/s，因此每 step 约需
>$$
>\frac{10{,}803.5}{247.5}\approx43.65\ \text{s}.
>$$
>训练 $400{,}000$ steps 共需约 $17.46\times10^6$ s，即约 $4{,}850$ 小时、$202$ 天（约 $6.6$ 个月）。AdamW 的 $14P$ FLOPs 相对该训练计算量可忽略。

### 最后训练

在 TinyStory 小数据集上训练的数据如图，具体来说用来 32 的 BS 跑 40k 轮迭代

![](http://oss.rainerseventeen.cn/blog/2026/202608171953576.png)

## 部分重点模块

1. RoPE 的实现
2. 归一化 softmax 为什么要归一化
3. 如何单次计算实现多头注意力， RoPE 是怎么应用到 MHA 的， MHA 的拆分多头是怎么实现的

### Cross Entropy Loss

Transformer 相关算法中主流的 Loss 实现，公式如下
$$
\ell(\theta;D)
=
\frac1{|D|m}
\sum_{x\in D}
\sum_{i=1}^{m}
-\log p_\theta(x_{i+1}|x_{1:i})
$$
外层的求和遍历数据集中每一个句子，内层的求和指一个句子遍历每一个 token，$p_\theta(x_{i+1}|x_{1:i})$ 表示给定前 $i$ 个token 预测第 $i + 1$ 个 token 的概率值。

同样要注意这里 log 运算和 softmax 是可以相互约去的：
$$
loss = -\log \frac{e^{x_y}}{\sum e^{x_i}}
=-\log(e^{x_y})+\log(\sum e^{x_i})
=-x_y+\log\sum e^{x_i}
$$
完整实现参见代码，尤其注意维度处理，已经在注释中标明。

### SGD  & AdamW Optimizer

优化器具体的工作包括：计算梯度，保存梯度，以及参数更新等工作

SGD 的算法公式如下
$$
\theta_{t+1}=\theta_t-\eta \nabla_\theta L(\theta_t)
$$
在初始化时需要保存一些超参数，例如 `param` 代表模型的可训练参数，另外 `lr` 这类超参数需要额外保存到字典中，因为不同的模块可能会对应不同的学习率，需要保存为 `parameter groups`, 每一个 `group` 有自己的超参数

在 AdamW 中我们需要额外存储一些 Tensor 到 state 中，并且他们的形状和输入参数 `param` 有关。

AdamW 的算法如下，其中 $\theta$ 可学习参数，$\alpha$ 是学习率，$\lambda$ 表示 `weitght_decay`数值：

第一步，让参数本身数值变小：
$$
\theta \leftarrow (1-\alpha\lambda)\theta
$$
第二步，计算优化器的状态量，其中两个 $\beta$ 都是超参数，经典值 `(0.9, 0.99)`
$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

第三步，更新梯度值：
$$
\theta \leftarrow \theta-\alpha_t\frac{m_t}{\sqrt{v_t}+\epsilon}
$$