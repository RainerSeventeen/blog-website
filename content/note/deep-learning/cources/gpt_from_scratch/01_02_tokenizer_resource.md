# 01~02 分词 & 计算资源

> 本笔记根据 CS336 课程撰写

## 1 分词（Tokenizer）

### 1.1 简介

分词有很多特别的性质，包括：

1. 单词前面的那个空格也是 token 的部分，例如 " world"
2. 一个单词，位于句子中不同位置，他们对应的 token 也有可能是不一样的

Compression Ratio 表明平均下来一个 token 可以表示多少 Byte 的数据。

在 python 中可以通过 `ord` 以及 `chr` 来将 char 转化为对应的 unicode 编码。

```python
assert ord("a") == 97
assert chr(97) == "a"
```

显然这个 compression ratio 不够好。 Vocabulary Size（词表大小）和压缩率是相对的，如果压缩率高了那么词表就必然会变大。

### 1.2 BPE 算法

最开始 BPE 是用来机器翻译的，GPT-2 使用的就是 BPE 算法。

方法为每次合并两个频率最高的组合，作为一个新的 token，随着迭代的进行，序列就会越来越小。

## 2 计算资源

### 2.1 浮点运算量与显存

以 H100 为例计算运算量

```python
# 参数量 * token数 * 6
total_flops = 6 * 70e9 * 15e12
h100_flop_per_sec = 1979e12 / 2
mfu = 0.5
flops_per_day = h100_flop_per_sec * mfu * 1024 * 60 * 60 * 24  
days = total_flops / flops_per_day  
```

计算显存

```python
h100_bytes = 80e9  
# parameters (2), gradients (2), optimizer state (4 + 4) 
bytes_per_parameter = 2 + 2 + (4 + 4)  
num_parameters = (h100_bytes * 8) / bytes_per_parameter  
```

### 2.2 存储带宽

GPU 的理论算力很高，但是在真正用于模型计算的有效算力可能只有理论的一部分，可以通过 **Model FLOPs utilization** 来衡量这个比值。

```python
mfu = actual_flop_per_sec / promised_flop_per_sec
```

一般来说 MFU 能够达到 50% 已经算是不错的了。

以 ReLU 为例，其存储带宽消耗计算如下

```python
def arithmetic_intensity_relu():
    n = 1024 * 1024
    x = torch.ones(n, dtype=torch.bfloat16, device=cuda_if_available())
    y = torch.relu(x)
    bytes = (2 * n) + (2 * n)  # Read x, write y (bf16 is 2 bytes/float)
    flops = n  # n comparisons
    communication_time = bytes / h100_bytes_per_sec  
    computation_time = flops / h100_flop_per_sec  
```

**Accelator Intensity** 指的是每一个 Byte 的传输可以进行多少次 FLOP 运算

**Arithmetic Intensity**（运算强度）在任务中，每一个 Byte 的实际运算量

```python
h100_accelerator_intensity = h100_flop_per_sec / h100_bytes_per_sec  
arithmetic_intensity = flops / bytes  # ~1/4 
# 对于 GELU 的 arithmetic intensity 大约是 5，比 ReLU 更优秀
# 但是在 H100 上这两者都会被存储带宽限制到（算的比传输要快）
```

在实际运算中，如果 Arithmetic Intensity < Accelator Intensity 就会被存储带宽瓶颈限制，此时 MFU 就会很低。

### 2.3 梯度

梯度同样需要浮点运算以及存储空间。在反向传播的时候计算代价会比前向传播昂贵不少。

对于给定的一个层 `h2 = h1 @ w2` 的运算（维度分别是$B \times D$ 以及 $D \times D$），在前向传播时需要的运算量为

```python
num_forward_flops = 2 * B * D * D
```

在反向传播时需要计算 h1 和 w2 的梯度：

```python
h1_grad = einsum(h2.grad, w2, "batch out, in out -> batch in")
w2_grad = einsum(h2.grad, h1, "batch out, batch in -> in out")
num_backward_flops = (2 * B * D * D) + (2 * B * D * D) 
```

因此在反向传播时至少需要 2 倍的运算量，对于前文中提到的 `参数量 * token数 * 6` 就是由 `2 + 4` 得到的

另外对于不同的优化器，需要保存的状态会有所不同，同时都会额外占用存储。

## 3 Pytorch

### 3.1 Tensor 数据类型

张量是最基础的存储数据类型，存储的大小取决于数据的类型

#### 3.1.1 FP32

最常见的浮点数（也是默认的张量类型）为 `float32` （也称为 fp32，IEEE754，1 符号位，8 指数位，23 个尾数位）

#### 3.1.2 FP16

为了节约存储并加快运算速度，也会有更小的精度，例如 `float16`（fp16，1 + 5 + 10），会导致更小的动态范围：

```python
x = torch.tensor([1e-8], dtype=torch.float16)  
assert x == 0  # Underflow!
```

#### 3.1.3 BF16

为了获得更大的动态范围，可以在指数分配更多的位，但这是以精度下降作为代价，得到了 `bfloat16`（bf16，1 + 8 + 7）

> 另外，还有 `FP8` 和 `FP4` 这些精度，NV 的底层硬会额外支持这些运算。

#### 3.1.4 混合精度

Pytorch 支持混合精度（AMP），例如 `BF16` 用在参数和梯度上，在优化器上则使用 `FP32`这些。

### 3.2 Einops

einops 是可以通过名称来对张量的维度执行操作的库，名称来源于 Einstein Operations

例如操作一个张量，使用传统的 Pytorch 需要写

```python
x.shape == (batch, channel, height, width)
# 改成 (batch, height, width, channel)
x = x.permute(0, 2, 3, 1)
```

使用 einops 可以写得很直观。

```python
from einops import rearrange
x = rearrange(x, "b c h w -> b h w c")
```

在 Transformer 中拆分多头时可以执行：

```python
assert q.shape == [batch, sequence, hidden]
# hidden = heads × head_dim
q = rearrange(
    q,
    "b n (h d) -> b h n d",
    h=num_heads
)
```

另外还有 `sum`, `reduce`等操作。 
