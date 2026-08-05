# Einops

`einops` 是一个用于**清晰地表达张量维度变换**的 Python 库，用统一的字符串表达式完成：`reshape`, `transpose / permute`, `squeeze / unsqueeze` 等张量操作的库。详细内容可以参见[官方文档说明](https://einops.rocks/)

## 1 优化方面

假设有一个图像张量：

```python
x.shape == [batch, channel, height, width]
```

需要把它转成：

```python
[batch, height * width, channel]
```

使用原生 PyTorch 写法

```python
b, c, h, w = x.shape
x = x.permute(0, 2, 3, 1)
x = x.reshape(b, h * w, c)
```

如果使用 einops 可以写作

```python
from einops import rearrange

x = rearrange(x, "b c h w -> b (h w) c")
```

含义是：

```
batch, channel, height, width
    ↓
batch, height×width, channel
```

因此，`einops` 最核心的收益不是“增加新的计算能力”，而是用带名字的维度表达式，代替大量容易混淆的 `reshape`、`view`、`permute` 和 `transpose`。

------

## 2 常用 API

### 2.1 `rearrange`

负责改变张量维度的组织方式。

#### 2.1.1 交换维度

```python
x = rearrange(x, "b c h w -> b h w c")
# 等价于
x = x.permute(0, 2, 3, 1)
```

#### 2.1.2 合并维度

```python
x = rearrange(x, "b c h w -> b c (h w)")
# [b, c, h, w] -> [b, c, h*w]
```

#### 2.1.3 拆分维度

```python
x = rearrange(
    x,
    "b (h w) c -> b h w c",
    h=14,
    w=14,
)
#[b, 196, c] -> [b, 14, 14, c]
```

#### 2.1.4 同时拆分和换序

```python
x = rearrange(
    x,
    "b (h ph) (w pw) c -> b (h w) (ph pw c)",
    ph=16,
    pw=16,
)
# [b, 224, 224, 3] -> [b, 196, 768]
# 196 = 14 × 14
# 768 = 16 × 16 × 3
```

### 2.2 `repeat`

负责复制维度或扩展数据。

例如把一张灰度图复制成三个通道：

```python
from einops import repeat

x = repeat(x, "b h w -> b c h w", c=3)
# [b, h, w] -> [b, 3, h, w]
```

再例如重复 batch：

```python
x = repeat(x, "b d -> (copy b) d", copy=4)
[b, d] -> [4*b, d]
```

### 2.3 `reduce`

负责在某些维度上聚合。

例如全局平均池化：

```python
from einops import reduce

x = reduce(x, "b c h w -> b c", "mean")
# 等价于
x = x.mean(dim=(2, 3))
```

也可以对 patch 做池化：

```
x = reduce(
    x,
    "b c (h ph) (w pw) -> b c h w",
    "mean",
    ph=2,
    pw=2,
)
```

这相当于进行 `2 × 2` 平均池化，`reduce` 支持 `mean`、`sum`、`min`、`max`、`prod` 等归约操作。

## 3 重要 API `einsum`

`einsum` 是 Einstein summation（爱因斯坦求和）的缩写。它用一条表达式同时描述：

1. 输入张量的每个轴叫什么；

2. 哪些轴需要对齐并相乘；

3. 哪些轴需要求和消去；

4. 输出保留哪些轴，以及这些轴的顺序。

它不只可以做矩阵乘法，还可以表达点积、外积、批量矩阵乘法、线性投影、注意力分数、加权求和等操作。

### 3.1 基本语法

```python
# einops
y = einsum(x, weight, "batch in_dim, out_dim in_dim -> batch out_dim")

# PyTorch 中含义相同的写法
y = torch.einsum("bi,oi->bo", x, weight)
```

阅读表达式时要记住：

- 逗号 `,` 分隔不同输入张量；

- 箭头 `->` 左边描述输入，右边描述输出；

- **左边出现、右边也出现的轴会保留；**

- **左边出现、右边没有出现的轴会被求和消去；**

- 同名轴表示这些位置属于同一个维度，需要对齐；

- 输出轴的排列顺序决定结果的维度顺序。

### 3.2 矩阵乘法演示

假设输入序列 `data` 的形状为 `[batch, sequence, in_dim]`，权重 `weight` 的形状为 `[out_dim, in_dim]`：

```python
data.shape == (batch, sequence, in_dim)
weight.shape == (out_dim, in_dim)

y = einsum(
    data,
    weight,
    "batch sequence in_dim, out_dim in_dim -> batch sequence out_dim",
)

y.shape == (batch, sequence, out_dim)
```

- `batch`、`sequence` 只来自 `data`，并出现在输出中，所以原样保留；

- `out_dim` 来自 `weight`，并出现在输出中，所以原样保留；

- `in_dim` 同时出现在两个输入中，但没有出现在输出中，所以先逐元素相乘，再沿 `in_dim` 求和。

对应的数学公式为：

$$
y_{b,s,o} = \sum_i data_{b,s,i} \cdot weight_{o,i}
$$

等价的 PyTorch 写法是：

```python
y = data @ weight.T
```

### 3.3 基础用法

#### 3.3.1 转置与调整轴顺序

```python
# x: [batch, height, width, channel]
y = einsum(x, "batch height width channel -> batch channel height width")
# y: [batch, channel, height, width]
```

这里只改变轴顺序，没有乘法和求和。实际项目中单纯调整维度通常优先使用语义更明确的 `rearrange`：

```python
y = rearrange(x, "batch height width channel -> batch channel height width")
```

#### 3.3.2 对所有元素求和

```python
# x: [height, width]
total = einsum(x, "height width ->")
# total 是标量，等价于 x.sum()
```

箭头右边为空，说明没有任何轴需要保留。

#### 3.3.3 沿指定轴求和

```python
# x: [batch, sequence, dim]
y = einsum(x, "batch sequence dim -> batch dim")
# y: [batch, dim]，等价于 x.sum(dim=1)
```

`sequence` 没有出现在输出中，因此沿该轴求和。

`einsum` 只能做求和归约；如果需要 `mean`、`max`、`min` 等归约，应使用 `reduce`。

#### 3.3.4 向量点积

```python
# x, y: [dim]
score = einsum(x, y, "dim, dim ->")
# score: 标量，等价于 (x * y).sum()
```

数学形式：

$$
score = \sum_d x_d y_d
$$

#### 3.3.5 批量点积

```python
# x, y: [batch, dim]
score = einsum(x, y, "batch dim, batch dim -> batch")
# score: [batch]
```

`batch` 被保留，`dim` 被求和，因此每个样本分别得到一个点积结果。

#### 3.3.6 外积

```python
# x: [row]
# y: [column]
matrix = einsum(x, y, "row, column -> row column")
# matrix: [row, column]
```

两个输入没有同名轴，也没有轴被消去，因此会计算所有元素两两相乘：

$$
matrix_{i,j} = x_i y_j
$$

#### 3.3.7 逐元素乘法

```python
# x, y: [batch, dim]
z = einsum(x, y, "batch dim, batch dim -> batch dim")
# z: [batch, dim]，等价于 x * y
```

所有轴都被保留，所以不会发生求和。

#### 3.3.8 矩阵与向量相乘

```python
# matrix: [row, dim]
# vector: [dim]
y = einsum(matrix, vector, "row dim, dim -> row")
# y: [row]
```

#### 3.3.9 矩阵乘法

```python
# a: [row, inner]
# b: [inner, column]
c = einsum(a, b, "row inner, inner column -> row column")
# c: [row, column]，等价于 a @ b
```

`inner` 是两个矩阵连接的维度，因为没有出现在输出中，所以会被求和。

#### 3.3.10 批量矩阵乘法

```python
# a: [batch, row, inner]
# b: [batch, inner, column]
c = einsum(
    a,
    b,
    "batch row inner, batch inner column -> batch row column",
)
# c: [batch, row, column]，等价于 torch.bmm(a, b)
```

这里的 `batch` 出现在两个输入和输出中，表示每个 batch 内部独立进行矩阵乘法，而不是沿 batch 求和。

#### 3.3.11 矩阵的迹与对角线

```python
# matrix: [dim, dim]
trace = einsum(matrix, "dim dim ->")
# trace: 标量，等价于 matrix.diagonal().sum()

diagonal = einsum(matrix, "dim dim -> dim")
# diagonal: [dim]，取出对角线
```

同一个输入中重复使用相同轴名，表示只选择这些轴索引相等的位置。是否继续求和，取决于该轴是否出现在输出中。

### 3.4 广播与任意前导维度

#### 3.4.1 不同输入之间广播

```python
# x: [batch, sequence, dim]
# scale: [dim]
y = einsum(x, scale, "batch sequence dim, dim -> batch sequence dim")
# y: [batch, sequence, dim]
```

`scale` 没有 `batch` 和 `sequence` 轴，但会应用到每个 batch 、每个 token 上，效果等价于 `x * scale`。

#### 3.4.2 使用省略号 `...`

当只关心最后几个轴，而前面可能有任意数量的维度时，可以使用 `...`：

```python
# data: [..., in_dim]，例如 [batch, sequence, in_dim]
# weight: [out_dim, in_dim]
y = einsum(weight, data, "out_dim in_dim, ... in_dim -> ... out_dim")
# y: [..., out_dim]
```

这样同一个表达式可以同时处理：

- `[in_dim] -> [out_dim]`；

- `[batch, in_dim] -> [batch, out_dim]`；

- `[batch, sequence, in_dim] -> [batch, sequence, out_dim]`。

### 3.5 如何快速设计表达式

面对一个新问题时，可以按下面的顺序写：

1. **给输入的每个轴命名**，并确保轴名数量与张量维度数量一致；

2. **给含义相同、需要对齐的轴使用相同名字**；

3. **把最终想保留的轴写到箭头右边**；

4. **检查需要求和的轴是否只出现在左边**；

5. **按照期望的输出形状排列右边的轴名**。

例如，要计算两组 token 的两两点积：

```text
x: [batch, token_x, dim]
y: [batch, token_y, dim]
```

- `batch` 需要保留；

- `token_x` 和 `token_y` 都需要保留，才能得到两两组合；

- `dim` 用于计算点积，需要求和消去。

所以表达式为：

```python
"batch token_x dim, batch token_y dim -> batch token_x token_y"
```

### 3.6 常用表达式速查

| 操作 | 输入形状 | 表达式 | 输出形状 |
| --- | --- | --- | --- |
| 全部求和 | `[h, w]` | `"h w ->"` | `[]` |
| 沿序列求和 | `[b, s, d]` | `"b s d -> b d"` | `[b, d]` |
| 向量点积 | `[d]`, `[d]` | `"d, d ->"` | `[]` |
| 批量点积 | `[b, d]`, `[b, d]` | `"b d, b d -> b"` | `[b]` |
| 外积 | `[i]`, `[j]` | `"i, j -> i j"` | `[i, j]` |
| 逐元素乘法 | `[b, d]`, `[b, d]` | `"b d, b d -> b d"` | `[b, d]` |
| 矩阵乘法 | `[i, k]`, `[k, j]` | `"i k, k j -> i j"` | `[i, j]` |
| 批量矩阵乘法 | `[b, i, k]`, `[b, k, j]` | `"b i k, b k j -> b i j"` | `[b, i, j]` |
| 线性投影 | `[b, s, i]`, `[o, i]` | `"b s i, o i -> b s o"` | `[b, s, o]` |
| 两两相似度 | `[b, n, d]`, `[b, m, d]` | `"b n d, b m d -> b n m"` | `[b, n, m]` |
| 矩阵的迹 | `[d, d]` | `"d d ->"` | `[]` |

完整参数说明与后端支持情况可查阅 [`einops.einsum` 官方文档](https://einops.rocks/api/einsum/)。
