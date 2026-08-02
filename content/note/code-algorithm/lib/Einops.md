# `Einops` 库函数

`einops` 是一个用于**清晰地表达张量维度变换**的 Python 库，用统一的字符串表达式完成：`reshape`, `transpose / permute`, `squeeze / unsqueeze` 等张量操作的库。详细内容可以参见[官方文档说明](https://einops.rocks/)

## 优化方面

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

## 常用 API

### 2.1 `rearrange`

负责改变张量维度的组织方式。

#### 交换维度

```python
x = rearrange(x, "b c h w -> b h w c")
# 等价于
x = x.permute(0, 2, 3, 1)
```

#### 合并维度

```python
x = rearrange(x, "b c h w -> b c (h w)")
# [b, c, h, w] -> [b, c, h*w]
```

#### 拆分维度

```python
x = rearrange(
    x,
    "b (h w) c -> b h w c",
    h=14,
    w=14,
)
#[b, 196, c] -> [b, 14, 14, c]
```

#### 同时拆分和换序

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
