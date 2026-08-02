# NumPy

NumPy 的核心数据结构是多维数组 `ndarray`。这篇笔记整理科学计算和深度学习中常用的数组操作，重点记录形状变化、广播规则以及容易混淆的行为。

```python
import numpy as np
```

## 1. 数组基础

### 1.1 创建数组

```python
np.array([1, 2, 3])                 # 从 Python 序列创建
np.zeros((2, 3))                    # 全 0 数组
np.ones((2, 3))                     # 全 1 数组
np.full((2, 3), 7)                  # 使用指定值填充
np.empty((2, 3))                    # 只分配内存，不初始化值

np.arange(0, 10, 2)                 # [0, 2, 4, 6, 8]
np.linspace(0, 1, 5)                # 在 [0, 1] 内生成 5 个等距点

np.zeros_like(x)                    # 形状和 dtype 与 x 相同
np.ones_like(x)
np.full_like(x, 7)
```

浮点数范围优先使用 `np.linspace`。`np.arange` 使用浮点步长时可能受浮点精度影响，终点也不一定符合直觉。

### 1.2 常用属性

```python
x = np.zeros((2, 3, 4), dtype=np.float32)

x.shape      # (2, 3, 4)，各维度大小
x.ndim       # 3，维度数量
x.size       # 24，元素总数
x.dtype      # dtype('float32')
x.itemsize   # 4，单个元素占用的字节数
x.nbytes     # 96，数组元素占用的总字节数
```

### 1.3 数据类型转换

```python
x = np.array([1, 2, 3])
y = x.astype(np.float32)
```

`astype` 默认返回新数组。常用类型包括 `np.float32`、`np.float64`、`np.int32`、`np.int64` 和 `np.bool_`。

## 2. 索引与切片

### 2.1 基本索引

```python
x = np.arange(12).reshape(3, 4)

x[0, 1]      # 第 0 行、第 1 列的元素
x[0]         # 第 0 行，shape 为 (4,)
x[:, 1]      # 第 1 列，shape 为 (3,)
x[-1]        # 最后一行
```

### 2.2 切片

切片写作 `start:stop:step`，区间遵循左闭右开规则，即包含 `start`、不包含 `stop`。

```python
x[1:3]       # 第 1、2 行
x[:, ::2]    # 所有行，每隔一列取一个元素
x[::-1]      # 反转第 0 维
```

在图像或特征图中截取局部窗口：

```python
# x.shape == (N, C, H, W)
patch = x[:, :, i:i + h, j:j + w]
```

其中 `:` 表示保留该维度的全部元素，返回结果仍是四维数组。

基本切片通常返回原数组的**视图（view）**，修改切片可能同时修改原数组：

```python
x = np.arange(5)
y = x[1:3]
y[0] = 100

x  # array([0, 100, 2, 3, 4])
```

需要独立数据时显式复制：

```python
y = x[1:3].copy()
```

### 2.3 布尔索引

```python
x = np.array([-2, -1, 0, 1, 2])

mask = x > 0
x[mask]          # array([1, 2])
x[x < 0] = 0     # 将负数原地截断为 0
```

直接使用布尔索引读取数据时，结果通常是一维副本。它常用于过滤无效值、实现掩码和条件赋值。

### 2.4 整数数组索引

```python
x = np.array([10, 20, 30, 40])
x[[3, 1, 1]]     # array([40, 20, 20])
```

整数数组索引属于高级索引，返回的是副本，而不是视图。

## 3. 形状变换

### 3.1 `reshape`

`reshape` 改变数组形状，但不改变元素的逻辑顺序。新形状的元素总数必须与原数组相同。

```python
x = np.arange(12)

x.reshape(3, 4)       # shape: (3, 4)
x.reshape(3, -1)      # 自动推断为 (3, 4)
x.reshape(-1)         # 展平为一维
```

`reshape` 会尽量返回视图，但当内存布局不满足要求时也可能产生副本，因此不要依赖它一定共享内存。

### 3.2 增加和删除维度

```python
x = np.arange(6).reshape(2, 3)  # shape: (2, 3)

x[None, ...].shape              # (1, 2, 3)
x[:, None, :].shape             # (2, 1, 3)
np.expand_dims(x, axis=0).shape # (1, 2, 3)

y = np.zeros((1, 2, 1, 3))
np.squeeze(y).shape             # (2, 3)，删除所有大小为 1 的维度
np.squeeze(y, axis=0).shape     # (2, 1, 3)，只删除指定维度
```

不确定其他维度是否为 `1` 时，优先为 `squeeze` 指定 `axis`，避免误删维度。

### 3.3 转置与维度交换

```python
# x.shape == (N, H, W, C)
y = x.transpose(0, 3, 1, 2)  # NHWC -> NCHW

z = np.swapaxes(x, 1, 2)     # 只交换第 1、2 维
matrix_t = matrix.T           # 二维矩阵转置
```

`transpose` 需要给出所有轴的新顺序。它通常返回视图，不改变数据值，但会改变内存访问顺序。

注意：一维数组的 `.T` 不会变成列向量。

```python
x = np.array([1, 2, 3])
x.T.shape             # (3,)
x[:, None].shape      # (3, 1)，列向量
x[None, :].shape      # (1, 3)，行向量
```

### 3.4 拼接与堆叠

```python
a = np.ones((2, 3))
b = np.zeros((2, 3))

np.concatenate([a, b], axis=0).shape  # (4, 3)，沿已有轴拼接
np.concatenate([a, b], axis=1).shape  # (2, 6)
np.stack([a, b], axis=0).shape        # (2, 2, 3)，创建新轴
```

- `concatenate`：沿已有维度连接，除连接轴外的维度必须相同。
- `stack`：创建一个新维度，所有输入数组的形状必须完全相同。

## 4. 广播机制

广播允许 NumPy 在逐元素运算时自动扩展大小为 `1` 的维度，而不必真正复制数据。

### 4.1 广播规则

比较两个数组的形状时，从最后一个维度开始向前对齐。每组对应维度必须满足以下条件之一：

1. 两个维度大小相同；
2. 其中一个维度大小为 `1`；
3. 其中一个数组缺少该维度，此时按大小为 `1` 处理。

例如：

```text
(8, 1, 6, 1)
(   7, 1, 5)
-------------
(8, 7, 6, 5)
```

### 4.2 常见示例

二维数组按列添加偏置：

```python
# x.shape == (N, D)
# bias.shape == (D,)
y = x + bias                 # bias 视为 (1, D)
```

NCHW 特征图按通道添加偏置：

```python
# x.shape == (N, C, H, W)
# bias.shape == (C,)
channel_bias = bias.reshape(1, C, 1, 1)
y = x + channel_bias
```

形状为 `(C,)` 的数组会从最后一维开始与 `(N, C, H, W)` 对齐，因此不能直接表示 NCHW 格式的通道偏置。

调试广播问题时可先查看目标形状：

```python
np.broadcast_shapes((8, 1, 6, 1), (7, 1, 5))
# (8, 7, 6, 5)
```

## 5. 数值运算

### 5.1 逐元素运算

```python
a + b
a - b
a * b
a / b
a ** 2

np.maximum(a, b)
np.minimum(a, b)
np.exp(a)
np.log(a)
np.sqrt(a)
np.abs(a)
```

输入数组只要能够广播，就可以执行逐元素运算，输出形状是广播后的形状。

深度学习中的常见例子：

```python
relu = np.maximum(x, 0)
sigmoid = 1 / (1 + np.exp(-x))
squared_error = (y_pred - y_true) ** 2
```

### 5.2 矩阵乘法

```python
c = a @ b
c = np.matmul(a, b)  # 与上式等价
```

二维矩阵乘法要求内层维度相同：

```text
(m, k) @ (k, n) -> (m, n)
```

必须区分逐元素乘法和矩阵乘法：

```python
a * b   # 逐元素乘法
a @ b   # 矩阵乘法
```

对于高维数组，`matmul` 将最后两个维度视为矩阵，其余维度按广播规则处理。

### 5.3 点积与通用求和

```python
np.dot(a, b)
np.einsum("ik,kj->ij", a, b)
```

`np.dot` 对高维数组的规则与 `matmul` 不同。表达矩阵乘法时优先使用 `@`；需要明确描述复杂维度收缩时使用 `np.einsum`。

## 6. 归约与统计

归约操作沿指定轴汇总数据：

```python
x.sum(axis=0)
x.mean(axis=1)
x.max(axis=-1)
x.min(axis=-1)
x.argmax(axis=1)
x.std(axis=0)
x.var(axis=0)
```

`axis` 表示执行归约的维度。默认情况下该维度会从结果中消失；设置 `keepdims=True` 可以保留大小为 `1` 的维度，便于后续广播。

```python
# x.shape == (N, C, H, W)
channel_mean = x.mean(axis=(0, 2, 3), keepdims=True)
# channel_mean.shape == (1, C, 1, 1)

x_centered = x - channel_mean
```

多个轴可以通过元组一次归约：

```python
spatial_mean = x.mean(axis=(2, 3))  # shape: (N, C)
```

含有缺失值 `NaN` 时，可以使用忽略 `NaN` 的版本：

```python
np.nanmean(x)
np.nansum(x)
np.nanmax(x)
```

## 7. 条件选择与掩码

### 7.1 `where`

```python
result = np.where(condition, value_if_true, value_if_false)

relu = np.where(x > 0, x, 0)
```

只传入条件时，`np.where(condition)` 返回满足条件的元素索引；若只是查找索引，使用 `np.nonzero(condition)` 语义更清晰。

### 7.2 多条件组合

NumPy 数组的条件需要使用 `&`、`|` 和 `~`，并为每个比较表达式添加括号：

```python
mask = (x >= 0) & (x <= 1)
selected = x[mask]
```

不能使用 Python 的 `and`、`or` 或链式比较 `0 <= x <= 1`。

### 7.3 裁剪数值范围

```python
clipped = np.clip(x, 0, 1)
```

当目标只是限制上下界时，`np.clip` 比嵌套使用 `minimum`、`maximum` 或 `where` 更直接。

## 8. 排序与查找

```python
np.sort(x, axis=-1)          # 返回排序后的值
np.argsort(x, axis=-1)       # 返回排序索引
np.argmax(x, axis=-1)        # 最大值索引
np.argmin(x, axis=-1)        # 最小值索引
np.unique(x)                 # 去重并排序
np.nonzero(x)                # 非零元素索引
```

获取一维数组中最大的 `k` 个元素时，`argpartition` 通常比完整排序更高效：

```python
k = 3
indices = np.argpartition(x, -k)[-k:]
top_k = x[indices]  # 这 k 个元素内部不保证有序
```

## 9. 随机数

新代码推荐使用独立的随机数生成器，便于控制和复现实验：

```python
rng = np.random.default_rng(seed=42)

rng.random((2, 3))           # [0, 1) 均匀分布
rng.normal(0, 1, (2, 3))     # 正态分布
rng.integers(0, 10, size=5)  # [0, 10) 内的随机整数
rng.permutation(10)          # 0 到 9 的随机排列
```

## 10. 数值检查与调试

```python
np.isnan(x)       # 判断元素是否为 NaN
np.isinf(x)       # 判断元素是否为正无穷或负无穷
np.isfinite(x)    # 判断元素是否为有限值
```

结合 `any` 可以快速判断整个数组是否存在异常值：

```python
if not np.isfinite(x).all():
    raise ValueError("x contains NaN or infinity")
```

比较浮点数组时不要直接使用 `==`，应使用带容差的比较：

```python
np.isclose(a, b)
np.allclose(a, b)
```

在测试中可以使用：

```python
np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
```

## 11. 常见场景

### 11.1 标准化

```python
# 按最后一维标准化
mean = x.mean(axis=-1, keepdims=True)
std = x.std(axis=-1, keepdims=True)
x_normalized = (x - mean) / (std + 1e-8)
```

### 11.2 稳定计算 Softmax

先减去最大值可以避免 `exp` 上溢：

```python
shifted = logits - logits.max(axis=-1, keepdims=True)
exp_values = np.exp(shifted)
probabilities = exp_values / exp_values.sum(axis=-1, keepdims=True)
```

### 11.3 One-hot 编码

```python
labels = np.array([2, 0, 1])
one_hot = np.eye(3, dtype=np.float32)[labels]

# one_hot ==
# [[0., 0., 1.],
#  [1., 0., 0.],
#  [0., 1., 0.]]
```

### 11.4 批量线性变换

```python
# x.shape == (batch, sequence, d_in)
# weight.shape == (d_in, d_out)
# bias.shape == (d_out,)
y = x @ weight + bias
# y.shape == (batch, sequence, d_out)
```

### 11.5 二维卷积的局部计算

下面这一行组合了切片、逐元素乘法和归约，是二维卷积单个位置的核心计算：

```python
# region.shape == kernel.shape == (C, kernel_h, kernel_w)
output[i, j] = (region * kernel).sum() + bias
```

完整实现还需要处理批次、多个输出通道、步长和填充。

## 12. 视图、副本与原地修改

是否共享内存会影响正确性和性能：

- 基本切片通常返回视图；
- 布尔索引和整数数组索引返回副本；
- `reshape`、`transpose` 可能返回共享内存的视图；
- `copy()` 明确创建独立副本。

可以用下面的方法辅助判断：

```python
np.shares_memory(a, b)
```

带 `out` 参数或对切片赋值会修改原数组，应避免在仍需使用原始数据时误操作：

```python
np.add(a, b, out=a)  # 将结果写回 a
x[x < 0] = 0         # 原地修改 x
```

## 13. 常用 API 速查表

| 需求 | 推荐写法 |
| --- | --- |
| 查看形状 | `x.shape` |
| 改变形状 | `x.reshape(...)` |
| 展平数组 | `x.reshape(-1)` 或 `x.ravel()` |
| 复制数据 | `x.copy()` |
| 调整轴顺序 | `x.transpose(...)` |
| 增加维度 | `x[None, ...]` 或 `np.expand_dims(x, axis)` |
| 删除大小为 1 的维度 | `np.squeeze(x, axis)` |
| 沿已有轴拼接 | `np.concatenate([...], axis=...)` |
| 沿新轴堆叠 | `np.stack([...], axis=...)` |
| 逐元素乘法 | `a * b` |
| 矩阵乘法 | `a @ b` |
| 按轴求和或均值 | `x.sum(axis=...)`、`x.mean(axis=...)` |
| 保留归约维度 | `keepdims=True` |
| 条件选择 | `np.where(condition, a, b)` |
| 限制数值范围 | `np.clip(x, lower, upper)` |
| 查找排序索引 | `np.argsort(x)` |
| 判断能否广播 | `np.broadcast_shapes(...)` |
| 检查异常值 | `np.isfinite(x).all()` |
| 浮点近似比较 | `np.allclose(a, b)` |

遇到形状错误时，优先打印参与运算数组的 `shape`，然后按照“从右向左对齐”的规则逐维检查。多数 NumPy 维度问题都可以通过这一步快速定位。
