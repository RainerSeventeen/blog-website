# PyTorch 笔记

## 构造 API

### `torch.tensor(data)`
直接从 Python 列表或 numpy 数组创建张量，会复制数据。

```python
torch.tensor([1, 2, 3])           # tensor([1, 2, 3])
torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # shape (2, 2)
torch.tensor([1, 2], dtype=torch.float32)  # 指定数据类型
```

### `torch.zeros(*size)`
创建全 0 张量，常用于初始化偏置或掩码。

```python
torch.zeros(3)         # tensor([0., 0., 0.])
torch.zeros(2, 3)      # shape (2, 3) 的全 0 矩阵
torch.zeros_like(x)    # 与 x 形状相同的全 0 张量
```

### `torch.ones(*size)`
创建全 1 张量，常用于初始化权重或掩码。

```python
torch.ones(3)          # tensor([1., 1., 1.])
torch.ones(2, 3)       # shape (2, 3) 的全 1 矩阵
torch.ones_like(x)     # 与 x 形状相同的全 1 张量
```

### `torch.arange(start, end, step)`
创建等差数列张量，类似 Python 的 `range`，**不包含** end。

```python
torch.arange(5)           # tensor([0, 1, 2, 3, 4])
torch.arange(1, 5)        # tensor([1, 2, 3, 4])
torch.arange(0, 1, 0.2)   # tensor([0.0, 0.2, 0.4, 0.6, 0.8])
```

### `torch.randn(*size)`
从标准正态分布 N(0, 1) 中随机采样，常用于参数初始化。

```python
torch.randn(3)        # 3 个随机数
torch.randn(2, 3)     # shape (2, 3) 的随机矩阵
torch.randn_like(x)   # 与 x 形状相同的随机张量
```

### `torch.empty(*size)`
创建**未初始化**张量，内存中原有什么值就是什么值。比 zeros/ones 快，但必须后续赋值才能使用。

```python
torch.empty(3)      # 值不确定，可能是任意数
torch.empty(2, 3)   # shape (2, 3)，常用于后续 fill_ 操作
```

---

## 形状变换 API

### `x.view(*shape)`
重塑张量形状，**要求内存连续**（contiguous）。不复制数据，只改变解释方式。

```python
x = torch.arange(6)       # shape (6,)
x.view(2, 3)               # shape (2, 3)
x.view(3, -1)              # -1 表示自动推断，shape (3, 2)
# 若 x 不连续，需先调用 x.contiguous().view(...)
```

### `x.reshape(*shape)`
重塑张量形状，**不要求内存连续**。可能返回视图也可能复制数据，比 view 更安全。

```python
x = torch.arange(6)
x.reshape(2, 3)     # shape (2, 3)
x.reshape(-1)       # 展平为 1D
```

### `x.transpose(dim0, dim1)`
交换两个指定维度，**只能交换两个维度**。

```python
x = torch.randn(2, 3)
x.transpose(0, 1)          # shape (3, 2)，等价于转置

x = torch.randn(2, 3, 4)
x.transpose(1, 2)          # shape (2, 4, 3)，交换第 1、2 维
```

### `x.permute(*dims)`
按指定顺序重排**所有维度**，是 transpose 的泛化版本。

```python
x = torch.randn(2, 3, 4)   # (batch, seq, dim)
x.permute(0, 2, 1)          # (batch, dim, seq)，shape (2, 4, 3)
x.permute(2, 0, 1)          # shape (4, 2, 3)
```

### `x.unsqueeze(dim)`
在指定位置**插入**一个大小为 1 的维度，常用于扩充 batch 或 channel 维。

```python
x = torch.tensor([1, 2, 3])   # shape (3,)
x.unsqueeze(0)                  # shape (1, 3)，在最前面加维度
x.unsqueeze(1)                  # shape (3, 1)，在最后面加维度
x.unsqueeze(-1)                 # 等价于 unsqueeze(1)
```

### `x.squeeze(dim)`
**删除**大小为 1 的维度。不指定 dim 则删除所有大小为 1 的维度。

```python
x = torch.randn(1, 3, 1, 4)
x.squeeze()         # shape (3, 4)，删除所有 size=1 的维度
x.squeeze(0)        # shape (3, 1, 4)，只删除第 0 维
x.squeeze(2)        # shape (1, 3, 4)，只删除第 2 维
```

---

## 掩码 API

### `torch.triu(input, diagonal=0)`
取矩阵的**上三角**部分（含对角线），其余元素置 0。diagonal > 0 向右偏移，< 0 向左偏移。

```python
x = torch.ones(4, 4)
torch.triu(x)           # 上三角矩阵（含主对角线）
torch.triu(x, diagonal=1)  # 严格上三角（不含主对角线），常用于 causal mask
```

### `torch.tril(input, diagonal=0)`
取矩阵的**下三角**部分（含对角线），其余元素置 0。

```python
x = torch.ones(4, 4)
torch.tril(x)              # 下三角矩阵（含主对角线）
torch.tril(x, diagonal=-1) # 严格下三角（不含主对角线）
```

### `tensor.masked_fill(mask, value)`
将 mask 中为 `True` 的位置填充为指定 value，**原地操作用** `masked_fill_`。常与 triu/tril 配合实现注意力掩码。

```python
x = torch.randn(3, 3)
mask = torch.triu(torch.ones(3, 3), diagonal=1).bool()  # 严格上三角为 True
x.masked_fill(mask, float('-inf'))  # 上三角位置填充 -inf，用于 causal attention
```

### 比较运算符 `==, !=, >, <`
逐元素比较，返回 bool 型张量，常用于生成掩码。

```python
x = torch.tensor([1, 2, 3, 0, 0])
x == 0          # tensor([False, False, False,  True,  True])，padding mask
x > 1           # tensor([False,  True,  True, False, False])
x != 0          # tensor([ True,  True,  True, False, False])
```

---

## 运算 API

### `tensor.sum(dim, keepdim)`
沿指定维度求和。不指定 dim 则对所有元素求和。

```python
x = torch.tensor([[1., 2., 3.], [4., 5., 6.]])  # shape (2, 3)
x.sum()             # tensor(21.)，所有元素之和
x.sum(dim=0)        # tensor([5., 7., 9.])，沿行求和，shape (3,)
x.sum(dim=1)        # tensor([6., 15.])，沿列求和，shape (2,)
x.sum(dim=1, keepdim=True)  # shape (2, 1)，保持维度
```

### `tensor.mean(dim, keepdim)`
沿指定维度求均值，用法同 sum。

```python
x = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
x.mean()             # tensor(3.5)
x.mean(dim=1)        # tensor([2., 5.])
x.mean(dim=1, keepdim=True)  # shape (2, 1)
```

### `tensor.max(dim, keepdim)`
沿指定维度取最大值。不指定 dim 返回标量；指定 dim 返回 `(values, indices)` 的命名元组。

```python
x = torch.tensor([[1., 3., 2.], [5., 4., 6.]])
x.max()              # tensor(6.)，全局最大值
values, indices = x.max(dim=1)
# values: tensor([3., 6.])，每行最大值
# indices: tensor([1, 2])，最大值所在列索引
```

### `torch.nn.functional.softmax(x, dim)`
对指定维度做 softmax，将数值转为概率分布（和为 1）。

```python
import torch.nn.functional as F

x = torch.tensor([1.0, 2.0, 3.0])
F.softmax(x, dim=0)   # tensor([0.0900, 0.2447, 0.6652])

# 在注意力中对最后一维做 softmax
scores = torch.randn(2, 4, 4)   # (batch, seq, seq)
attn = F.softmax(scores, dim=-1) # 每行和为 1
```

### `tensor.argmax(dim)`
返回指定维度上最大值的**索引**，不返回值本身。

```python
x = torch.tensor([[1., 3., 2.], [5., 4., 6.]])
x.argmax()           # tensor(5)，全局最大值的扁平索引
x.argmax(dim=1)      # tensor([1, 2])，每行最大值所在列索引
# 常用于分类任务取预测类别：pred = logits.argmax(dim=-1)
```

---

## torch API

### `torch.clamp(input, min, max)`
将张量中所有数值**限制在 [min, max] 范围内**，超出范围的值会被截断到边界。

```python
x = torch.tensor([-1., 0., 0.5, 2., 3.])
torch.clamp(x, 0, 1)       # tensor([0., 0., 0.5, 1., 1.])
torch.clamp(x, min=0)      # tensor([0., 0., 0.5, 2., 3.])，只限下界
torch.clamp(x, max=1)      # tensor([-1., 0., 0.5, 1., 1.])，只限上界

# 实际场景：把网络输出限制在 >= 1
clipped_peds = torch.clamp(net(features), 1, float("inf"))
```

### `torch.arange(start, end, step)`
创建等差数列，类似 Python 的 `range`，**不包含** end。

```python
torch.arange(5)           # tensor([0, 1, 2, 3, 4])
torch.arange(1, 5)        # tensor([1, 2, 3, 4])
torch.arange(0, 1, 0.2)   # tensor([0.0, 0.2, 0.4, 0.6, 0.8])
```

---

## tensor API

### `tensor.repeat(*sizes)`
沿每个维度**复制数据**指定次数，会真实分配新内存（区别于 expand 的零拷贝广播）。

```python
x = torch.tensor([1, 2, 3])   # shape (3,)
x.repeat(2)                    # [1,2,3,1,2,3]，在唯一维度重复 2 次

x = torch.tensor([[1, 2, 3]])  # shape (1, 3)
x.repeat(2, 1)   # [[1,2,3],[1,2,3]]，第 0 维重复 2 次，第 1 维重复 1 次
x.repeat(2, 3)   # [[1,2,3,1,2,3,1,2,3],[1,2,3,1,2,3,1,2,3]]

# 与 expand 的区别：
# expand 只能扩展 size=1 的维度，不复制数据（节省内存）
# repeat 可以重复任意维度，会复制数据
```

### `tensor.expand(*sizes)`
将 size 为 1 的维度**广播扩展**到指定大小，不复制数据（零拷贝），比 repeat 更高效。

```python
x = torch.tensor([[1], [2], [3]])  # shape (3, 1)
x.expand(3, 4)   # shape (3, 4)，每行的 1 个元素扩展为 4 个
# tensor([[1,1,1,1],[2,2,2,2],[3,3,3,3]])

# -1 表示该维度保持不变
x.expand(-1, 4)  # 等价于 x.expand(3, 4)
```

### `tensor.contiguous()`
返回在内存中**连续存储**的张量副本。transpose/permute 后内存不连续，需要先调用此方法再使用 view。

```python
x = torch.randn(2, 3)
y = x.transpose(0, 1)     # y 内存不连续
y.contiguous().view(-1)   # 先变连续，再 reshape
# 或直接用 reshape（内部会自动处理）
```
