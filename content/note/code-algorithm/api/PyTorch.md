# PyTorch

## 1 构造 API

### 1.1 `torch.tensor(data)`
直接从 Python 列表或 numpy 数组创建张量，会复制数据。

```python
torch.tensor([1, 2, 3])           # tensor([1, 2, 3])
torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # shape (2, 2)
torch.tensor([1, 2], dtype=torch.float32)  # 指定数据类型
```

### 1.2 `torch.zeros(*size)`
创建全 0 张量，常用于初始化偏置或掩码。

```python
torch.zeros(3)         # tensor([0., 0., 0.])
torch.zeros(2, 3)      # shape (2, 3) 的全 0 矩阵
torch.zeros_like(x)    # 与 x 形状相同的全 0 张量
```

### 1.3 `torch.ones(*size)`
创建全 1 张量，常用于初始化权重或掩码。

```python
torch.ones(3)          # tensor([1., 1., 1.])
torch.ones(2, 3)       # shape (2, 3) 的全 1 矩阵
torch.ones_like(x)     # 与 x 形状相同的全 1 张量
```

### 1.4 `torch.arange(start, end, step)`
创建等差数列张量，类似 Python 的 `range`，**不包含** end 。

```python
torch.arange(5)           # tensor([0, 1, 2, 3, 4])
torch.arange(1, 5)        # tensor([1, 2, 3, 4])
torch.arange(0, 1, 0.2)   # tensor([0.0, 0.2, 0.4, 0.6, 0.8])
```

### 1.5 `torch.randn(*size)`
从标准正态分布 N(0, 1) 中随机采样，常用于参数初始化。

```python
torch.randn(3)        # 3 个随机数
torch.randn(2, 3)     # shape (2, 3) 的随机矩阵
torch.randn_like(x)   # 与 x 形状相同的随机张量
```

### 1.6 `torch.empty(*size)`
创建**未初始化**张量，内存中原有什么值就是什么值。比 zeros/ones 快，但必须后续赋值才能使用。

```python
torch.empty(3)      # 值不确定，可能是任意数
torch.empty(2, 3)   # shape (2, 3)，常用于后续 fill_ 操作
```

---

## 2 形状变换 API

### 2.1 `x.view(*shape)`
重塑张量形状，**要求内存连续**（contiguous）。不复制数据，只改变解释方式。

```python
x = torch.arange(6)       # shape (6,)
x.view(2, 3)               # shape (2, 3)
x.view(3, -1)              # -1 表示自动推断，shape (3, 2)
# 若 x 不连续，需先调用 x.contiguous().view(...)
```

### 2.2 `x.reshape(*shape)`
重塑张量形状，**不要求内存连续**。可能返回视图也可能复制数据，比 view 更安全。

```python
x = torch.arange(6)
x.reshape(2, 3)     # shape (2, 3)
x.reshape(-1)       # 展平为 1D
```

### 2.3 `x.transpose(dim0, dim1)`
交换两个指定维度，**只能交换两个维度**。

```python
x = torch.randn(2, 3)
x.transpose(0, 1)          # shape (3, 2)，等价于转置

x = torch.randn(2, 3, 4)
x.transpose(1, 2)          # shape (2, 4, 3)，交换第 1、2 维
```

### 2.4 `x.permute(*dims)`
按指定顺序重排**所有维度**，是 transpose 的泛化版本。

```python
x = torch.randn(2, 3, 4)   # (batch, seq, dim)
x.permute(0, 2, 1)          # (batch, dim, seq)，shape (2, 4, 3)
x.permute(2, 0, 1)          # shape (4, 2, 3)
```

### 2.5 `x.unsqueeze(dim)`
在指定位置**插入**一个大小为 1 的维度，常用于扩充 batch 或 channel 维。

```python
x = torch.tensor([1, 2, 3])   # shape (3,)
x.unsqueeze(0)                  # shape (1, 3)，在最前面加维度
x.unsqueeze(1)                  # shape (3, 1)，在最后面加维度
x.unsqueeze(-1)                 # 等价于 unsqueeze(1)
```

### 2.6 `x.squeeze(dim)`
**删除**大小为 1 的维度。不指定 dim 则删除所有大小为 1 的维度。

```python
x = torch.randn(1, 3, 1, 4)
x.squeeze()         # shape (3, 4)，删除所有 size=1 的维度
x.squeeze(0)        # shape (3, 1, 4)，只删除第 0 维
x.squeeze(2)        # shape (1, 3, 4)，只删除第 2 维
```

---

## 3 索引与选择 API

### 3.1 基本索引与切片 `x[...]`
使用整数、切片 `start:stop:step`、省略号 `...` 或 `None` 访问张量。基本索引通常返回原张量的视图（view）。

```python
x = torch.arange(12).reshape(3, 4)  # shape (3, 4)

x[0]            # tensor([0, 1, 2, 3])，shape (4,)
x[0, 1]         # tensor(1)
x[:, 1]         # tensor([1, 5, 9])，shape (3,)
x[1:3, ::2]     # tensor([[4, 6], [8, 10]])，shape (2, 2)
x[..., 0]       # 取最后一维的第 0 个元素
x[:, None, :]   # shape (3, 1, 4)，插入大小为 1 的维度
```

### 3.2 整数张量高级索引 `x[indices]`
使用整型张量批量选择元素。索引张量的 shape 会替换被索引维度的 shape；读取结果是新张量，而非视图。索引张量通常应为 `torch.long` 类型。

```python
x = torch.tensor([[10, 11], [20, 21], [30, 31]])  # shape (3, 2)
indices = torch.tensor([[0, 2], [1, 0]])          # shape (2, 2)

x[indices]
# tensor([[[10, 11], [30, 31]],
#         [[20, 21], [10, 11]]])，shape (2, 2, 2)

# indices 的 shape (2, 2) 替换 x 的第 0 维，保留最后一维 size=2
```

多个索引张量可按坐标逐元素配对取值：

```python
x = torch.tensor([[10, 11, 12], [20, 21, 22]])
rows = torch.tensor([0, 1])
cols = torch.tensor([2, 0])

x[rows, cols]   # tensor([12, 20])
```

### 3.3 布尔索引 `x[mask]`
使用 bool 型张量筛选元素。`mask` 与 `x` 形状相同时，读取结果会展平为一维张量。

```python
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
mask = x % 2 == 0

x[mask]         # tensor([2, 4, 6])
x[x > 3]        # tensor([4, 5, 6])
```

### 3.4 `torch.index_select(input, dim, index)`
沿 `dim` 维按一维 `index` 选择元素。`index` 必须是一维整型张量，输出仅在 `dim` 维的大小变为 `len(index)`。

```python
x = torch.tensor([[10, 11, 12], [20, 21, 22]])  # shape (2, 3)
index = torch.tensor([2, 0])

torch.index_select(x, dim=1, index=index)
# tensor([[12, 10], [22, 20]])，shape (2, 2)
```

### 3.5 `torch.gather(input, dim, index)`
沿 `dim` 维按 `index` 中的每个位置分别取值。`index` 必须为整型张量，且与 `input` 的维度数相同；输出 shape 与 `index` 相同。

```python
x = torch.tensor([[10, 11, 12], [20, 21, 22]])  # shape (2, 3)
index = torch.tensor([[2, 0], [1, 1]])          # shape (2, 2)

torch.gather(x, dim=1, index=index)
# tensor([[12, 10], [21, 21]])，shape (2, 2)
```

`index_select` 对同一维中的所有位置使用同一组索引；`gather` 可以为其他维度的每个位置提供不同索引。

### 3.6 `torch.take_along_dim(input, indices, dim)`
与 `gather` 类似，沿指定维度按索引取值；`indices` 可在非选择维度上与输入张量广播。`dim=None` 时会先将输入展平。

```python
x = torch.tensor([[10, 11, 12], [20, 21, 22]])
indices = torch.tensor([[2, 0], [1, 1]])

torch.take_along_dim(x, indices, dim=1)
# tensor([[12, 10], [21, 21]])
```

---

## 4 掩码 API

### 4.1 `torch.triu(input, diagonal=0)`
取矩阵的**上三角**部分（含对角线），其余元素置 0 。 diagonal > 0 向右偏移，< 0 向左偏移。

```python
x = torch.ones(4, 4)
torch.triu(x)           # 上三角矩阵（含主对角线）
torch.triu(x, diagonal=1)  # 严格上三角（不含主对角线），常用于 causal mask
```

### 4.2 `torch.tril(input, diagonal=0)`
取矩阵的**下三角**部分（含对角线），其余元素置 0 。

```python
x = torch.ones(4, 4)
torch.tril(x)              # 下三角矩阵（含主对角线）
torch.tril(x, diagonal=-1) # 严格下三角（不含主对角线）
```

### 4.3 `tensor.masked_fill(mask, value)`
将 mask 中为 `True` 的位置填充为指定 value，**原地操作用** `masked_fill_`。常与 triu/tril 配合实现注意力掩码。

```python
x = torch.randn(3, 3)
mask = torch.triu(torch.ones(3, 3), diagonal=1).bool()  # 严格上三角为 True
x.masked_fill(mask, float('-inf'))  # 上三角位置填充 -inf，用于 causal attention
```

### 4.4 比较运算符 `==, !=, >, <`
逐元素比较，返回 bool 型张量，常用于生成掩码。

```python
x = torch.tensor([1, 2, 3, 0, 0])
x == 0          # tensor([False, False, False,  True,  True])，padding mask
x > 1           # tensor([False,  True,  True, False, False])
x != 0          # tensor([ True,  True,  True, False, False])
```

---

## 5 运算 API

### 5.1 `tensor.sum(dim, keepdim)`
沿指定维度求和。不指定 dim 则对所有元素求和。

```python
x = torch.tensor([[1., 2., 3.], [4., 5., 6.]])  # shape (2, 3)
x.sum()             # tensor(21.)，所有元素之和
x.sum(dim=0)        # tensor([5., 7., 9.])，沿行求和，shape (3,)
x.sum(dim=1)        # tensor([6., 15.])，沿列求和，shape (2,)
x.sum(dim=1, keepdim=True)  # shape (2, 1)，保持维度
```

### 5.2 `tensor.mean(dim, keepdim)`
沿指定维度求均值，用法同 sum 。

```python
x = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
x.mean()             # tensor(3.5)
x.mean(dim=1)        # tensor([2., 5.])
x.mean(dim=1, keepdim=True)  # shape (2, 1)
```

### 5.3 `tensor.max(dim, keepdim)`
沿指定维度取最大值。不指定 dim 返回标量；指定 dim 返回 `(values, indices)` 的命名元组。

```python
x = torch.tensor([[1., 3., 2.], [5., 4., 6.]])
x.max()              # tensor(6.)，全局最大值
values, indices = x.max(dim=1)
# values: tensor([3., 6.])，每行最大值
# indices: tensor([1, 2])，最大值所在列索引
```

### 5.4 `torch.nn.functional.softmax(x, dim)`
对指定维度做 softmax，将数值转为概率分布（和为 1）。

```python
import torch.nn.functional as F

x = torch.tensor([1.0, 2.0, 3.0])
F.softmax(x, dim=0)   # tensor([0.0900, 0.2447, 0.6652])

# 在注意力中对最后一维做 softmax
scores = torch.randn(2, 4, 4)   # (batch, seq, seq)
attn = F.softmax(scores, dim=-1) # 每行和为 1
```

### 5.5 `tensor.argmax(dim)`
返回指定维度上最大值的**索引**，不返回值本身。

```python
x = torch.tensor([[1., 3., 2.], [5., 4., 6.]])
x.argmax()           # tensor(5)，全局最大值的扁平索引
x.argmax(dim=1)      # tensor([1, 2])，每行最大值所在列索引
# 常用于分类任务取预测类别：pred = logits.argmax(dim=-1)
```

---

## 6 torch API

### 6.1 `torch.clamp(input, min, max)`
将张量中所有数值**限制在 [min, max] 范围内**，超出范围的值会被截断到边界。

```python
x = torch.tensor([-1., 0., 0.5, 2., 3.])
torch.clamp(x, 0, 1)       # tensor([0., 0., 0.5, 1., 1.])
torch.clamp(x, min=0)      # tensor([0., 0., 0.5, 2., 3.])，只限下界
torch.clamp(x, max=1)      # tensor([-1., 0., 0.5, 1., 1.])，只限上界

# 实际场景：把网络输出限制在 >= 1
clipped_peds = torch.clamp(net(features), 1, float("inf"))
```

### 6.2 `torch.arange(start, end, step)`
创建等差数列，类似 Python 的 `range`，**不包含** end 。

```python
torch.arange(5)           # tensor([0, 1, 2, 3, 4])
torch.arange(1, 5)        # tensor([1, 2, 3, 4])
torch.arange(0, 1, 0.2)   # tensor([0.0, 0.2, 0.4, 0.6, 0.8])
```

### 6.3 `torch.cat(tensors, dim=0)`
沿已有维度拼接多个张量，**不会增加维度**。除拼接维度外，其余维度的大小必须相同。

```python
x = torch.tensor([[1, 2], [3, 4]])  # shape (2, 2)
y = torch.tensor([[5, 6]])          # shape (1, 2)
torch.cat([x, y], dim=0)
# tensor([[1, 2], [3, 4], [5, 6]])，shape (3, 2)

a = torch.tensor([[1], [2]])        # shape (2, 1)
b = torch.tensor([[3], [4]])        # shape (2, 1)
torch.cat([a, b], dim=1)
# tensor([[1, 3], [2, 4]])，shape (2, 2)

# Transformer 中可沿序列维拼接历史 token 与新 token
# all_tokens = torch.cat([past_tokens, new_tokens], dim=1)
```

### 6.4 `torch.stack(tensors, dim=0)`
先在指定位置新增一个维度，再沿该新维度堆叠张量。所有输入张量的 shape 必须**完全相同**。

```python
x = torch.tensor([1, 2, 3])         # shape (3,)
y = torch.tensor([4, 5, 6])         # shape (3,)

torch.stack([x, y], dim=0)
# tensor([[1, 2, 3], [4, 5, 6]])，shape (2, 3)

torch.stack([x, y], dim=1)
# tensor([[1, 4], [2, 5], [3, 6]])，shape (3, 2)

# 等价关系：stack 是 unsqueeze 后再 cat
torch.stack([x, y], dim=0) == torch.cat([x.unsqueeze(0), y.unsqueeze(0)], dim=0)
# 常用于把多个样本组成 batch：batch = torch.stack(samples, dim=0)
```

`torch.cat` 用于延长或拼接已有维度；`torch.stack` 用于把多个同形状样本组织成新的 batch / 序列维度。

---

## 7 tensor API

### 7.1 `tensor.repeat(*sizes)`
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

### 7.2 `tensor.expand(*sizes)`
将 size 为 1 的维度**广播扩展**到指定大小，不复制数据（零拷贝），比 repeat 更高效。

```python
x = torch.tensor([[1], [2], [3]])  # shape (3, 1)
x.expand(3, 4)   # shape (3, 4)，每行的 1 个元素扩展为 4 个
# tensor([[1,1,1,1],[2,2,2,2],[3,3,3,3]])

# -1 表示该维度保持不变
x.expand(-1, 4)  # 等价于 x.expand(3, 4)
```

### 7.3 `tensor.contiguous()`
返回在内存中**连续存储**的张量副本。 transpose/permute 后内存不连续，需要先调用此方法再使用 view 。

```python
x = torch.randn(2, 3)
y = x.transpose(0, 1)     # y 内存不连续
y.contiguous().view(-1)   # 先变连续，再 reshape
# 或直接用 reshape（内部会自动处理）
```
