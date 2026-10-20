---
title: Triton
description: 从 Triton 的基本概念出发，逐句理解 CS336 加权求和示例的前向、反向和 PyTorch 封装。
---

# Triton

> 代码来自于 cs336 Assignment 2 中的 demo 演示，本笔记仅包含部分核心算法。

## 1 基本概念

### 1.1 Python 封装与 GPU kernel

使用 PyTorch 时，`x`、`weight` 是已经分配好存储空间的张量。普通 Python 代码可以检查它们的形状、分配输出，再发起 GPU 计算。

**kernel 是在 GPU 上执行的计算函数。** 本例有两个 kernel：`weighted_sum_fwd` 计算输出，`weighted_sum_backward` 计算梯度。`@triton.jit` 是 Triton 编译的修饰器；函数内部使用 `triton.language`，描述 GPU 上的加载、计算和写回。

`WeightedSumFunc` 是连接 PyTorch 自动微分的 Python 类。外部调用 `WeightedSumFunc.apply(x, weight)` 时，它的 `forward` 分配输出并启动前向 kernel；求导时，PyTorch 把上游梯度交给它的 `backward`，再由后者启动反向 kernel 。

| 组成部分 | 在本例中做什么 |
| --- | --- |
| PyTorch 张量 | 保存显存中的输入、输出、梯度和中间结果 |
| `WeightedSumFunc.forward` | 整理形状、保存反向所需输入、分配输出、启动前向 kernel |
| `weighted_sum_fwd` | 读取输入和权重，计算每行加权和并写回 |
| `WeightedSumFunc.backward` | 整理上游梯度、分配梯度、启动反向 kernel 、合并局部权重梯度 |
| `weighted_sum_backward` | 写出输入梯度，以及每组行对权重梯度的贡献 |

Python 发起 GPU 操作通常是异步的，函数调用返回不代表 GPU 已完成计算。同一 CUDA stream 中的操作按顺序执行，因此后续求和可以读取先前 kernel 写出的中间结果。

### 1.2 program 、 tile 与启动网格

一次 kernel 启动会运行多个 **program instance**，执行同一段代码，通过各自的编号决定处理哪块数据。这里可以把一个 program 近似理解为一个 CUDA thread block：内部有多个 GPU 线程协作执行张量运算。

**tile 是一次处理的数据块。** 假设完整输入是 `[N, D]`，本例让一个 program 负责若干行，再沿列分批读取。一次读取的 tile 形状是 `[ROWS_TILE_SIZE, D_TILE_SIZE]`。

**grid 是启动网格**，决定运行多少个 program 。`tl.program_id(0)` 给出当前 program 在一维网格中的编号。

代码中对一个 tile 的乘法、广播和求和是张量运算，由 program 内的线程协作完成。源码没有逐个指定某个线程计算哪个元素。

### 1.3 block pointer

向 kernel 传入 PyTorch 张量时，kernel 接收到的是其数据的起始指针。指针本身不包含张量大小等信息，因此还需要形状和步长。

| 对象或操作 | 含义 |
| --- | --- |
| `x_ptr` | 输入数据的起始地址 |
| `stride`（步长） | 某一维索引增加 1 时，地址移动多少个元素 |
| block pointer（分块指针） | 描述完整张量中的一块区域，包括形状、步长、起点和块大小 |
| `tl.load(...)` | 按指针从显存读取，得到 kernel 内可计算的局部张量 |
| `tl.store(...)` | 把局部计算结果写回显存 |
| `.advance(...)` | 返回起点移动后的分块指针，供下一次读取或写入 |

例如，`x_block_ptr` 描述“要读哪块数据”，`row = tl.load(x_block_ptr, ...)` 才得到这块数据的值。`row` 虽然名字叫 row，但在本例中是一个二维块。

## 2 算法

PyTorch 参考实现只有一行：

```python
def weighted_sum(x, weight):
    return (weight * x).sum(axis=-1)
```

输入 `x` 的形状为 `[..., D]`，`weight` 的形状为 `[D]`，输出保留 `x` 除最后一维以外的形状。例如 `[B, T, D]` 输入得到 `[B, T]` 输出，所有位置共用同一组权重。

先把前面的维度合并为 $N$，就得到二维问题：

$$
X\in\mathbb{R}^{N\times D},\qquad
w\in\mathbb{R}^{D},\qquad
y_i=\sum_{j=0}^{D-1}X_{i,j}w_j.
$$

例如某一行是 `[1, 2, 3]`，权重是 `[2, 1, 3]`，这一行的输出就是 $1\times2+2\times1+3\times3=13$。对二维输入，它在数学上也等于 `X @ w`。

普通 PyTorch eager 执行通常先产生 `[N, D]` 的乘法结果，再调用归约算子。 Triton 可以把加载、乘法、求和和写回融合到一个 GPU kernel 中，避免将完整的乘法中间结果写入显存。`torch.compile` 也可能进行类似融合。

除了得到输出，训练时还要返回两个梯度。设上游梯度为 $g_i=\partial L/\partial y_i$，则：

$$
\frac{\partial L}{\partial X_{i,j}}=g_iw_j,
\qquad
\frac{\partial L}{\partial w_j}=\sum_{i=0}^{N-1}g_iX_{i,j}.
$$

前向把每行的 $D$ 个加权值加起来；输入梯度把每行的上游梯度乘到每个权重上；权重梯度则把所有行对同一个权重的贡献加起来。

| 要得到的结果 | 展平后的形状 | 计算方向 |
| --- | --- | --- |
| `y` | `[N]` | 对每行沿列求和 |
| `grad_x` | `[N, D]` | 逐元素计算 $g_iw_j$ |
| `grad_weight` | `[D]` | 对每列沿行求和 |

## 3 3. program 分工

源码使用下面两个分块参数：

```python
ROWS_TILE_SIZE = 16
D_TILE_SIZE = max(1, triton.next_power_of_2(max(D, 1)) // 16)
```

记这两个参数为 $B_R$ 和 $B_D$。一个 program 负责 $B_R$ 行，在内部沿列循环，每次处理一个 `[B_R, B_D]` 数据块。启动的 program 数量为：

$$
P=\left\lceil\frac{N}{B_R}\right\rceil.
$$

为便于手算，先用一组教学参数 $N=12,D=12,B_R=4,B_D=4$，它们不是源码默认参数：

| program | 负责的行 | 内部依次处理的列 |
| --- | --- | --- |
| 0 | 0–3 | 0–3 、 4–7 、 8–11 |
| 1 | 4–7 | 0–3 、 4–7 、 8–11 |
| 2 | 8–11 | 0–3 、 4–7 、 8–11 |

不同 program 可以并行处理不同的行。同一个 program 的源码循环逐块遍历列，而每块中的乘法和归约仍是并行运算。

### 3.1 启动网格与 program 编号

```python
weighted_sum_fwd[(triton.cdiv(n_rows, ROWS_TILE_SIZE),)](...)
```

方括号设置启动网格，圆括号传入参数。这里使用一维网格，启动 $P$ 个 program 。 kernel 内通过下面的语句确定自己负责哪一组行：

```python
row_tile_idx = tl.program_id(0)
```

`tl.program_id(0)` 是当前 program 在网格第 0 维的编号；`tl.num_programs(0)` 则是这一维的 program 总数。

### 3.2 `tl.constexpr`

函数签名中的两个参数为：

```python
ROWS_TILE_SIZE: tl.constexpr,
D_TILE_SIZE: tl.constexpr,
```

在编译期确定，使编译器知道局部张量的形状。

`NUM_ROWS` 和 `D` 描述完整输入的逻辑尺寸，分块大小描述一次处理多少数据。

## 4 4. Forward

### 4.1 kernel 流程

设当前 program 编号为 $p$。它负责从 $pB_R$ 开始的 $B_R$ 行，目标是给每个有效行算出一个完整的加权和。不同 program 的输出行互不重叠。

因为一次只读取 $B_D$ 列，所以每行的结果要分多次计算：读取一块输入和对应权重，得到这块列的部分和，累加后继续读取下一块。所有列处理完成后，才把完整结果写回。

### 4.2 program & block pointer 设置

kernel 内部指定不同的 program 操作不同的数据区域，并设置好对应的指针。

```python
row_tile_idx = tl.program_id(0)
```

取得当前 program 的编号。比如编号为 2 、`ROWS_TILE_SIZE = 16`，它负责的行起点就是 `2 * 16 = 32`。

输入指针的设置如下：

```python
x_block_ptr = tl.make_block_ptr(
    x_ptr,
    shape=(NUM_ROWS, D),
    strides=(x_stride_row, x_stride_dim),
    offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
    block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
    order=(1, 0),
)
```

`x_ptr` 是输入的起始指针。`shape=(NUM_ROWS, D)` 描述完整二维输入，也用于后续判断哪些索引越界。

`strides=(x_stride_row, x_stride_dim)` 描述行、列方向的步长。对连续的 `[100, 256]` 输入，这两个值为 `(256, 1)`，元素 $(i,j)$ 的地址是（单位是元素不是字节）：
$$
X_{\mathrm{ptr}}+i\,s_r+j\,s_d.
$$

`offsets` 给出当前块的左上角：行起点取决于 program 编号，列起点为 0 。`block_shape` 给出一次读取的局部形状 `[B_R, B_D]`。

`order=(1, 0)` 描述第 1 维（row）优先的访问布局。

权重和输出指针同理如下：

```python
weight_block_ptr = tl.make_block_ptr(
    weight_ptr,
    shape=(D,),
    strides=(weight_stride_dim,),
    offsets=(0,),
    block_shape=(D_TILE_SIZE,),
    order=(0,),
)
output_block_ptr = tl.make_block_ptr(
    output_ptr,
    shape=(NUM_ROWS,),
    strides=(output_stride_row,),
    offsets=(row_tile_idx * ROWS_TILE_SIZE,),
    block_shape=(ROWS_TILE_SIZE,),
    order=(0,),
)
```

权重只有一维，每个 program 都从第 0 列开始，逐块读取相同的权重向量；权重块有 `D_TILE_SIZE` 个元素。

输出也只有一维。`output_block_ptr` 从当前 program 的起始行开始，覆盖 `ROWS_TILE_SIZE` 个结果。输出指针之后不沿列移动，因为一行无论分多少个列块计算，最终只对应一个输出位置。

### 4.3 Forward 计算详解

```python
output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)

for i in range(tl.cdiv(D, D_TILE_SIZE)):
    row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
    weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")

    output += tl.sum(row.to(tl.float32) * weight[None, :].to(tl.float32), axis=1)

    x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
    weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))

tl.store(output_block_ptr, output, boundary_check=(0,))
```

```python
output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)
```

创建长度为 $B_R$ 的局部张量，每个位置对应当前 program 负责的一行。初值全为零，之后把各个列块的部分和加到这里， 是 kernel 内的局部计算结果

```python
for i in range(tl.cdiv(D, D_TILE_SIZE)):
```

`tl.cdiv(D, D_TILE_SIZE)` 是向上取整的除法，即 $\lceil D/B_D\rceil$。例如 $D=10,B_D=4$ 时，需要处理 3 块，分别覆盖列 0–3 、 4–7 、 8–11；最后一块只有列 8 、 9 有效。

```python
row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")
```

`row` 按照 `x_block_ptr` 描述的位置读取数据，得到形状为 `[B_R, B_D]` 的局部张量。`boundary_check=(0, 1)` 中的 0 、 1 是输入的维度编号，表示同时检查行和列是否越界。

最后一组行可能不足 $B_R$ 行，最后一个列块也可能不足 $B_D$ 列。`padding_option="zero"` 让这些越界位置读作零；它不会把零写入原始输入。

得到长度为 $B_D$ 的权重块。权重是一维张量，因此只检查第 0 维。输入若正在读取列 4–7，权重也必须读取索引 4–7，才能计算对应元素的乘积。

```python
output += tl.sum(row.to(tl.float32) * weight[None, :].to(tl.float32), axis=1)
```

| 子表达式 | 形状 | 做什么 |
| --- | --- | --- |
| `row.to(tl.float32)` | `[B_R, B_D]` | 将输入块转为 FP32 |
| `weight[None, :].to(tl.float32)` | `[1, B_D]` | 给权重增加一个行维度，并转为 FP32 |
| 两者相乘 | `[B_R, B_D]` | 每一行乘以同一组权重 |
| `tl.sum(..., axis=1)` | `[B_R]` | 对列求和，每行得到当前列块的部分和 |
| `output += ...` | `[B_R]` | 将部分和加到该行此前的累加值上 |

```python
x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
```

输入指针的行偏移增加 0 、列偏移增加 $B_D$，因此仍然处理同一组行。权重是一维的，只需要把索引增加 $B_D$。`.advance()` 返回移动后的指针，赋值后供下一轮使用

```python
tl.store(output_block_ptr, output, boundary_check=(0,))
```

 `output` 已经包含所有有效列的贡献，写回的形状为 `[B_R]`。边界检查只针对输出的第 0 维，防止最后一个 program 写入不存在的行。

写回时，FP32 的局部结果会转为 `x` 使用相同类型。

## 5 5. Backward

### 5.1 kernel 流程

反向仍按行分配 program，沿列循环。对于当前块，它需要同时计算：

$$
(\mathrm{grad\_x})_{i,j}=g_iw_j,
\qquad
G_{p,j}=\sum_{i=pB_R}^{\min((p+1)B_R,N)-1}g_iX_{i,j}.
$$

`grad_x` 可以直接写回，因为不同 program 负责不同的输入行。权重则被所有行共用，每个 program 只能算出一部分权重梯度。

因此 Python 端准备形状为 `[P, D]` 的 `partial_grad_weight`，其中 $P=\lceil N/B_R\rceil$。 program $p$ 只写第 $p$ 行。 kernel 结束后，再把这 $P$ 行加起来，得到最终的 `[D]` 权重梯度。

例如 $N=5,B_R=4$，program 0 计算输入行 0–3 的贡献，program 1 计算输入行 4 的贡献。它们分别写中间缓冲区的第 0 、 1 行，后续求和时合并。这避免了多个 program 同时读改写同一位置造成的并发累加问题。

### 5.2 block pointor

具体设定同 Forward，各自功能如下：

| 指针 | 完整形状 | 当前块形状 | 起点与用途 |
| --- | --- | --- | --- |
| `grad_output_block_ptr` | `[N]` | `[B_R]` | 从 $pB_R$ 开始，读取当前这组行的上游梯度 |
| `x_block_ptr` | `[N, D]` | `[B_R, B_D]` | 从输入的 $(pB_R,0)$ 开始，计算权重梯度 |
| `weight_block_ptr` | `[D]` | `[B_D]` | 从权重索引 0 开始，计算输入梯度 |
| `grad_x_block_ptr` | `[N, D]` | `[B_R, B_D]` | 从 $(pB_R,0)$ 开始，写当前输入块的梯度 |
| `partial_grad_weight_block_ptr` | `[P, D]` | `[1, B_D]` | 从 $(p,0)$ 开始，写本 program 的权重梯度贡献 |

注意最后一个指针的行起点是 `row_tile_idx`，不是 `row_tile_idx * ROWS_TILE_SIZE`。输入中一组 $B_R$ 行的贡献已经求和，写到中间缓冲区时只占一行。

### 5.3 Backward 计算

详解略，这里只放出代码

```python
for i in range(tl.cdiv(D, D_TILE_SIZE)):
    grad_output = tl.load(grad_output_block_ptr, boundary_check=(0,), padding_option="zero")
    weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")
    grad_x_row = grad_output[:, None] * weight[None, :]
    tl.store(grad_x_block_ptr, grad_x_row, boundary_check=(0, 1))

    row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
    grad_weight_row = tl.sum(row.to(tl.float32) * grad_output[:, None].to(tl.float32), axis=0, keep_dims=True)
    tl.store(partial_grad_weight_block_ptr, grad_weight_row, boundary_check=(1,))

    x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
    weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
    partial_grad_weight_block_ptr = partial_grad_weight_block_ptr.advance((0, D_TILE_SIZE))
    grad_x_block_ptr = grad_x_block_ptr.advance((0, D_TILE_SIZE))
```
