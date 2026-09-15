# 06 Kernel 与 Triton

## GPU 细节

回顾上一章， GPU 内部有 Thread, Thread Block, Warps 几个概念。

Block 是一组 Thread，可能包含多个 Warps。在同一个周期内，Warps 必须在同一个 SM 上执行同一组命令，一个 SM 也可以同时面向多个 Warps，并且通过调度器执行切换。

### Bank Conflict

每个 SM 会被分成若干个 bank（例如 32 个 bank，每个 bank 有 4 字节），在同一个时钟周期内一个 bank 只能被一个 thread 访问，也就是说多个 thread 想访问同一个地址必须串行访问。

体现在实际计算中，例如一个矩阵多个 thread 都想访问第一行，则只能进行串行排队等待。

同样地，由于物理 SM 数量的限制，即使可以创建很多个 thread block，但是一个 wave 只能运行 SM 数量的 thread。

## Benchmark & Profile

在修改或者写 kernel 之前应该要做好 Benchmark 来首先判断好代码本身的运行情况。

在正式开始测试前应当做好 warm up 确保代码已经完成编译。

测试 GPU 时间可以使用 `torch.cuda.Event()` 来标记开始和结束的时间，另外注意使用 `torch.cuda.synchronize() ` 等待 GPU 的同步，这是因为 GPU 是异步执行的。

Profiling 的作用是分析不同代码运行所花费的时间。

对同一个 GeLU 的实现中，普通的自行实现需要调用多个 Kernel，会涉及到多个 HBM 的读写，官方实现（内置 Kernel）以及 `torch compile` 的实现（Triton Kernel）都快得多，因为只需要一次 HBM 的读写。

## Triton Kernel

在使用 CUDA 编程时，是逐个线程进行编程的，可以同时编程一个 Grid 的 Thread Block 做元素级别的操作，非常接近于硬件底层。Triton 是由 OpenAI 开发的一种类似于新的规范，在 Triton 中思考的是 Block 级别的操作。

### GeLU 运算

以一个 GeLU 为例，以下是一个简单的 triton block 初始化

```python
def triton_gelu(x: torch.Tensor):
    # Check input
    assert x.is_cuda
    assert x.is_contiguous()

    # Allocate output tensor
    y = torch.empty_like(x)

    # Determine grid (elements divided into blocks)
    # | T T T T T T T T | T T T T T T T T | T T T T T T T T | T T T T T T T T |
    # |    Block 0      |    Block 1      |     Block 2      |    Block 3     |
    num_elements = x.numel()  # @inspect num_elements
    BLOCK_SIZE = 1024  # Number of threads
    num_blocks = triton.cdiv(num_elements, BLOCK_SIZE)  # @inspect num_blocks

    # Launch the kernel
    kernel = triton_gelu_kernel[(num_blocks,)](x, y, num_elements, BLOCK_SIZE=BLOCK_SIZE)

    # Write out PTX (look at this later)
    output_ptx("triton_gelu", kernel)  # @stepover

    return y
```

下方是 kernel 的实际实现，每一个 block 对应一个 pid，所有的 thread 执行的是同一段代码。

```python
@triton.jit
def triton_gelu_kernel(x_ptr, y_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    # Input starts at `x_ptr`
    # Output starts at `y_ptr`

    # | T T T T T T T T | T T T T T T T T | T T T T T T T T | T T T T T T T T |
    # |    Block 0      |    Block 1      |     Block 2      |    Block 3     |

    pid = tl.program_id(axis=0)      # Identifies the block
    start = pid * BLOCK_SIZE         # Starting index of this block

    # Indices where this thread block should operate
    offsets = start + tl.arange(0, BLOCK_SIZE)

    # Don't read/write past the end of the tensor
    mask = offsets < num_elements

    # Read
    x = tl.load(x_ptr + offsets, mask=mask)

    # Approx gelu is 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Compute (tl.tanh doesn't exist, use tanh(a) = (exp(2a) - 1) / (exp(2a) + 1)
    a = 0.79788456 * (x + 0.044715 * x * x * x)
    exp = tl.exp(2 * a)
    tanh = (exp - 1) / (exp + 1)
    y = 0.5 * x * (1 + tanh)

    # Store
    tl.store(y_ptr + offsets, y, mask=mask)
```

在经过编译后生成的是 PTX 代码，再向下还有额外的细节是无法干预的。

### Softmax

在 softmax 中会有更加复杂的运算，包括归约，取最大值等，如果用朴素的实现，需要用到非常多的 Kernel，导致很多额外的读写开销。

在 Triton 中一般把一个行（row）作为一个 block，因为 softmax 的操作只与行内元素有关。这里只展示 Kernel 部分：

```python
@triton.jit
def triton_softmax_kernel(x_ptr, y_ptr, x_row_stride, y_row_stride, num_cols, BLOCK_SIZE: tl.constexpr):
    assert num_cols <= BLOCK_SIZE

    # Process each row independently
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # Read from global memory
    x_start_ptr = x_ptr + row_idx * x_row_stride
    x_ptrs = x_start_ptr + col_offsets
    x_row = tl.load(x_ptrs, mask=col_offsets < num_cols, other=float("-inf"))

    # Compute
    x_row = x_row - tl.max(x_row, axis=0)
    numerator = tl.exp(x_row)
    denominator = tl.sum(numerator, axis=0)
    y_row = numerator / denominator

    # Write back to global memory
    y_start_ptr = y_ptr + row_idx * y_row_stride
    y_ptrs = y_start_ptr + col_offsets
    tl.store(y_ptrs, y_row, mask=col_offsets < num_cols)
```

如果一次操作的所有的数据都可以放进一个 block，那么这个写法就类似于 pytorch 了。难点在于如果是一个非常大的矩阵，一个 block 无法完成。

![](http://oss.rainerseventeen.cn/blog/2026/202609091455983.png)

此时每一个 block 仍然是负责同一个行，每一个 thread 共同维护各自的累加器，最后再一个 loop 全部加起来，具体的代码省略。
