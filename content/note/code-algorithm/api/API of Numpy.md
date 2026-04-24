# Numpy 笔记

> 写一点深度学习中常用且需要掌握的一些基础用法

## **切片（Slicing）**


  表示在某些维度上取指定区间的数据，其余维度保持原有结构不变。


  NumPy 的切片规则是 **左闭右开** [start, end)。

  ```
# X.shape = (N, C, H, W)
patch = X[:, :, i:i+h, j:j+w]
  ```

  这里：


  - : 表示该维度 **不做切片**
  - i:i+h 表示在空间维度上取一个窗口
  - 返回的仍然是一个四维张量


  切片本质上是：


  - 不发生数据拷贝（view）
  - 仅改变索引映射


  在卷积、池化、局部统计中是最基础的操作。


------


  ## **广播机制（Broadcasting）**


  在进行**逐元素运算**时，NumPy 会自动扩展形状较小的数组，使其与目标数组在维度上对齐。


  广播成立的条件是：


  - 从 **最后一个维度开始对齐**
  - 对应维度要么相等，要么其中一个为 1


  ```
# X.shape = (N, C, H, W)
# b.shape = (C,)
Y = X + b
  ```

  这里：


  - b 会被自动扩展为 (1, C, 1, 1)
  - 对每个 batch、每个空间位置都加上对应通道的偏置


  广播机制是：


  - bias
  - normalization 中 scale / shift
  - loss 中标量参与运算


  的数学基础。


------


  ## **元素级运算（Hadamard 运算）**


  对两个形状可广播的数组进行**逐元素运算**。

  ```
A * B
A + B
A / B
A ** 2
  ```

  在深度学习中，以下操作本质都是元素级运算：


  - ReLU：np.maximum(X, 0)
  - Sigmoid：1 / (1 + np.exp(-X))
  - MSE：(y_pred - y_true) ** 2


  元素级运算不涉及维度缩减，**输出 shape 与广播后的 shape 一致**。


------


  ## **矩阵乘法（Matrix Multiplication）**


  矩阵乘法表示 **线性组合与特征变换**，是线性层和注意力机制的核心。

  ```
C = A @ B
# 等价于
C = np.matmul(A, B)
  ```

  矩阵乘法的规则是：


  - (m, k) @ (k, n) -> (m, n)
  - 内层维度必须相等


  需要明确区分：

  ```
A * B   # 元素乘
A @ B   # 矩阵乘法
  ```

  在深度学习中：


  - 全连接层
  - self-attention 中的 QKᵀ
  - im2col 后的卷积实现


  都依赖矩阵乘法。


------


  ## **维度变换（reshape / transpose）**


  ### **reshape**


  改变数组的形状，但不改变数据顺序。

  ```
X = X.reshape(N, -1)
  ```

  常用于：


  - CNN 输出 → 全连接输入
  - batch 与特征维度重排


  -1 表示由 NumPy 自动推断该维度大小。


------


  ### **transpose / swapaxes**


  改变维度的排列顺序。

  ```
X = X.transpose(0, 3, 1, 2)  # NHWC -> NCHW
  ```

  在不同框架中：


  - NumPy / TensorFlow 常见 NHWC
  - PyTorch 使用 NCHW


  转置不会改变数据内容，但会改变访问顺序，对性能有影响。


------


  ## **归约操作（Reduction）**


  将某个维度“压缩”为一个统计量。

  ```
np.sum(X, axis=0)
np.mean(X, axis=1)
np.max(X, axis=-1)
np.argmax(X, axis=1)
  ```

  在深度学习中：


  - loss 的计算一定包含归约
  - pooling 是空间维度上的归约
  - softmax 需要沿类别维度归约


  axis 表示 **被消除的维度**。


------


  ## **条件索引（Mask）**


  通过布尔条件选取或修改数组中的部分元素。

  ```
X[X < 0] = 0
  ```

  这在数学上等价于 ReLU。


  条件索引的特点：


  - 会返回一维结果（如果直接取）
  - 或用于就地修改


  在以下场景中常用：


  - mask loss
  - padding 忽略
  - attention mask


------


  ## **统计与归一化**


  计算均值、方差并进行标准化。

  ```
mean = np.mean(X, axis=0)
std  = np.std(X, axis=0)
X = (X - mean) / (std + 1e-8)
  ```

  这是：


  - 数据预处理
  - BatchNorm / LayerNorm 的数学原型


  加上 1e-8 是为了数值稳定性，防止除零。


------


  ## **随机数与初始化**


  ```
np.random.randn(C, H, W)
np.random.uniform(-a, a, size)
np.random.randint(0, n, size)
  ```

  用于：


  - 权重初始化
  - 随机采样
  - 教学级 dropout / data augmentation


------


  ## **数值检查（调试用）**


  ```
np.isnan(X)
np.isinf(X)
  ```

  在训练中用于排查：


  - loss 为 NaN
  - 梯度爆炸
  - 数值不稳定问题


------


  ## **与你当前 CNN 代码的对应关系**


  ```
(X[i:i+h, j:j+w] * K).sum()
  ```

  这一行同时使用了：


  - 切片（局部窗口）
  - 元素级乘法
  - 广播（如果有 bias）
  - 归约（sum）


  这正是卷积在 NumPy 层面的完整数学表达。


------


  如果你下一步希望：


  - 把这些操作 **整理成一份 NumPy → CNN 对照表**
  - 或按 **“卷积 / 池化 / BN / FC”分别拆解 NumPy 实现**


  可以直接说需求，我按同一写作风格继续补齐。