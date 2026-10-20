---
paper:
 - "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"

 - "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
---

# FlashAttention

> 论文：[FlashAttention](https://arxiv.org/abs/2205.14135)（2022）与 [FlashAttention-2](https://arxiv.org/abs/2307.08691)（2023）
>
> 本文内容涉及 AI 辅助生成

FlashAttention 将注意力分块计算，利用在线 softmax 和算子融合，避免把完整的注意力矩阵写入显存。 FlashAttention-2 在此基础上减少非矩阵乘法操作，并调整线程块和 warp 的任务分配。

## 1 原始 Attention

考虑单个样本、单个 head 的 self attention，query 与 key/value 的序列长度均为 N，向量维度均为 d，定义以下符号：

| 符号 | 定义与形状 |
|---|---|
| $N,d$ | 序列长度、每个注意力头的向量维度，均为正整数 |
| $Q,K,V\in\mathbb{R}^{N\times d}$ | 查询、键、值矩阵；$\mathbb{R}$ 表示实数域 |
| $q_i,k_j,v_j\in\mathbb{R}^d$ | 对应矩阵第 i 或 j 行所表示的向量，写成列向量 |
| $S\in\mathbb{R}^{N\times N}$ | 缩放并加掩码后的分数矩阵 |
| $P\in\mathbb{R}^{N\times N}$ | 每行归一化后的注意力概率矩阵 |
| $O\in\mathbb{R}^{N\times d}$ | 输出矩阵，第 i 行对应输出向量 $o_i$ |
| $M\in\mathbb{R}^{N\times N}$（允许含 $-\infty$） | 加性掩码，允许访问处为 0，被屏蔽处为 $-\infty$ |

$\top$ 表示转置，$\operatorname{softmax}_{\mathrm{row}}$ 表示对矩阵的每一行独立计算 softmax，缩放因子 $1/\sqrt d$ 用于控制 scale：

$$
S=\frac{QK^\top}{\sqrt d}+M,\qquad
P=\operatorname{softmax}_{\mathrm{row}}(S),\qquad O=PV.
$$


传统的分离算子实现通常需要：

1. 计算 $S$，将其写入 HBM 。

2. 从 HBM 读出 $S$，计算 softmax，再写入 $P$。

3. 从 HBM 读出 $P$，计算 $PV$。

例如 $N=8192$ 时，一个 FP16 的 $N\times N$ 矩阵就占 128 MiB，还未计算 batch 、多头和反向传播的倍率。尽管矩阵乘法可以使用 Tensor Core，但 softmax 和中间矩阵的显存读写仍会限制速度。

相对标准 Attention，分块计算把“生成完整 P，再计算 PV”改成“每块分数算完后立即累加加权分子”。在线 softmax 解决跨块归一化问题，分块融合避免 S/P 写入显存，反向重计算避免为了求梯度而保存 P 。下文统一采用 FlashAttention-2 的计算方式。

## 2 Online Softmax

### 2.1 稳定 softmax

固定一个 query，令 $x_j$ 表示它与第 j 个 key 的分数，$p_j$ 表示相应概率。对一行分数 $x_1,\ldots,x_N$，直接计算 $e^{x_j}$ 可能溢出。利用 softmax 的平移不变性：

$$
m=\max_j x_j,\qquad
\ell=\sum_j e^{x_j-m},\qquad
p_j=\frac{e^{x_j-m}}{\ell}.
$$

这里 m 是整行最大值，$\ell$ 是减去最大值后的指数和，不是原始的 $\sum_j e^{x_j}$。两者之所以给出相同概率，是因为分子和分母同时乘以 $e^{-m}$：

$$
\frac{e^{x_j}}{\sum_t e^{x_t}}
=\frac{e^{-m}e^{x_j}}{e^{-m}\sum_t e^{x_t}}
=\frac{e^{x_j-m}}{\ell}.
$$

此处 t 是遍历该行所有 key 的求和下标。如果按独立遍历实现，通常先求最大值，再求指数和，最后生成归一化结果。实际一个 kernel 能否融合这些遍历，取决于一行数据能否放入一个 tiling

### 2.2 在线更新

把需要完整看完所有数值取最大值的操作，改成随最大值实时更新的算法。

对已处理前 t 个元素后维护最大值 $m_t$ 和指数和 $\ell_t$：

$$
m_t=\max_{j\le t}x_j,\qquad
\ell_t=\sum_{j\le t}e^{x_j-m_t}.
$$

加入 $x_t$ 时，从上一状态更新；$\alpha_t$ 表示把旧指数和换算到新最大值基准的缩放因子：

$$
\begin{aligned}
m_t&=\max(m_{t-1},x_t),\\
\alpha_t&=e^{m_{t-1}-m_t},\\
\ell_t&=\alpha_t\ell_{t-1}+e^{x_t-m_t}.
\end{aligned}
$$

初始化 $m_0=-\infty,\ell_0=0$；首个有限输入对应 $\alpha_1=0$。修正因子的来源是：

$$
\sum_{j<t}e^{x_j-m_t}
=e^{m_{t-1}-m_t}\sum_{j<t}e^{x_j-m_{t-1}}
=\alpha_t\ell_{t-1}.
$$

因此不必重新读取旧元素，就能把旧分母换算到新的最大值基准。当最大值不变时，$\alpha_t=1$；最大值增大时，$0<\alpha_t<1$。

如果需要的是整个 $P$ 矩阵，那么这个 online 没什么意义，在遍历完所有元素后要回头把前面的全部更新，但是在 Attention 中还有额外的优化，不需要拿到完整的 $P$ 推进计算。

### 2.3 分块合并

设旧状态为 $(m,\ell)$，$B$ 是新块包含的 key 下标集合。撇号表示更新后的量，不表示导数；$w_j$ 是以新最大值为基准的未归一化指数权重：

$$
\begin{aligned}
m'&=\max\left(m,\max_{j\in B}x_j\right),\\
\alpha&=e^{m-m'},\qquad w_j=e^{x_j-m'},\\
\ell'&=\alpha\ell+\sum_{j\in B}w_j.
\end{aligned}
$$

也可以先独立计算两个互不重叠、非空的 key 下标集合 A 、 B 的统计量。$m_A,\ell_A$ 分别是 A 的局部最大值和局部指数和，$m_B,\ell_B$ 同理，再合并：

$$
m=\max(m_A,m_B),\qquad
\ell=e^{m_A-m}\ell_A+e^{m_B-m}\ell_B.
$$

理论上分块合并的顺序不影响结果，这也是沿 K/V 维度并行后能合并部分结果的依据。

## 3 分块累加 Attention 的分子与分母

### 3.1 分块矩阵无法直接相加

注意到 softmax 的分母需要累加，如果是两个块直接求和得到的是 $o_A+o_B=\frac{N_A}{Z_A}+\frac{N_B}{Z_B}$ 而不是 $o_{A\cup B}=\frac{N_A+N_B}{Z_A+Z_B}$
$$
o_{A\cup B}=\frac{\sum_{j\in A}e^{x_j}v_j+\sum_{j\in B}e^{x_j}v_j}
{\sum_{j\in A}e^{x_j}+\sum_{j\in B}e^{x_j}}.
$$

这里 A 、 B 是互不重叠的 key 下标集合，$x_j=q^\top k_j/\sqrt d$ 为固定 query 向量 q 与第 j 个 key 的分数，$v_j$ 为对应的值向量。$N_A=\sum_{j\in A}e^{x_j}v_j$ 是加权分子，$Z_A=\sum_{j\in A}e^{x_j}$ 是分母，B 同理；$o_A=N_A/Z_A$ 表示局部归一化输出。$N_A$ 与序列长度 N 无关。

因此直接保存并合并分子与分母，在处理完全部 key 后统一归一化。为保证数值稳定，两者都需要减去最大值后再累加。

### 3.2 分块保存的数据

旧状态为 $(m_A,\ell_A,u_A)$。$m_A,\ell_A$ 为标量，$u_A\in\mathbb{R}^d$ 是以旧最大值为基准的加权分子：

$$
m_A=\max_{j\in A}x_j,\qquad
\ell_A=\sum_{j\in A}e^{x_j-m_A},\qquad
u_A=\sum_{j\in A}e^{x_j-m_A}v_j.
$$

读取新块 B 后，先计算覆盖旧、新两块的最大值 $m'$，再让新块直接使用这个最大值计算权重：

$$
m'=\max\left(m_A,\max_{j\in B}x_j\right),\qquad
\alpha=e^{m_A-m'},\qquad w_j=e^{x_j-m'}\quad(j\in B).
$$

$\alpha$ 是旧状态的缩放因子，$w_j$ 是新块的未归一化指数权重。新块已经使用累计最大值 $m'$，无需额外修正。

### 3.3 完整合并算法

**第一步，合并分母。** 两块都换算到以 $m'$ 为基准的指数和：
$$
\ell'=\alpha\ell_A+\sum_{j\in B}w_j
=\sum_{j\in A\cup B}e^{x_j-m'}.
$$

**第二步，合并加权分子。** 旧分子整体乘以 $\alpha$，新块直接使用已经按 $m'$ 计算的权重：

再对两部分做新的最大值修正，得到合并后的加权分子 $u'\in\mathbb{R}^d$：

$$
\begin{aligned}
u'&=\alpha u_A+\sum_{j\in B}w_jv_j\\
&=\sum_{j\in A}e^{x_j-m'}v_j+\sum_{j\in B}e^{x_j-m'}v_j.
\end{aligned}
$$

因此每一步只更新 $(m',\ell',u')$。将新块并入 A 后继续处理下一块，不需要恢复旧分子，也不需要在循环内计算归一化输出。

**第三步，全部 key 处理完后归一化。** 最终输出为：

$$
\boxed{o=\frac{u}{\ell}
=\frac{\sum_j e^{x_j-m}v_j}{\sum_j e^{x_j-m}}.}
$$

为了说明合并的含义，可在数学上把任一步的归一化结果记作 $o'=u'/\ell'$。若新块的独立输出为 $o_B=\sum_{j\in B}w_jv_j/\sum_{j\in B}w_j$，则有：

$$
o'=\frac{\alpha\ell_A}{\ell'}o_A+
\frac{\sum_{j\in B}w_j}{\ell'}o_B.
$$

这两个系数之和为 1，分别等于两块在全局 softmax 中占据的概率质量。每处理完一个块，$(m',\ell',o')$ 就精确表示所有已处理 key 的 Attention；以此递推，最后一个块处理完就是整行结果。

这个等式用于解释结果，计算时只保留未归一化分子 u，不在每步生成 $o'$。初始化取 $m=-\infty,\ell=0,u=0$，旧状态的贡献为零；掩码导致的空块需要单独处理。

## 4 分块融合计算

### 4.1 从一行扩展到矩阵块

设 $B_r$ 为每个 Q 块的行数，$B_c$ 为每个 K/V 块的行数；尾块可以更短。这里 a 、 b 是块编号，与单个位置的下标 i 、 j 区分。把 Q 划分为 $B_r\times d$ 的行块 $Q_a$，K/V 划分为 $B_c\times d$ 的行块 $K_b,V_b$：

$$
S_{ab}=Q_aK_b^\top/\sqrt d+M_{ab}\in\mathbb{R}^{B_r\times B_c}.
$$

$M_{ab}$ 是对应的掩码块。旧状态 $m_a,\ell_a\in\mathbb{R}^{B_r}$ 保存每行统计量，$U_a\in\mathbb{R}^{B_r\times d}$ 保存未归一化分子。当前块使用新的累计最大值计算指数权重 $W_{ab}\in\mathbb{R}^{B_r\times B_c}$。

下式的 $m',\ell',\alpha$ 均为长度 $B_r$ 的向量；max 、 exp 、$\odot$ 分别表示逐元素取最大值、求指数和相乘。$\operatorname{rowmax}$、$\operatorname{rowsum}$ 表示逐行取最大值、求和。$\operatorname{diag}(z)$ 是以向量 z 为对角线的矩阵，左乘它相当于按 z 缩放各行，逆矩阵则按各行除以对应元素。

逐行应用第 3 节更新式：

$$
\begin{aligned}
m'&=\max(m_a,\operatorname{rowmax}(S_{ab})),\\
\alpha&=\exp(m_a-m'),\\
(W_{ab})_{rc}&=\exp((S_{ab})_{rc}-m'_r),\\
\ell'&=\alpha\odot\ell_a+\operatorname{rowsum}(W_{ab}),\\
U_a'&=\operatorname{diag}(\alpha)U_a+W_{ab}V_b.
\end{aligned}
$$

$r=1,\ldots,B_r$、$c=1,\ldots,B_c$ 是块内行列下标。两次主要矩阵乘法是 $Q_aK_b^\top$ 和 $W_{ab}V_b$。处理完所有 K/V 块后，用最终统计量计算：

$$
O_a=\operatorname{diag}(\ell_a)^{-1}U_a,\qquad
L_a=m_a+\log\ell_a.
$$

$O_a\in\mathbb{R}^{B_r\times d}$ 为归一化输出，$L_a\in\mathbb{R}^{B_r}$ 为逐行 log-sum-exp，供反向恢复概率；log 为自然对数。计算过程中无需构造对角矩阵，按行缩放即可。

### 4.2 前向计算顺序与数据位置

外层遍历 Q 块，内层遍历 K/V 块。每个线程块保留自己的 $Q_a,m_a,\ell_a,U_a$，每次加载一块 K/V，计算局部分数和权重，随即更新分母与加权分子。局部分数和权重使用后即可丢弃；全部 K/V 块处理完后，才归一化并写回 $O_a,L_a$。

不同 Q 块的输出彼此独立，可以分配给不同线程块并行计算。完整的 $N\times N$ 分数与概率矩阵都不需要写入 HBM（GPU 显存）。

### 4.3 为什么能减少显存访问

设片上可用容量 C 以元素数计。复杂度记号 $O(f)$ 表示渐近上界，$\Theta(f)$ 表示同阶的上下界，均忽略固定常数，与输出矩阵 O 无关。主要片上数据占用为：

$$
O(B_rd+B_cd+B_rB_c),
$$

分别对应 Q 累加状态、 K/V 和局部分数等存储。块大小需要使这些数据同时放入片上空间。

Q 块数为 $T_r=\lceil N/B_r\rceil$，其中 $\lceil\cdot\rceil$ 表示向上取整。每个 Q 块读取一次全部 K/V，忽略跨线程块缓存复用时，需要 $O(Nd)$ 次元素读取；Q 读取和最终输出写回合计 $O(Nd)$，因此：

$$
\text{HBM 访问量}
=O(Nd+T_rNd)
=O\left(Nd+\frac{N^2d}{B_r}\right).
$$

这个表达式对应本文采用的 Q 块外层循环。增大 $B_r$ 可以复用更多 K/V 数据，但同时增加片上状态和分数块的容量需求，并可能限制并行度，不能无限增大。

显式构造 S/P 的标准实现仅这些矩阵的读写就有 $\Theta(N^2)$ 个元素。分块融合省去这部分搬运，但仍存在重复读取 K/V 的成本；实际访问量与块大小、缓存复用有关。算术量仍为 $O(N^2d)$，因为每个有效 query-key 配对仍参与计算。

### 4.4 掩码与数值细节

- **因果掩码**：若 key 的位置大于 query 的位置，该分数为 $-\infty$。完全位于未来的块可直接跳过；完全位于过去的块不必逐元素施加因果掩码；与对角线相交的块需要逐元素判断。

- **全屏蔽块或行**：旧、新最大值均为 $-\infty$ 时，直接相减会产生 NaN 。实现应逐行跳过无有效贡献的块，让已有状态保持不变；尚无有效 key 的行保持 $\ell=0,U=0$。若整行都被屏蔽，softmax 本身没有定义，需由接口约定输出行为。

- **尾块**：当 $N$ 不是块大小的整数倍时，越界位置需要屏蔽，不能参与最大值或求和。

- **精度与块大小**：GPU 内核通常用 FP32 维护归一化统计量和输出累加器。块大小需结合注意力头维度、寄存器、共享内存和占用率选择，并非越大越快。

## 5 反向重计算

### 5.1 保存什么

前向保留 Q 、 K 、 V 、 O，以及逐行统计向量 $L\in\mathbb{R}^N$。对第 i 个 query：

$$
L_i=m_i+\log\ell_i=\log\sum_j e^{S_{ij}}.
$$

$m_i,\ell_i$ 是前向结束时第 i 行的最大值和指数和，合并为 L 后不需要分别保存。反向按块重算分数 $S_{ij}$，再恢复对应概率：

$$
P_{ij}=e^{S_{ij}-L_i}
=\frac{e^{S_{ij}-m_i}}{\ell_i}.
$$

这里 i 、 j 表示单个 query 和 key 的位置。一次只恢复当前块所需的概率，计算梯度后就丢弃，不必保存完整 S 或 P 。

### 5.2 梯度公式

设 $\mathcal{L}$ 为标量损失函数，$\nabla_X\mathcal{L}=\partial\mathcal{L}/\partial X$ 表示它对矩阵 X 的梯度，形状与 X 相同。$G=\partial\mathcal{L}/\partial O\in\mathbb{R}^{N\times d}$ 是上游梯度；$\mathcal{L}$ 与逐行统计量 L 是不同对象。由 $O=PV$ 的矩阵乘法求导得到：

$$
\nabla_V\mathcal{L}=P^\top G,\qquad
\nabla_P\mathcal{L}=GV^\top.
$$

以下 i 为 query 位置，j 、 t 为 key 位置，k 为向量分量下标；$\delta_{jt}$ 在 $j=t$ 时为 1，否则为 0 。 softmax 的雅可比为：

$$
\frac{\partial P_{ij}}{\partial S_{it}}=P_{ij}(\delta_{jt}-P_{it}).
$$

代入链式法则，对同一行的所有 j 求和：

$$
\begin{aligned}
(\nabla_S\mathcal{L})_{it}
&=\sum_j(\nabla_P\mathcal{L})_{ij}P_{ij}(\delta_{jt}-P_{it})\\
&=P_{it}\left((\nabla_P\mathcal{L})_{it}
-\sum_jP_{ij}(\nabla_P\mathcal{L})_{ij}\right).
\end{aligned}
$$

其中看起来需要遍历整行概率的求和，可以利用 $O=PV$ 改写：

$$
\begin{aligned}
\sum_jP_{ij}(\nabla_P\mathcal{L})_{ij}
&=\sum_jP_{ij}\sum_kG_{ik}V_{jk}\\
&=\sum_kG_{ik}\left(\sum_jP_{ij}V_{jk}\right)
=\sum_kG_{ik}O_{ik}.
\end{aligned}
$$

因此先用已保存的 O 和上游梯度 G 求出每行一个标量 D：

$$
D_i=\sum_jP_{ij}(\nabla_P\mathcal{L})_{ij}
=\sum_{k=1}^{d}G_{ik}O_{ik},
$$

$$
(\nabla_S\mathcal{L})_{ij}
=P_{ij}\left((\nabla_P\mathcal{L})_{ij}-D_i\right).
$$

$D\in\mathbb{R}^{N}$ 可以直接从 G 和 O 计算，无需先构造整行 P 。由 $S=QK^\top/\sqrt d+M$ 得：

$$
\nabla_Q\mathcal{L}=\frac{(\nabla_S\mathcal{L})K}{\sqrt d},\qquad
\nabla_K\mathcal{L}=\frac{(\nabla_S\mathcal{L})^\top Q}{\sqrt d}.
$$

计算每个局部块的 P 、$\nabla_P\mathcal{L}$ 和 $\nabla_S\mathcal{L}$ 后，立即累加到 Q/K/V 梯度，随后复用临时存储。使用 dropout 时，还需恢复与前向一致的随机掩码及缩放，不能直接照搬上述无 dropout 公式。

### 5.3 重计算与显存节省的关系

标准反向若保留完整 P，需要额外保存 N×N 个概率值。保存 L 共 N 个统计量，反向分块重算 QKᵀ 和指数权重；局部 P 随即用于 V 、 S 、 Q 、 K 的梯度计算，不产生完整的梯度中间矩阵。

因此前向融合和反向重计算需要配合：只在前向不保存 S，却为了反向仍保留完整 P，就仍然存在二次规模的激活存储。重计算多做了一次局部分数矩阵乘法等运算，但减少了大矩阵的显存读写；GPU 上矩阵乘法吞吐较高时，这种取舍可以同时节省显存和时间。

## 6 并行组织与前向实现

### 6.1 线程块：沿 query 序列维度并行

令 $n_{\mathrm{batch}}$ 表示批量大小，H 表示查询头数。仅按样本和注意力头分配线程块时，长序列、小 batch 的任务可能没有足够多的线程块填满 GPU 。 FA2 前向进一步把不同的 query 行块分配给不同线程块：

$$
\text{线程块数}\approx n_{\mathrm{batch}}\times H\times\left\lceil N/B_r\right\rceil.
$$

每个线程块持有自己的 Q 块，遍历 K/V 块，独立计算对应的输出行。不同 query 行之间没有 softmax 归一化依赖，因此前向无需跨线程块合并输出。代价是不同 Q 块会重复读取 K/V，需要结合缓存和块大小权衡。

### 6.2 Warp：按 Q 行划分

CUDA 中一个 warp 通常有 32 个线程，一个线程块包含多个 warp 。线程块之间的并行与线程块内部的工作划分是两个层次。

线程块内部按 Q 行划分任务，不同 warp 共享 K/V，各自计算不同输出行。这种划分称为 sliced-Q；每个输出行的加权累加由负责该行的 warp 完成，无需跨 warp 合并不同 K/V 片段对同一输出行的贡献。

这样减少了共享内存读写和同步开销。实际内核仍需要协作加载数据；“减少通信”不等于完全不使用共享内存或同步。反向的依赖更多，不能直接照搬前向的划分方式。

### 6.3 反向的线程块划分

FA2 论文中的反向按 K/V 列块分配线程块，每个线程块遍历 Q 行块，在本地累加自己负责的 K 、 V 梯度。多个列块会对同一 Q 梯度行块产生贡献，论文实现使用原子加法合并这些贡献。这与前向各线程块独立写不同 O 行的情况不同。

### 6.4 Pytorch 实现代码

以下符号与 CS336 课程作业 2 的算法图对应，随后是前向实现：

| 符号 | 定义与正文对应关系 |
|---|---|
| $N_q,N_k$ | query 与 key/value 的序列长度；前文自注意力推导取二者均为 N |
| $Q\in\mathbb{R}^{N_q\times d}$，$K,V\in\mathbb{R}^{N_k\times d}$ | 输入矩阵，d 仍为各向量的维度 |
| $B_q,B_k$ | 每个 Q 块、 K/V 块的行数，分别对应前文 $B_r,B_c$ |
| $T_q=\lceil N_q/B_q\rceil,T_k=\lceil N_k/B_k\rceil$ | Q 块数与 K/V 块数 |
| $Q_i,K^{(j)},V^{(j)}$ | 第 i 个 Q 块、第 j 个 K/V 块；此处 i 、 j 都是块编号 |
| $S_i^{(j)},\widetilde P_i^{(j)}\in\mathbb{R}^{B_q\times B_k}$ | 当前分数块与未归一化的指数权重块 |
| $m_i^{(j)},\ell_i^{(j)}\in\mathbb{R}^{B_q}$ | 第 i 个 Q 块处理完前 j 个 K/V 块后的逐行最大值与指数和 |
| $O_i^{(j)}\in\mathbb{R}^{B_q\times d}$ | **未归一化的加权分子**，对应正文 U；最终除以分母才得到输出 |
| $O_i\in\mathbb{R}^{B_q\times d},L_i\in\mathbb{R}^{B_q}$ | 最终归一化输出块、逐行 log-sum-exp 统计量 |

上标 $(j)$ 表示块编号或处理进度，不表示幂。$\operatorname{rowmax}$、$\operatorname{rowsum}$ 分别表示沿每一行取最大值、求和；矩阵减行向量表示从每一行减去该行对应的标量。 exp 和 log 分别是逐元素自然指数和自然对数，$\odot$ 为逐元素乘法，diag 的含义见第 4.1 节。

使用 pytorch 实现的 flashattention 代码

```python
class FlashAttentionTorch(autograd.Function):
    @staticmethod
    def forward(cxt, Q, K, V, is_causal=False):
        # Q: (..., N_q, d), K/V: (..., N_k, d)
        N_q, d = Q.shape[-2], Q.shape[-1]
        N_k = K.shape[-2]
        T_q = (N_q + B_q - 1) // B_q     # 向上取整
        T_k = (N_k + B_k - 1) // B_k

        # 半精度输入使用 float32 计算分数及累加，输出恢复输入类型。
        acc_dtype = torch.float32 if Q.dtype in (torch.float16, torch.bfloat16) else Q.dtype
        O = torch.empty_like(Q)
        L = torch.empty(Q.shape[:-1], device=Q.device, dtype=acc_dtype)

        for i in range(T_q):
            q_start = i * B_q
            q_end = min(q_start + B_q, N_q)
            # 直接切片，最后一个块可以不足 B_q 行。
            Q_i = Q[..., q_start:q_end, :].to(acc_dtype)
            O_i = torch.zeros_like(Q_i)
            l_i = torch.zeros(Q_i.shape[:-1], device=Q.device, dtype=acc_dtype)
            m_i = torch.full_like(l_i, float('-inf'))

            for j in range(T_k):
                m_i_pre = m_i
                l_i_pre = l_i
                O_i_pre = O_i
                k_start = j * B_k
                k_end = min(k_start + B_k, N_k)
                K_j = K[..., k_start:k_end, :].to(acc_dtype)
                V_j = V[..., k_start:k_end, :].to(acc_dtype)
                # Compute tile of pre-softmax attention scores
                S_i = einops.einsum(Q_i, K_j, "... B_q d, ... B_k d -> ... B_q B_k") / d ** 0.5
                m_i = torch.max(m_i_pre, S_i.amax(dim=-1))
                P_i = torch.exp(S_i - m_i.unsqueeze(-1))
                alpha = torch.exp(m_i_pre - m_i)
                l_i = alpha * l_i_pre + torch.sum(P_i, dim=-1)
                O_i = alpha.unsqueeze(-1) * O_i_pre + P_i @ V_j

            O_i = O_i / l_i.unsqueeze(-1)
            l_i = m_i + torch.log(l_i)
            O[..., q_start:q_end, :] = O_i.to(Q.dtype)
            L[..., q_start:q_end] = l_i

        cxt.save_for_backward(Q, K, V, O, L)
        return O
```

代码中的 `l_i` 对应公式的 $\ell_i$，`P_i` 对应 $W_{ab}$，`O_i` 在内层循环中对应未归一化分子 U 。`P_i` 直接使用累计最大值计算，旧分母和旧分子均乘以相同的 `alpha` 修正。

循环内更新分母和分子的两行对应第 3 、 4 节的递推。内层循环结束后，`O_i` 除以 `l_i` 才得到归一化输出；随后 `l_i` 被赋值为 $m_i+\log\ell_i$ 并存入 L 。全局输出 $O\in\mathbb{R}^{N_q\times d}$ 与统计量 $L\in\mathbb{R}^{N_q}$ 由各块按 query 顺序拼接。

## 7 训练、 Prefill 与 Decode

预填充（prefill）指一次处理输入上下文，解码（decode）指随后逐步生成新词元。训练和 prefill 通常同时处理很多 query，能利用 Q 行块并行和矩阵乘法。自回归 decode 每步通常只有一个新 query，而 K/V 来自逐渐增长的键值缓存（KV cache），此时分数形状约为 $1\times N$，计算常受 KV cache 读取带宽和并行度限制。

为了让长 KV cache 的 decode 使用更多线程块，可以沿 K/V 序列切分。设第 b 个分片的 key 下标集合为 $J_b$，各集合互不重叠且覆盖全部有效 key 。每个分片计算 $m_b=\max_{j\in J_b}x_j$、$\ell_b=\sum_{j\in J_b}e^{x_j-m_b}$、$u_b=\sum_{j\in J_b}e^{x_j-m_b}v_j$，最后归并：

$$
\begin{aligned}
m&=\max_bm_b,\\
\ell&=\sum_b e^{m_b-m}\ell_b,\\
u&=\sum_b e^{m_b-m}u_b,\qquad o=u/\ell.
\end{aligned}
$$

这里多了一次跨块归并，与 FA2 前向按 Q 行独立并行不同。这类优化常见于 Flash-Decoding 等解码方案，应与 FA2 论文的主要贡献区分。

多查询注意力（MQA）和分组查询注意力（GQA）则通过多个 query head 共享 K/V head 减少 KV cache 容量，可与 FlashAttention 结合。 FlashAttention 本身不会消除 KV cache，也不会把长上下文 decode 的 K/V 读取变成常数开销。

## 8 复杂度与适用范围

### 8.1 复杂度变化

| 项目 | 显式构造 S/P 的标准实现 | FlashAttention / FA2 |
|---|---|---|
| 稠密注意力算术量 | $O(N^2d)$ | $O(N^2d)$ |
| 完整 S/P 的 HBM 存储 | $O(N^2)$ | 无需存储 |
| 输入与输出 | $O(Nd)$ | $O(Nd)$ |
| 前向供反向使用的归一化统计量 | 依实现而定 | 保存 L，占 $O(N)$ |
| 片上临时数据 | 依内核而定 | 由分块尺寸和 d 决定 |

“线性显存”指消除注意力的二次规模中间激活，不代表整个模型的全部内存都只占 $O(N)$，也不代表总显存访问量是线性的。按 Q 块独立计算时，仍需重复读取 K/V；对应的访问量估算见第 4.3 节。

### 8.2 使用限制

长序列训练和 prefill 通常能受益于减少中间激活和显存读写。短序列或很小的任务中，内核启动与调度开销可能占较大比例。特殊掩码、数据类型和注意力头维度是否支持，需要查看具体实现。

如果业务需要返回完整的注意力概率矩阵用于分析，就仍需输出 $N\times N$ 数据，无法保留“不存储完整 P”的全部收益。 FlashAttention 也没有降低稠密注意力的二次计算量，极长序列仍可能需要稀疏注意力或其他模型设计。
