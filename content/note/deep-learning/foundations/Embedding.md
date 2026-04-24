# Embedding

## 基础

所有的语言模型等最基础的一层，经过 tokenizer 分词之后就是 embedding 的工作

在 pytroch 中可以使用 `nn.Embedding` 模块来调用，本质上是一个可以学习的查表矩阵

## pytorch 实现

在 pt 中 `nn.Embedding(num_embeddings, embedding_dim)` 是基本用法，

- `num_embeddings`：词表大小，或者说“可查的离散 ID 总数”
- `embedding_dim`：每个 ID 对应的向量维度

内部会构建一个  $W \in \mathbb{R}^{num\_embeddings \times embedding\_dim}$ 的可学习矩阵，对于输入一个 token id，实际做的工作是执行查表

```python
emb = nn.Embedding(10000, 512) # 把 10000 个词映射到对应 512 长度的向量空间中
x = torch.tensor([3, 7, 2]) 
out = emb(x) # 取出矩阵的的 第 3、7、2 行取出来
out.size() # (3, 512)
```

在实际训练的时候，通过梯度的反向传播可以更新到被访问的行， 对于没有出现的词语则没有梯度的贡献，这里和普通的线性层不一样， embedding 只会取一部分行

## 训练方法

embedding 是一个层的功能描述，而下面分了很多的训练方式，包括：

1. 上下文预测训练：CBOW, Skip-Gram
2. 考虑子词结构：Subword Embedding, FastText

**Embedding 的本质是一个可以学习的低维向量表示**

我们经常使用 one hot 来向量化数据，但是这对NLP领域的词不好用，因为无法表达多个词之间相似度的关系

我们经常使用 余弦相似度（Cosine Similarity） 来衡量两个向量在方向上的相似程度，本质就是两个夹角的余弦值
$$
\cos(\theta)
=
\frac{\mathbf{A} \cdot \mathbf{B}}
{\|\mathbf{A}\| \|\mathbf{B}\|}
=
\frac{\sum_{i=1}^{n} A_i B_i}
{\sqrt{\sum_{i=1}^{n} A_i^2} \;\sqrt{\sum_{i=1}^{n} B_i^2}}
$$
One hot 的所有值的余弦相似度全都是 0

## 上下文预测

### Skip-Gram

**Word2Vec 的核心思想是把高位稀疏（完全无关）的表示映射到低维稠密的连续空间**（有cosine 相关性）

对于某一个句子 "The man loves his son." 考虑中心词语 loves 中上下文中其他单词的概率

$$
$P \left(\right. "\text{the}" \mid "\text{loves}" \left.\right) \cdot P \left(\right. "\text{man}" \mid "\text{loves}" \left.\right) \cdot P \left(\right. "\text{his}" \mid "\text{loves}" \left.\right) \cdot P \left(\right. "\text{son}" \mid "\text{loves}" \left.\right)$
$$
在跳元模型中，每个词都有两个$d$（这是超参数）维向量表示，用于计算条件概率。对于词典中索引为$i$的任何词，分别用$\mathbf{v}_i\in\mathbb{R}^d$和$\mathbf{u}_i\in\mathbb{R}^d$表示其用作*中心词*和*上下文词*时的两个向量。

给定中心词$w_c$（词典中的索引$c$），生成任何上下文词$w_o$（词典中的索引$o$）的条件概率可以通过对向量点积的softmax操作来建模：
$$
P(w_o \mid w_c) = \frac{\text{exp}(\mathbf{u}_o^\top \mathbf{v}_c)}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)},
$$

- $\mathbf{v}_c$：中心词的向量（input embedding）
- $\mathbf{u}_o$：上下文词的向量（output embedding）
- $\mathbf{u}_o^\top \mathbf{v}_c$：两个向量的点积（打分）

最后用 softmax 我们就可以计算出这个概率的具体值

### CBOW

连续词袋（Continuous Bag-of-Words）模型和 Skip-Gram 反过来，是使用上下文词来预测中心词
$$
$P \left(\right. "\text{loves}" \mid "\text{the}" , "\text{man}" , "\text{his}" , "\text{son}" \left.\right)$
$$
由于连续词袋模型中存在多个上下文词，因此在计算条件概率时对这些上下文词向量进行平均。

对于字典中索引$i$的任意词，分别用$\mathbf{v}_i\in\mathbb{R}^d$和$\mathbf{u}_i\in\mathbb{R}^d$表示用作*上下文*词和*中心*词的两个向量（符号与跳元模型中相反）。给定上下文词$w_{o_1}, \ldots, w_{o_{2m}}$（在词表中索引是$o_1, \ldots, o_{2m}$）生成任意中心词$w_c$（在词表中索引是$c$）的条件概率可以由以下公式建模:
$$
P(w_c \mid w_{o_1}, \ldots, w_{o_{2m}}) = \frac{\text{exp}\left(\frac{1}{2m}\mathbf{u}_c^\top (\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}}) \right)}{ \sum_{i \in \mathcal{V}} \text{exp}\left(\frac{1}{2m}\mathbf{u}_i^\top (\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}}) \right)}
$$
另外可以看到，由于上下文词是有多个的，所以求向量 $v$ 的时候实际上是取平均值的，这里的 $m$ 是单侧窗口大小

### 对照

| **特性**   | **CBOW** | **Skip-Gram** |
| ---------- | -------- | ------------- |
| 训练速度   | 更快     | 更慢          |
| 对小数据集 | 更稳定   | 容易过拟合    |
| 对低频词   | 一般     | 更好          |
| 向量质量   | 稍弱     | 通常更好      |

Skip-Gram 为每个上下文单独建模，对稀有词学习更充分

## Subword Embedding

子词嵌入，研究词的各种形式（单复数，时态等）和词汇之间的关系，在上文中的  Skip-Grem 和 CBOW 中都是直接将不同的格式的词视作为不同的向量，需要优化这一点

### fastText

fastText 可以被视作为是子词级的跳元模型，而不是学习词级向量表示

核心思想： 将单词拆成多个字符 n-gram，然后用 n-gram 的向量求和作为词向量

例如对于单词`where` 添加首尾的标记得到 `<where>`，假设 n = 3 则得到以下几种切分方式：

`< w h,w h e,h e r,,e r e,r e >` 注意添加的那两个也算是符号，随后得到的每一个 3-gram 都有一个 embedding 向量

在fastText中，对于任意词$w$，用$\mathcal{G}_w$表示其长度在3和6（指的是这个超参数 $n \in [3,6]$）之间的所有子词与其特殊子词的并集。词表是所有词的子词的集合。假设$\mathbf{z}_g$是词典中的子词$g$的向量，则跳元模型中作为中心词的词$w$的向量$\mathbf{v}_w$是其子词向量的和：

$$
\mathbf{v}_w = \sum_{g\in\mathcal{G}_w} \mathbf{z}_g.
$$
注意在 Skip-gram 的 fastText  中只由对中心词的那个向量做 fastText 的子词操作，本质是对每一个子词都使用相同的梯度信号，这个子词的操作也是比较昂贵的

### Byte Pair Encoding

> BPE 和 fastText 是两条技术路线，他们不是迭代关系，是全新的方法

fastText 中每一个子词都是固定的长度，导致了词表的大小不能够预定义[^1]，所以这就有了 **字节对编码的** 思想来提取子词

BPE 执行训练数据集的统计分析，以发现单词内的公共符号，诸如任意长度的连续字符。从长度为1的符号开始，字节对编码迭代地合并最频繁的连续符号对以产生新的更长的符号，不考虑跨越单词边界的字符对。

总结来说，BPE 会将出现频率高的相邻的两个token合并成为一个token，然后反复重复这个工作，直到构成的词表符合这个目标大小

以英语为例子，经过合并后可能会出现 `est` ,`tion`, `ing` 这种词作为一个 token，而不是完整的字母

BPE 的算法可能用例子更好理解，看代码就行了，这里不再详细说明

## 现代 LLM

现代 LLM 中一般不在关系是 Skip-Gram 或者 CBOW 这种分词方式了，而是直接使用 BPE 类似的分词方式，先执行 tokenizer，然后 LLM 最开头直接放一个 embedding 层，参与整体的训练中

在训练的梯度反向传播中，整个 embedding 也在不断更新（所以是一个端到端的流程），有的模型会在输出层共享参数，也就是输入输出的词表是一致的



[^1]: 词表（vocabulary）：所有可被模型索引的 token 的集合。 这里不能预定义指的是**不能在不看语料的情况下提前确定真实子词集合大小**，如果统计所有可能的词表（一个非常大的排列组合，现实也不会这么做）会用到 hash 表，这表示我们在看到完整的语料之前是不知道会出现哪些词语的
