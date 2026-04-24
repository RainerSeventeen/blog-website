---
paper: "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
---

# Sentence-BERT

> 文章全名：*Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*
>
> 具体实现参考 github 库 [sentence-transformers](https://github.com/huggingface/sentence-transformers)

## 研究目标

研究的目标是 **Semantic Textual Similarity** 探究两个句子之间的语义相似度，在 BERT 中必须对两个句子之间两两执行这个运算，这个代价很昂贵，不可行

核心原因就是 BERT 是 cross-encoder（输出依赖于句子对，而不是独立的一个句子），但是语义搜索需要 independent embeddings

## 方案

### Pooling

用 Siamese 结构微调 BERT，就能得到可比较的句向量，命名为 Sentence-BERT（SBERT）

方法是在 BERT 的上面再加一层 pooling，包括 CLS, MEAN, MAX 三种

- CLS: 使用输出的 cls token，但是效果很差，不适合 cosine 距离
- MEAN（实际使用的）: 对所有的 token 执行平均操作，更加抗噪声，相对平滑，适合 cosine 距离
- MAX: 每一个维度都选择最大的那个值，强化句子的特征（实际 cosine 表现是最差的）

### Classification (使用 NLI[^2])

训练的分类目标 $o = \mathrm{softmax}\!\left( W_t [u,\, v,\, |u - v|] \right)$ （后面再经过一个 cross entropy）其中 $u, v$ 是第一个第二个句子的向量，$|u - v|$ 是逐个元素的绝对值差， 拼接后得到一个三维的向量

训练的时候 BERT 本身的全部参数都是加入反向传播的，BERT 的所有参数 + 这个分类目标的 $\! W_t$ 都是参加到训练中的，但是这个参数只在训练的时候保留，推理的时候只使用 cosine 计算相似度

- 为什么 NLI 有用？

因为entailment 构成远距离，contradiction构成近距离，这是自然的语义几何结构

### Regression (使用 STS[^3])

回归目标是直接计算两个句子之间的 cosine 距离差值，然后计算 $\text{MSE}(\cos(u,v), \text{gold score})$ 使用的是 STS 的 0~5 的那个相似度分数

目标是让 cosine 相似度等于真实语义相似度，适合检索任务

### Triplet

三元组目标 $\max(||a-p|| - ||a-n|| + \epsilon, 0)$ 

指的是给出  anchor, positive, negative 三句话，然后让这个值尽可能的大，执行局部几何排序结构。关心的不是绝对值，而是两两之间的相对距离

Triplet 是 Metric Learning（度量学习），目标就是让相关的更近，错误的更远

如果 $||a - p|| + \epsilon < ||a - n||$ 则这个时候 loss 是 0，已经没有梯度了，到达了训练目标



[^1]: 词表（vocabulary）：所有可被模型索引的 token 的集合。 这里不能预定义指的是**不能在不看语料的情况下提前确定真实子词集合大小**，如果统计所有可能的词表（一个非常大的排列组合，现实也不会这么做）会用到 hash 表，这表示我们在看到完整的语料之前是不知道会出现哪些词语的

[^2]: **Natural Language Inference** 自然语言推断，是一个三分类任务，给出Premise（前提）, Hypothesis（假设）判断他们属于entailment（包含）, contradiction（矛盾）, neutral（无关）

[^3]: **Semantic Textual Similarity** 语义文本相似度，是一个回归任务，输出一个 0 ~ 5 之间的分数（连续值）

