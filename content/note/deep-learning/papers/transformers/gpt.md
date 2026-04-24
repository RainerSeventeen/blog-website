---
paper: "Improving Language Understanding by Generative Pre-Training"
---

# GPT

> 论文全名：*Improving Language Understanding by Generative Pre-Training*
>
> 通常被称为 GPT-1，是 GPT 系列的开山之作，由 OpenAI 于 2018 年提出

## 简介

GPT 提出了一种半监督学习框架：先在大量无标注文本上做**生成式预训练**（Generative Pre-Training），再在各个下游任务上做**判别式微调**（Discriminative Fine-Tuning）。

核心动机是：NLP 中有标注数据稀缺，但无标注文本数据非常丰富。之前的方法（如 Word2Vec、GloVe）主要迁移的是词级别的信息，而 GPT 的目标是学到更高层次的语义表示，并能以最小的架构改动迁移到各种下游任务。

与之前基于 LSTM 的预训练方法不同，GPT 选择了 **Transformer Decoder** 作为基础架构，因为 Transformer 的结构化记忆能更好地处理长距离依赖。

最终 GPT 在 12 个任务中的 9 个取得了 SOTA，包括常识推理（Story Cloze +8.9%）、问答（RACE +5.7%）和文本蕴含（MultiNLI +1.5%）等。

## 框架

整个训练流程分为两个阶段：

1. **无监督预训练**：在大规模无标注语料上训练语言模型
2. **有监督微调**：在下游任务的标注数据上微调模型参数

### 无监督预训练

给定一个无标注的 token 序列 $\mathcal{U} = \{u_1, \dots, u_n\}$，使用标准的语言模型目标，最大化以下似然：

$$
L_1(\mathcal{U}) = \sum_i \log P(u_i \mid u_{i-k}, \dots, u_{i-1}; \Theta)
$$

其中 $k$ 是上下文窗口大小，$\Theta$ 是模型参数。也就是给定前面 $k$ 个 token，预测下一个 token 的概率，然后对所有位置求和取对数——就是经典的自回归语言建模目标。

模型使用多层 Transformer Decoder，具体的前向计算过程是：

$$
h_0 = UW_e + W_p
$$

$$
h_l = \text{transformer\_block}(h_{l-1}), \quad \forall l \in [1, n]
$$

$$
P(u) = \text{softmax}(h_n W_e^T)
$$

- $U = (u_{-k}, \dots, u_{-1})$ 是上下文的 token 序列
- $W_e$ 是 token embedding 矩阵
- $W_p$ 是 position embedding 矩阵（GPT 用的是**可学习的**位置编码，不是原始 Transformer 的正余弦编码）
- $h_l$ 是第 $l$ 层 Transformer block 的输出
- 最后一层输出 $h_n$ 乘以 $W_e^T$（权重共享）再做 softmax 得到下一个 token 的概率分布

### 有监督微调

预训练完成后，将模型迁移到有标注的下游任务。假设标注数据集 $\mathcal{C}$ 中每个样本是一个 token 序列 $x^1, \dots, x^m$ 和对应的标签 $y$。

将输入送入预训练模型，取最后一层 Transformer block 的输出 $h_l^m$，接一个线性层预测标签：

$$
P(y \mid x^1, \dots, x^m) = \text{softmax}(h_l^m W_y)
$$

微调的目标函数是：

$$
L_2(\mathcal{C}) = \sum_{(x, y)} \log P(y \mid x^1, \dots, x^m)
$$

此外，作者发现在微调时**同时加入语言模型目标作为辅助损失**，可以带来两个好处：提升泛化能力，加速收敛。因此最终的微调目标函数是：

$$
L_3(\mathcal{C}) = L_2(\mathcal{C}) + \lambda \cdot L_1(\mathcal{C})
$$

其中 $\lambda$ 是辅助目标的权重，论文中设为 0.5。

微调时额外需要的参数非常少，只有输出层的 $W_y$ 和分隔符 token 的 embedding。

### 任务特定的输入变换

由于预训练模型是在连续文本上训练的，但下游任务的输入格式各不相同（句对、三元组等），GPT 设计了一系列**输入变换**（Input Transformations），将各种结构化输入转换成模型可以处理的连续 token 序列，从而避免对模型架构做大幅改动。

所有变换都会加入随机初始化的起始标记 $\langle s \rangle$ 和结束标记 $\langle e \rangle$。

**分类（Classification）**：直接将文本拼接为 `[Start] Text [Extract]`，送入 Transformer 后接线性层分类。

**文本蕴含（Entailment）**：将前提 $p$ 和假设 $h$ 用分隔符 `$` 连接：`[Start] Premise [Delim] Hypothesis [Extract]`。

**相似度（Similarity）**：因为两个句子没有固定顺序，所以构造两种排列（AB 和 BA），分别送入模型得到两个表示 $h_l^m$，逐元素相加后再送入线性层。

**问答与常识推理（Multiple Choice）**：给定上下文 $z$、问题 $q$ 和候选答案集合 $\{a_k\}$，为每个答案构造一个序列 `[Start] Context [Delim] Answer_k [Extract]`，分别过模型后用 softmax 在候选答案上归一化。

## 模型细节

### 预训练

- **数据集**：BooksCorpus（约 7000 本未出版的书籍），包含长段连续文本，有利于学习长距离依赖
- **架构**：12 层 Transformer Decoder，768 维隐藏状态，12 个注意力头
- **FFN 内部维度**：3072（即 $4 \times 768$）
- **上下文窗口**：512 个 token
- **优化器**：Adam，最大学习率 2.5e-4，前 2000 步线性 warmup，之后余弦退火到 0
- **训练**：100 个 epoch，batch size = 64
- **词表**：BPE（Byte Pair Encoding），40000 次合并操作
- **正则化**：残差、Embedding、注意力的 dropout 率 = 0.1；修改版 L2 正则化（$w = 0.01$）
- **激活函数**：GELU（而非原始 Transformer 的 ReLU）
- **位置编码**：可学习的位置 embedding（而非正余弦）
- **归一化**：LayerNorm，权重初始化 $\mathcal{N}(0, 0.02)$

### 微调

- 大部分超参数复用预训练的设置
- 分类器 dropout 率 = 0.1
- 学习率 = 6.25e-5，batch size = 32
- 大部分任务训练 3 个 epoch 就足够
- 线性学习率衰减，warmup 占训练的 0.2%
- $\lambda = 0.5$

## 实验结果

### 自然语言推理（NLI）

在 5 个 NLI 数据集上评估（SNLI、MNLI、QNLI、SciTail、RTE），在其中 4 个数据集上超过了之前的 SOTA：

- MNLI：82.1（之前最好 80.6）
- SciTail：88.3（之前最好 83.3）
- QNLI：88.1（之前最好 82.3）
- SNLI：89.9（之前最好 89.3）

### 问答与常识推理

- Story Cloze：86.5（之前最好 77.6，提升 **+8.9**）
- RACE 总分：59.0（之前最好 53.3，提升 **+5.7**）

### 语义相似度

在 MRPC、QQP、STS-B 三个数据集上取得了有竞争力的结果，QQP 上比之前的 SOTA 提高了 4.2 个绝对点。

### 分类

- CoLA：45.4（之前最好 35.0，提升 **+10.4**）
- SST-2：91.3（接近 SOTA）
- GLUE 总分：72.8（之前最好 68.9）

总共在 12 个数据集中的 9 个取得了新的 SOTA。

## 分析

### 迁移层数的影响

论文研究了从预训练模型迁移不同层数的效果：

- 每多迁移一层 Transformer，下游性能都有提升
- 在 MultiNLI 上，完整迁移所有 12 层相比只迁移 Embedding 提升高达 9%
- 说明预训练模型的**每一层都包含了对下游任务有用的信息**

### Zero-shot 行为

在预训练过程中，模型逐渐习得了多种下游任务的能力（即使没有经过微调）。随着预训练时间的增加，zero-shot 在情感分析、Winograd 消歧、问答等任务上的表现持续稳定地提升，这说明生成式预训练确实在学习与各种任务相关的语言能力。

同时 Transformer 在 zero-shot 上的表现方差比 LSTM 更大，暗示 Transformer 的归纳偏置（inductive bias）在迁移学习中提供了更大的帮助。

### 消融实验

论文做了三组消融实验：

1. **去掉辅助 LM 目标**：在较大数据集（NLI、QQP）上有帮助，较小数据集上帮助不明显
2. **用 LSTM 替换 Transformer**：平均得分下降 5.6，仅在 MRPC 上 LSTM 略优
3. **去掉预训练（直接在下游任务训练 Transformer）**：性能下降 14.8%，证明预训练至关重要

## 总结与思考

GPT-1 的核心贡献：

1. **提出了"预训练 + 微调"范式**：这一范式后来成为 NLP 的标准做法，直接催生了 BERT、GPT-2/3/4 等后续工作
2. **将 Transformer 用于预训练**：之前的工作（如 ULMFiT）用 LSTM 做预训练，GPT 首次证明了 Transformer 的优越性
3. **统一的输入变换**：通过巧妙的输入格式设计，避免了为每个任务设计不同架构，使得一个模型可以适配多种任务
4. **辅助目标的价值**：微调时加入 LM 辅助目标能提升泛化和收敛速度

与后续的 BERT 的关键区别在于：GPT 使用的是**单向**（从左到右）的 Transformer Decoder，而 BERT 使用的是**双向**的 Transformer Encoder。这使得 BERT 在理解类任务上更有优势，但 GPT 的自回归特性让它天然适合文本生成任务。

---

# GPT-2

> 论文全名：*Language Models are Unsupervised Multitask Learners*
>
> GPT-2 由 OpenAI 于 2019 年提出，是 GPT-1 的直接继承者

## 简介

GPT-2 的核心主张是：**语言模型是无监督的多任务学习者**。当语言模型在足够大且足够多样的数据集上训练时，它能够在**无需任何显式监督**的情况下，开始学会执行各种 NLP 任务。

与 GPT-1 的"预训练 + 微调"范式不同，GPT-2 完全聚焦于 **zero-shot** 设置——不对下游任务做任何参数更新或架构修改，直接用语言模型在各种任务上进行评估。

GPT-2 的动机来自对当前 ML 系统的批判：现有系统在单一数据集上训练，本质上是"狭隘专家"（narrow experts），对数据分布的微小变化非常脆弱。作者认为，要构建真正鲁棒的通用系统，需要在多种领域和任务上训练和评估。

最大的 GPT-2 模型拥有 **15 亿参数**（比 GPT-1 大 10 倍以上），在 8 个语言建模数据集中的 7 个上达到了 zero-shot SOTA。

## 核心思想

### 任务条件化

GPT-2 对多任务学习提出了一种新的视角。传统的多任务学习建模为 $p(\text{output} \mid \text{input})$，但一个真正通用的系统应该建模为：

$$
p(\text{output} \mid \text{input}, \text{task})
$$

也就是说，模型不仅要根据输入产生输出，还需要知道当前要执行的是什么任务。

以往的方法通过架构层面的 task-specific head 或者算法层面的元学习（如 MAML）来实现任务条件化。而 GPT-2 的关键洞察是：**语言本身就可以天然地指定任务**。例如：

- 翻译任务可以表示为：`translate to french, english text, french text`
- 阅读理解可以表示为：`answer the question, document, question, answer`

因此，一个足够强大的语言模型在做下一个 token 预测的过程中，就隐式地在学习所有这些可以用自然语言描述的任务。

### 语言模型建模

与 GPT-1 相同，GPT-2 使用自回归语言模型目标，将序列的联合概率分解为条件概率的乘积：

$$
p(x) = \prod_{i=1}^{n} p(s_n \mid s_1, \dots, s_{n-1})
$$

GPT-2 的核心假设是：如果语言模型的容量足够大，它为了最大化语言建模的似然（更好地预测下一个 token），就会被迫去学习各种自然语言中隐含的任务模式。

## 训练数据：WebText

之前的语言模型大多在单一领域上训练（新闻、Wikipedia、小说等）。GPT-2 的目标是构建一个尽可能**大**且**多样**的数据集。

### 构建方式

- 从 Reddit 上抓取所有获得至少 3 个 karma 的外链（以此作为人工筛选质量的代理）
- 共获得约 **4500 万个链接**
- 使用 Dragnet 和 Newspaper 提取正文内容
- 经过去重和启发式清洗后，最终得到约 **800 万篇文档**，总计约 **40 GB** 文本
- 移除了所有 Wikipedia 内容，避免与下游评估数据集重叠

### 与 GPT-1 数据的对比

| | GPT-1 | GPT-2 |
|---|---|---|
| 数据集 | BooksCorpus | WebText |
| 规模 | ~5 GB | ~40 GB |
| 领域 | 未出版小说 | 多领域网页 |

## 输入表示：Byte-Level BPE

GPT-2 使用了 **Byte-Level BPE**（字节级 BPE），这是对 GPT-1 使用的标准 BPE 的改进。

传统 BPE 在 Unicode 码点上操作，需要一个很大的基础词表（超过 130,000 个 Unicode 符号）来覆盖所有语言，这远大于通常使用的 32,000-64,000 的词表大小。

而 Byte-Level BPE 以 **UTF-8 字节**为基本单元，基础词表仅需 256 个字节，然后在此基础上做 BPE 合并。这带来了几个好处：

1. 可以表示**任何** Unicode 字符串，不会出现 `<UNK>` token
2. 可以在**任何**数据集上评估，无需特殊的预处理或 tokenization
3. 词表大小扩展到 **50,257**

为防止 BPE 跨字符类别合并（如 `dog`、`dog!`、`dog?` 被当作不同 token），作者禁止了跨字符类别的合并操作（空格除外），提升了压缩效率。

## 模型架构

GPT-2 沿用 GPT-1 的 Transformer Decoder 架构，但做了若干修改：

### 与 GPT-1 的架构差异

1. **Layer Normalization 前置**：将 LayerNorm 移到每个 sub-block 的**输入**位置（Pre-Norm），类似于 pre-activation ResNet，而 GPT-1 是 Post-Norm
2. **额外的 LayerNorm**：在最后一个 self-attention block 之后增加了一个额外的 LayerNorm
3. **残差初始化缩放**：随模型深度缩放残差路径的权重初始化，缩放因子为 $1/\sqrt{N}$，其中 $N$ 是残差层总数，目的是随着模型变深，控制残差路径的初始贡献
4. **上下文窗口**：从 512 扩展到 **1024** 个 token
5. **Batch size**：增大到 **512**

### 模型规模

GPT-2 训练了 4 个不同大小的模型：

| 参数量 | 层数 | $d_{\text{model}}$ |
|---|---|---|
| 117M | 12 | 768 |
| 345M | 24 | 1024 |
| 762M | 36 | 1280 |
| 1542M (GPT-2) | 48 | 1600 |

最小的 117M 模型与 GPT-1 等价，345M 模型与 BERT-Large 等价。最大的 GPT-2（1542M）比 GPT-1 大了一个数量级。

所有模型在 WebText 上仍然是 **欠拟合** 的（训练和测试 perplexity 都还在下降），给更多训练时间还可以继续提升。

## 实验结果

所有实验均为 **zero-shot**，不做任何微调。

### 语言建模

在 8 个语言建模基准中的 7 个上达到了 SOTA：

- **LAMBADA**（长距离依赖）：perplexity 从 99.8 降到 **8.63**，准确率从 19% 提升到 **63.24%**
- **Children's Book Test**：common nouns 准确率 **93.30%**，named entities 准确率 **89.05%**
- **Penn Treebank**：perplexity **35.76**
- **WikiText-2**：perplexity **18.34**

唯一未超越的是 One Billion Word Benchmark（1BW），可能因为它经过了句子级打乱，破坏了长距离结构，且其激进的预处理（标准化文本、断开缩写）与 WebText 差异较大。

性能与模型大小呈 **log-linear** 关系增长：模型每大一个级别，各任务性能都有一致提升。

### 阅读理解（CoQA）

在对话式问答数据集 CoQA 上，GPT-2 通过 greedy decoding（给定文档和对话历史，用 `A:` 作为最终提示）达到了 **55 F1**，匹配或超过了 4 个有监督 baseline 中的 3 个——而这些 baseline 使用了 127,000+ 标注样本训练。

### 摘要（CNN/Daily Mail）

在文章末尾添加 `TL;DR:` 提示后，使用 Top-$k$ 随机采样（$k=2$）生成 100 个 token，取前 3 句作为摘要。

在 ROUGE 指标上 GPT-2 接近了经典的神经网络 baseline，但仍弱于 SOTA。去掉 `TL;DR:` 提示后 ROUGE 下降了 6.4 分，说明模型确实理解了这个**任务提示**的含义。

### 翻译（WMT-14）

使用 few-shot 格式（`english sentence = french sentence` 的示例对）作为上下文，GPT-2 在英法翻译上获得 **5 BLEU**，在法英翻译上获得 **11.5 BLEU**。

法英方向远好于英法，这并不令人意外——WebText 以英语为主，模型的英语生成能力远强于法语。令人惊讶的是，训练数据中仅有约 10MB 的法语文本（比专门的无监督翻译语料小 500 倍），模型仍能展现出翻译能力。

### 问答（Natural Questions）

GPT-2 在最有信心的 1% 问题上达到了 **63.1%** 的准确率。整体上回答正确的问题数量是最简单 baseline（返回最常见答案类型）的 **5.3 倍**。但仍远弱于结合了检索的开放域问答系统（30-50% 准确率范围）。

### Winograd 消歧

GPT-2 达到了 **70.70%** 的准确率，比之前的 SOTA 提升了 7 个百分点。

## 泛化 vs 记忆

论文专门讨论了一个重要问题：GPT-2 的强大 zero-shot 表现是否只是因为它记住了训练数据？

### 数据重叠分析

作者使用 Bloom Filter 检查 WebText 训练集与各评估数据集测试集之间的 8-gram 重叠：

- 常见的语言模型数据集与 WebText 的重叠率为 **1-6%**，平均 3.2%
- 有些数据集与自身训练集的重叠更大（如 1BW 有 13.2%）
- 排除重叠样本后，结果变化很小（如 LAMBADA 准确率从 63.2% 降至 62.9%）

### 欠拟合证据

训练集和测试集的 perplexity 随模型变大**同步下降**，说明 GPT-2 仍在**欠拟合** WebText，性能提升来自于更好的泛化而非记忆。

## 总结与思考

GPT-2 相对于 GPT-1 的关键演进：

1. **从"预训练+微调"到"纯 zero-shot"**：GPT-1 的贡献在于证明预训练的价值，GPT-2 则更进一步证明了足够大的语言模型可以不微调就执行任务
2. **规模的力量**：从 117M 到 1.5B 参数，GPT-2 清晰地展示了 scaling law——性能随模型大小 log-linear 增长，且所有任务一致受益
3. **用自然语言做任务条件化**：不再需要 task-specific 的输入变换或输出头，用文本 prompt 就能指定任务，这一思想直接启发了后续的 prompt engineering 和 in-context learning
4. **数据质量与多样性**：WebText 的构建方式（Reddit karma 筛选）提供了一种低成本获取高质量多样数据的方案
5. **Byte-Level BPE**：解决了多语言和特殊字符的 tokenization 问题，成为后续模型的标配

GPT-2 是 GPT-1 到 GPT-3 之间的重要桥梁：GPT-1 证明了预训练有用，GPT-2 证明了更大的模型不微调也能做任务，GPT-3 则将这一思路推到了 175B 参数，并正式提出了 in-context learning 的概念。
