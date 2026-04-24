---
paper: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
---

# BERT

> 论文全名：*BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*

## 简介

BERT 这篇文章在 GPT-1 之后诞生的，主要解决了设计一个可以双向利用上下文的通用预训练模型

所谓双向上下文指的是传统的 Mask Self Attention 的时候是在根据前文预测后文。在自然语言推断，问答等场景中，判断一个词是不是答案的时候需要同时看前后文。

但是我们都知道直接双向看，那么训练模型就会发生偷看答案的问题了。

## 架构

BERT 的模型架构是双向的 Transformer Encoder 架构，由多个 Block 堆叠出来的。

这里的双向指的是在做 SA 的时候模型能够看到左右的 token。

文章中将多种任务都包装成统一的表示

输入由三部分组成：Token Embedding, Segment Embedding, Position Embedding

使用一些特殊 token：

- `[CLS]` 整个序列的最开头，作为序列的聚合表示
- `[SEP]`分割句子作用
- Segment A/B embedding 用于标识这个句子是 A 还是 B 的作用，用在句子匹配中

例如对于问答问题就是：

```
[CLS] question [SEP] passage [SEP]
```

## 预训练

论文中主要提出了两个任务： Masked LM 与 Next Sentence Prediction 任务

### Masked LM

在之前的训练方式中需要使用 causal mask，让模型不能偷看答案，实现代码就是一个上三角的矩阵

作者提出一种全新的方法，在不破坏模型因果的情况下，又能训练模型的前后文关联能力

在预训练任务中，随机选择 15% 的 token 作为预测目标，对被选中的位置中：

**80%** 替换成 `[MASK]`，**10%** 替换成随机词，**10%** 保持不变，让模型来预测这个地方原来的单词是什么

没有全部都换成 `[MASK]`就是因为下游微调等是没有这个 token 的，避免过度依赖这个信号

这个最主要的作用是，模型能够获得上下文关系的能力，而不是单纯进行下一个词语的预测

### NSP

构造句子对，让模型尝试理解两个句子之间的关系。

给定句子对 A/B，有 50% 的 B 是 A 的下一句，另外就是随机的句子，需要让模型学习判断是否是下一句

提升的是模型**句子**级别的关系能力。

## 微调

微调基本上就是换了个任务头，也就是在下游的输出中，额外增加一个用于输出目标任务数据的 head

例如对于文本情感分类任务，得到每个位置的 hidden state 的序列，接一个线性层再 softmax 就输出分类，这里的线性层就是任务头

总共有 3 类任务：

1. 文本分类 / 句子对分类任务
2. 序列标注，对每一个 token 加分类器
3. 引入一个 start 向量 S 和 end 向量 E，分别对每个 token 打分，预测哪一个 token 是不是答案的起点终点。
