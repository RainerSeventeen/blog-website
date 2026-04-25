---
paper: "Visual Instruction Tuning"
---

# LLaVA

> 论文全名：*Visual Instruction Tuning*
>
> LLaVA 是 BLIP2 之后的又一里程碑之作

## 背景

cv 领域的任务是各种标注分割等工作，大多数模型都是任务驱动的；相对应的  llm 领域的都是指令驱动型的全能模型

本文的目标就是希望能够实现一个能够拥有 llm 强大指令遵循能力的一个可以看图的模型

在这之前有过 blip2 等模型，但是他们都没有做过 Instruction Tuning [^1]

总体而言可以是以下三个挑战：

1. 多模态数据（caption, QA）都有一些，但是指令微调的数据几乎没有
2. 老生常谈，embedding 的空间不同，难对齐
3. 现有的指令只会看图说话，不会做指令

## 主要工作

之前的多模态模型不是能力不够，而是缺少 instruction 的相关训练，为此该文章主要做了这些工作：

1. 多模态指令微调数据生成：面向多模态的专用的指令微调数据生成，把图像文本对转化为一个指令对的形式
2. 一个完整的多模态大模型：使用 CLIP + LLM，然后使用指令数据执行微调。
3. 提出了一个多模态指令遵循能力的 benchmark

最后，文章内容和相关工作都是开源的

## GPT-assisted Visual Instruction Data Generation

参照原文中的例子，作者首先用常见的 cv 工作中的例如图像检测的工作，以及 caption 的相关数据，生成一个 prompt 给 gpt，如下方所示，是一个纯文本的信息

```
Context type 1: Captions
A group of people standing outside of a black vehicle with various luggage.
Luggage surrounds a vehicle in an underground parking area
People try to fit all of their luggage in an SUV.
The sport utility vehicle is parked in the public garage, being packed for a trip
Some people with luggage near a van that is transporting it.
Context type 2: Boxes
person: [0.681, 0.242, 0.774, 0.694], backpack: [0.384, 0.696, 0.485, 0.914], suitcase: ...<omitted>
```

喂给 GPT 之后可以生成一串指令数据集，例如

```
Response type 1: conversation
Question: What type of vehicle is featured in the image?
Answer: The image features a black sport utility vehicle (SUV) ...<omitted>
```

## Visual Instruction Tuning

现在我们已经有了完整数据，后续就是如何将 视觉模型和大语言模型 结合起来

### 架构

![LLaVA 架构](https://oss.rainerseventeen.cn/blog/2026/202604251246003.png)

使用 ViT 提取视觉特征得到 $Z_v = g(X_v)$， 然后直接用个简单的线性层映射到 llm 的语言维度上 $H_v = W \cdot Z_v$

在文章中使用了 Vicuna 作为 llm，是一个指令遵循能力相对较强的一个模型

这里其实就可以发现了，LLaVA 的架构是非常简单的，不像是 BLIP2 那样设计了一个 Q former 的架构，而是提出了一种用指令指导的视觉模型微调方式

### 训练

数据是图像和 对应的 QA 数据多轮对话，拼接出来就是这样的数据结构：

```
Xsystem-message <STOP>
Human: X1_instruct <STOP> Assistant: X1_a <STOP>
Human: X2_instruct <STOP> Assistant: X2_a <STOP>
...
```

其中对于 第一轮的处理相对特殊一点，第一轮图像和问题随机交换，提升一点鲁棒性，后续都是只输入 query 了
$$
\begin{array}{c} X_{instruct}^t =
\begin{cases}
\text{随机选择 } [X_q^1, X_v] \text{ 或 } [X_v, X_q^1], & t=1 \\
X_q^t, & t>1
\end{cases} \end{array}
$$
总体来说训练分成两步骤（可以对照 Q former 来看）

#### Pre-training for Feature Alignment

这是第一步，只训练 projection layer（也就是除了这个线性层 $W$ 全部都被冻结了），让线性层学会如何把图像特征翻译到 llm 的空间内

这里用的数据是最简单的数据，问题类型是“请简要描述图像” + 基础标注 caption 的步骤，也就是最最基础的图像理解

也是相对部分直觉的，先把最基础的桥接工作做好，才可以做后续的更复杂的任务

#### End-to-End Fine-tuning

第二步，让模型学习会指令跟随功能，冻结视觉编码器，将后面的 projection layer $W$ 和 llm 的参数加入训练流程

冻结 CLIP 的原因主要是，CLIP 本身的视觉表征是足够的，重点在于对齐语义以及对话的指引；这么做的另一好处就是省下了计算量，训练成本降低了

在这里给出了两个训练场景，

一个是 MultiModal Chatbot，就是拿 GPT 生成的那个数据，多轮对话数据。

一个是 Science QA 数据集，是另一个多模态的数据集，包含了思考流程（*Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering*），这里作者把他压缩成了单轮的对话。

## LLaVA Benchmark

LLaVA 的另一个卓越贡献就是提供了一个 benchmark

之前 COCO 这样的视觉数据集，但实际上只能评价模型对图像的理解能力，而很难评价多模态模型遵循指令的能力

### 数据来源

1. LLaVA-Bench (COCO)：用来测试分析能力，随机在 COCO 中选择了 30 张图像，每张图像生成了 3 类数据：conversation, detailed description, complex reasoning 一共 90 个 QA 
2. LLaVA-Bench (In-the-Wild)：用来测试泛化能力，24张人工标注的图片，有更加复杂的问题，涵盖了各种复杂的场景，测试模型在“非标准数据 + 高难任务”下的表现

### 测评方式

测评方式我认为才是更加核心的地方，作者提出使用 GPT 模型作为 judge

这里有一个非常巧妙的思想，就是我在做项目的时候也遇到了如果尝试使用 GPT 对某个内容执行打分的时候，每一轮的调用没有相对值，导致对比起来实际上效果很差

作者使用 Question + Description 在 Text only 的 GPT 上先生成一个答案，等于是让 GPT 的能力作为一个对照组，然后再通过 LLaVA 走多模态的路线，Question + Image 生成一个答案

最后在 helpfulness（是否有帮助）relevance（是否相关）accuracy（是否正确）detail（是否详细）这四个指标上让 GPT 执行打分，这个综合评分越高越好



[^1]: Instruction Tuning（指令微调）：是一种让大模型学会按照人类指令做事的训练方法，本质是对预训练模型进行监督微调（SFT, Supervised Fine-Tuning）。