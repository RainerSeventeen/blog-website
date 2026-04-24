# Loss Function

## Cross Entropy

在 torch API 里面，使用 `torch.nn.functional.crosss_entropy(input, target)` 可以直接算，已经自动执行了 Softmax

- input：模型输出的 **原始 logits**，维度是 `(N, C)` 也就是样本数以及所有类别的个数对应的预测分数
- target：真实标签， 维度就是 `(N, )`，也就是样本期望的那个类别

