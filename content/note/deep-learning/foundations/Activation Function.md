# Activation Function

一个简单的总表:

| 激活函数             | 常见用途                 | 优点         | 缺点               |
| ---------------- | -------------------- | ---------- | ---------------- |
| Sigmoid          | 二分类输出层               | 输出可解释为概率   | 梯度消失             |
| Tanh             | 早期 RNN               | 零中心        | 梯度消失             |
| ReLU             | CNN、MLP              | 简单高效       | 神经元死亡            |
| Leaky ReLU       | CNN、MLP              | 缓解 ReLU 死亡 | 负斜率需设置           |
| PReLU            | 视觉模型                 | 负斜率可学习     | 增加参数             |
| ELU              | 深层网络                 | 输出更平滑      | 计算较慢             |
| Softplus         | ReLU 平滑替代            | 可导         | 计算较慢             |
| GELU             | Transformer、BERT、ViT | 平滑，效果好     | 比 ReLU 慢         |
| SiLU/Swish       | CNN、现代网络             | 平滑，性能好     | 计算稍复杂            |
| Mish             | 视觉模型                 | 平滑         | 使用不如 GELU/SiLU 广 |
| Softmax          | 多分类输出                | 概率分布       | 通常不用作隐藏层         |
| GLU/GeGLU/SwiGLU | Transformer/LLM FFN  | 表达能力强      | 结构更复杂            |
