## 快速入门

> 以一个训练流程作为例子，讲解大体流程

按照 中文 Tutorial 文档中的示例流程

###  训练流程

####  代码入口及训练准备

```Bash
# 终端输入命令
python basicsr/train.py -opt options/train/SRResNet_SRGAN/train_MSRResNet_x4.yml
```

启动入口会做以下工作：

1. 获取路径并启动 pipeline （因为作为一个库使用，需要获取实际调用位置）
2. 解析测试配置 yml 文件
3. 如果配置了 resume 相关参数则会尝试恢复
4. 项目文件初始化：创建输出目录，复制配置文件，初始化 logger

####  dataloader

dataloader 主要处理在路径 `basicsr/data/__init__.py` 中实现

具体包括两个函数 `build_dataloader()` , `build_dataset()`

`build_dataset()` 根据 yml 中 `dataset` 类型 动态创建实例

`build_dataloader()` 根据 其他参数构建 dataloader

####  model

model 主要处理逻辑在 `basicsr/models/__init__.py` 中定义的 `build_model()` 实现

`build_model` 会根据配置文件 yml 中的 model 类型创建相应的实例

model 中主要会完成以下内容的构建：

1. network 构建：`build_network()`根据配置创建实例，并初始化训练相关设置
2. loss 构建：`build_loss()` 实现
3. optimize_parameters，即一个 iteration 下的 train step 。这个函数里面主要包含了 network forward ，loss 计算，backward 和优化器的更新
4. metric：主要是在 validation 里面，`calculate_metric()` 根据 yml 配置调用对应函数

####  Train 主函数

外层 for 循环 根据 epoch 索引进行，实际上根据 iteration 判定结束

训练包含以下流程：

1. 检查 epoch 是否还有数据，iteration 是否达到目标，进入训练轮次
2. 更新 Learning Rate，输入数据，并进行优化计算
3. 一轮后进行 log 保存信息
4. 每隔一段时间保存模型和训练状态
5. 每隔一段时间进行 validation

### 测试流程

指的是使用 `basicsr/test.py` 和配置文件 yml 来测试模型，以得到测试结果，同时输出指标结果的过程

测试很多部分和训练共用，不再赘述，下述为测试命令

```Bash
python basicsr/test.py -opt options/test/SRResNet_SRGAN/test_MSRResNet_x4.yml
```

主要流程包括：

1. 解析 yml 文件，加载配置参数
2. 新建 logger 并初始化，打印基础信息
3. 创建测试集和 dataloader 。和训练过程一样，调用 `build_dataset()` 和 `build_dataloader()`
4. 创建模型，和训练过程一样，调用 `build_model()`
5. 测试多个测试集，调用的是 model 里面的 validation 函数

### 推理流程

指的是，使用 inference 目录下的代码，快速方便地测试结果

推理部分相对独立，用到了很少的 Basic SR 的内容，便于部署在各类任务中

```Bash
python inference/inference_esrgan.py --input input_path --output out_path
```



## 配置文件

### 命名方式

```
001_MSRResNet_x4_f64b16_DIV2K_1000k_B16G1_wandb.yml
```

配置文件命名方式

001: 我们一般给实验进行数字打头的标号, 方便进行实验管理

• MSRResNet: 模型名称, 这里指代Modified SRResNet

• x4_f64b16: 重要配置参数, 这里表示放大4倍; 中间feature通道数是64, 使用了16个Residual

Block

• DIV2K: 训练数据集是DIV2K

• 1000k: 训练了1000K iterations

• B16G1: Batch size 为16, 使用一卡GPU 训练

• wandb: 使用了wandb, 训练过程上传到了wandb 云服务器

### Debug 模式

正式训练之前检查是否正常运行的方式

```bash
python basicsr/train.py -opt \
options/train/SRResNet_SRGAN/train_MSRResNet_x4.yml --debug
```

### Train 配置文件

#### 通用配置

```yml
# 实验名称, 若实验名字中有debug字样, 则会进入debug模式
name: 001_MSRResNet_x4_f64b16_DIV2K_1000k_B16G1_wandb 
model_type: SRModel # 使用的model 类型
scale: 4 # 输出比输入的倍数, 在SR中是放大倍数; 若有些任务没有这个配置, 则写1
num_gpu: 1 # 指定使用的GPU 卡数
manual_seed: 0 # 指定随机种子
```

#### Dataset 配置

```yaml
datasets:
  train: # 训练dataset 的配置
    name: DIV2K # 自定义的数据集名称
    type: PairedImageDataset # 读取数据的Dataset 类
    # 以下属性是灵活的, 可在相应类的说明文档中获得。新加的数据集可根据需要添加
    dataroot_gt: datasets/DF2K/DIV2K_train_HR_sub # GT (Ground-Truth) 图像
    ˓→ 的文件夹路径
    dataroot_lq: datasets/DF2K/DIV2K_train_LR_bicubic_X4_sub # LQ
    ˓→ (Low-Quality) 输入图像的文件夹路径
    meta_info_file: basicsr/data/meta_info/meta_info_DIV2K800sub_GT.txt # 预
    ˓→ 先生成的 meta_info 文件
    # (for lmdb)
    # dataroot_gt: datasets/DIV2K/DIV2K_train_HR_sub.lmdb
    # dataroot_lq: datasets/DIV2K/DIV2K_train_LR_bicubic_X4_sub.lmdb
    filename_tmpl: '{}' # 文件名字模板, 一般LQ文件会有类似'_x4' 这样的文件后
						# 缀, 这个就是来处理GT和LQ文件后缀不匹配的问题的
    io_backend: # IO 读取的backend
    type: disk # disk 表示直接从硬盘读取
    # (for lmdb)
    # type: lmdb
    gt_size: 128 # 训练阶段裁剪(crop) 的GT图像的尺寸大小，即训练的label 大小
    use_hflip: true # 是否开启水平方向图像增强(随机水平翻转图像)
    use_rot: true # 是否开启旋转图像增强(随机旋转图像)
    # data loader - 下面这块是data loader 的设置
    num_worker_per_gpu: 6 # 每一个GPU 的data loader 读取进程数目
    batch_size_per_gpu: 16 # 每块GPU 上的batch size
    dataset_enlarge_ratio: 100 	# 放大dataset 的长度倍数(默认为1)。可以扩大
    							# 一个 epoch 所需 iterations
    prefetch_mode: ~ # 预先读取数据的方式
    
  val: # validation 数据集的设置
	name: Set5 # 数据集名称
	type: PairedImageDataset # 数据集的类型
	# 以下属性是灵活的, 类似训练数据集
	dataroot_gt: datasets/Set5/GTmod12
	dataroot_lq: datasets/Set5/LRbicx4
	io_backend:
	type: disk

```

#### Network 配置

```yaml
# network structures - 网络结构的设置
network_g: # 网络g 的设置
  type: MSRResNet # 网络结构(Architecture) 的类型
  # 以下属性是灵活且特定的, 可在相应类的说明文档中获得
  num_in_ch: 3 # 模型输入的图像通道数
  num_out_ch: 3 # 模型输出的图像通道数
  num_feat: 64 # 模型内部的feature map 通道数
  num_block: 16 # 模型内部基础模块的堆叠数
  upscale: 4 # 上采样倍数
```



#### Train 配置

```yaml
# training settings
train: # 这块是训练策略相关的配置
  ema_decay: 0.999 # EMA 更新权重
  optim_g: # 这块是优化器的配置
  type: Adam # 选择优化器类型，例如Adam
  # 以下属性是灵活的, 根据不同优化器有不同的设置
  lr: !!float 2e-4 # 初始学习率
  weight_decay: 0 # 权重衰退参数
  betas: [0.9, 0.99] # Adam 优化器的beta1 和beta2
  scheduler: # 这块是学习率调度器的配置
  type: CosineAnnealingRestartLR # 选择学习率更新策略
  # 以下属性是灵活的, 根据学习率Scheduler 的不同有不同的设置
  periods: [250000, 250000, 250000, 250000] # Cosine Annealing 的更新周期
  restart_weights: [1, 1, 1, 1] # Cosine Annealing 每次Restart 的权重
  eta_min: !!float 1e-7 # 学习率衰退到的最小值
  total_iter: 1000000 # 总共进行的训练迭代次数
  warmup_iter: -1 # warm up 的迭代次数, 如是-1, 表示没有warm up
  # losses - 这块是损失函数的设置
  pixel_opt: # loss 名字，这里表示pixel-wise loss 的options
  type: L1Loss # 选择loss 函数，例如L1Loss
  # 以下属性是灵活的, 根据不同损失函数有不同的设置
  loss_weight: 1.0 # 指定loss 的权重
  reduction: mean # loss reduction 方式
```



#### Validation 配置

```yaml
# validation settings
val: # 这块是validation 的配置
  val_freq: !!float 5e3 # validation 频率, 每隔5000 iterations 做一次
  ˓→ validation
  save_img: false # 否需要在validation 的时候保存图片
  metrics: # 这块是validation 中使用的指标的配置
  psnr: # metric 名字, 这个名字可以是任意的
  type: calculate_psnr # 选择指标类型
  # 以下属性是灵活的, 根据不同metric 有不同的设置
  crop_border: 4 # 计算指标时crop 图像边界像素范围(不纳入计算范围)
  test_y_channel: false # 是否转成在Y(CbCr) 空间上计算
  better: higher # 该指标是越高越好，还是越低越好。选择higher 或者
  				 # lower，默认为 higher
  niqe: # 这是在validation 中使用的另外一个指标
  type: calculate_niqe
  crop_border: 4
  better: lower # the lower, the better
```

#### Logger 设置

```yaml
# logging settings
logger: # 这块是logging 的配置
  print_freq: 100 # 多少次迭代打印一次训练信息
  save_checkpoint_freq: !!float 5e3 # 多少次迭代保存一次模型权重和训练状态
  use_tb_logger: true # 是否使用tensorboard logger
  wandb: # 是否使用wandb logger
  project: ~ # wandb 的project名字。默认是None, 即不使用wandb
  resume_id: ~ # 如果是resume, 可以输入上次的wandb id, 则log 可以接起来
```

### Test 配置文件

test 的配置和 train 大致相同，所以重点讲解一些不同的即可

`val` 就是 train 中的 val，具体逻辑需要参考自定义的 dataset 类的设置

pretrain 就是自己需要测试的模型



### 处理函数

basicsr 内置了很多的函数，可以直接用来处理数据

具体可以参见 [API 文档](https://basicsr.readthedocs.io/en/latest/)

选择 `basics.utils` 查看操作函数