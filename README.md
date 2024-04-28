

# 背景

基于nanoGPT项目测试Pytorch 2.0的scaled_dot_product_attention新算子和compile模型编译新特性。



# 测试环境

+ Ubuntu 22.04
+ CPU Intel(R) Core(TM) i5-13600K
+ 显卡RTX 4070
+ CUDA 12.2
+ Python 3.10.12
+ Pytorch 2.1.0

# 基础超参数

## 模型结构

测试选用的是nanoGPT中最小的版本并对模型结构进行了相应的修改，以便能够在4070的12G显存上正常训练，仅用于验证相应特性在消费级的4070上的加速效果，不关注模型的精度。

| n_layer | n_head | n_embd | bias | total param |
| ------- | ------ | ------ | ---- | ----------- |
| 12      | 12     | 768    | No   | 85.00M      |

## 训练策略

| 数据集                              | 迭代次数 | 优化器 | 梯度累积步数 | 权重衰减 | bs   | seq len |
| ----------------------------------- | -------- | ------ | ------------ | -------- | ---- | ------- |
| shakespeare_char (1,003,854 tokens) | 500      | AdamW  | 5            | 1e-2     | 6    | 1024    |

这些策略主要是简单模真实训练用到的相关工作流程。



# 使用指令

## 安装依赖

```bash
pip install -r requirements.txt
```

## 测试命令

```bash
python nanonano_train.py
```

通过修改**hyper_param.json**的**compile**和**attn_type**字段来测试不同的训练选项。**compile**控制是否启用torch.compile编译模型，**attn_type**字段可选naive,math,flash_attn和efficient_attn四个选项，分别代表朴素的Attention实现，MATH，FLASH_ATTENTION，EFFICIENT_ATTENTION。

# 测试结果

每次测试分别跑500iter,batch size=6,seq len=1024

## 测试平台1

**不同Attention结果**

分别是手写的Attention操作，scaled_dot_product_attention中的SDPBackend.MATH，SDPBackend.FLASH_ATTENTION和SDPBackend.EFFICIENT_ATTENTION这四种Attention实现的速度差异对比结果。

| Attn type           | ms/iter | RAM (MB) |
| ------------------- | :-----: | :------: |
| Naive               |  1361   | 9834MiB  |
| MATH                |   779   | 5862MiB  |
| FLASH_ATTENTION     |   475   | 3896MiB  |
| EFFICIENT_ATTENTION |   485   | 3912MiB  |

**torch.compile结果**

对应四种Attention使用torch.compile的测试结果

| Attn type           | compile | compile time | 1st iter | avg speed(ms/iter) | avg speed (except 1st iter) | RAM (MB) |
| ------------------- | ------- | :----------: | :------: | :----------------: | :-------------------------: | :------: |
| Naive               | No      |      -       |          |        1361        |            1360             | 9834MiB  |
| Naive               | Yes     |     534      |  10808   |        710         |             690             | 9360MiB  |
| MATH                | No      |      -       |   984    |        779         |             779             | 5862MiB  |
| MATH                | Yes     |     509      |  10278   |        812         |             793             | 5762MiB  |
| FLASH_ATTENTION     | No      |      -       |   694    |        475         |             475             | 3896MiB  |
| FLASH_ATTENTION     | Yes     |     527      |   9939   |        509         |             490             | 3730MiB  |
| EFFICIENT_ATTENTION | No      |      -       |   698    |        485         |             485             | 3912MiB  |
| EFFICIENT_ATTENTION | Yes     |     533      |  10038   |        519         |             500             | 3768MiB  |

### 测试平台2

新增在AMD EPYC 7543 32-Core, A100-40G,CUDA 11.6平台测试结果

**不同Attention结果**

| Attn type           | ms/iter | RAM (MB) |
| ------------------- | :-----: | :------: |
| Naive               |   507   | 10982MiB |
| MATH                |   292   | 7010MiB  |
| FLASH_ATTENTION     |   172   | 5044MiB  |
| EFFICIENT_ATTENTION |   210   | 5060MiB  |

**torch.compile结果**

| Attn type           | compile | compile time | 1st iter | avg speed(ms/iter) | avg speed (except 1st iter) | RAM (MB) |
| ------------------- | ------- | :----------: | :------: | :----------------: | :-------------------------: | :------: |
| Naive               | No      |      -       |   1147   |        507         |             505             | 10982MiB |
| Naive               | Yes     |     293      |  12891   |        270         |             245             | 10528MiB |
| MATH                | No      |      -       |   959    |        292         |             290             | 7010MiB  |
| MATH                | Yes     |     297      |   9214   |        310         |             292             | 6910MiB  |
| FLASH_ATTENTION     | No      |      -       |   848    |        172         |             171             | 5044MiB  |
| FLASH_ATTENTION     | Yes     |     301      |   8793   |        205         |             188             | 4878MiB  |
| EFFICIENT_ATTENTION | No      |      -       |   2617   |        210         |             205             | 5060MiB  |
| EFFICIENT_ATTENTION | Yes     |     302      |   8910   |        237         |             219             | 4932MiB  |

以上所有时间均以毫秒为单位。

从上述数据中可以看出两个现象：

1. 在我的机器上，相对于**朴素Attention**实现，scaled_dot_product_attention算子的三种实现均能达到明显的加速效果以及明显节省显存，其中flash attention实现效果最好，加速最高达到**65%**，显存占用也最小，memory efficient attention稍差，math attention实现提升效果最小。
2. torch.compile在朴素实现上速度提升明显，提速约47%(1361->710)，显存占用也减小了接近500M。但在Pytorch自带的math,flash,memeory efficient attention实现上速度均有不同程度变慢，显存占用略有减小。
3. torch.compile即使相对于朴素实现提升较明显，但效果仍然比不过flash attention和memory efficient attention。

从上可以得出初步结论：

1. Pytorch的scaled_dot_product_attention一般来说总会比我们手动实现的Attention算子更加高效，flash attention实现可以无脑用。
2. torch.compile对于朴素的模型实现有较好的加速效果，但仍然比不过高度优化的算子实现，比较适用于自定义算子或者操作比较多的模型加速。

# Note

由于手头上目前只有一张RTX 4070，故只有以上这些数据，因为以上结论不一定准确，可能存在不合理性，仅供参考。另外由于时间限制，当前的结果只是一个比较粗略的GPT模型的结果，测试并没有充分地利用GPU的最大算力。如果大家感兴趣可以试一试在不同计算平台和不同模型的效果，才能全面地评价Pytorch2.0新特性的训练性能。
