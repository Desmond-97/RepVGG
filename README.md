# RepVGG_quantization(PyTorch)

首先感谢丁霄汉博士开源的RepVGG代码~该repo仅用作记录量化RepVGG模型到int8所做的尝试。

起因是尝试量化RepVGG到INT8来验证RepVGG的量化效果，发现直接按照原repo提供的qat量化pipeline会得到差强人意的量化模型。量化后的INT8模型在ImageNet上掉点严重（大于10个点）。

原pipeline：

The best solution for quantization is to constrain the equivalent kernel (get_equivalent_kernel_bias() in repvgg.py) to be low-bit (e.g., make every param in {-127, -126, .., 126, 127} for int8), instead of constraining the params of every kernel separately for an ordinary model.

For the simplicity, we can also use the off-the-shelf quantization toolboxes to quantize RepVGG. We use the simple QAT (quantization-aware training) tool in torch.quantization as an example.

1. The base model is trained with the custom weight decay (```--custwd```) and converted into inference-time structure. We insert BN after the converted 3x3 conv layers because QAT with torch.quantization requires BN. Specifically, we run the model on ImageNet training set and record the mean/std statistics and use them to initialize the BN layers, and initialize BN.gamma/beta accordingly so that the saved model has the same outputs as the inference-time model. 

```
python train.py -a RepVGG-A0 --dist-url 'tcp://127.0.0.1:23333' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --workers 32 [imagenet-folder] --tag hello --custwd
python convert.py RepVGG-A0_hello_best.pth.tar RepVGG-A0_base.pth -a RepVGG-A0 
python insert_bn.py [imagenet-folder] RepVGG-A0_base.pth RepVGG-A0_withBN.pth -a RepVGG-A0 -b 32 -n 40000
```

2. Build the model, prepare it for QAT (torch.quantization.prepare_qat), and conduct QAT. The hyper-parameters may not be optimal and I am tuning them.

```
python quantization/quant_qat_train.py [imagenet-folder] -j 32 --epochs 20 -b 256 --lr 1e-3 --weight-decay 4e-5 --base-weights RepVGG-A0_withBN.pth --tag quanttest
```

我认为是这里插入的BN层导致最终INT8模型效果变差，于是修改了./quantization/repvgg_quantized.py的L50行，将融合模块中的'bn', 'relu'去掉。

这样就可以跳过插入BN的操作，直接运行quantization/quant_qat_train.py即可。

在多次尝试后得到较优的参数如下：（现pipeline）

```
python quantization/quant_qat_train.py [imagenet-folder] -j 32 --epochs 20 -b 256 --lr 1e-3 --weight-decay 55e-6 --base-weights RepVGG-A0_withBN.pth --tag quanttest
```

-b（batchsize）尽量与训练时保持一致，lr为训练时的千分之一。

这样QAT过后的INT8模型，在ImageNet掉点1.7%。（虽然掉点还是比较多，但比原pipeline好很多）



### 尝试使用TensorRT量化：

**先说结论：初步测试发现对于RepVGG模型来说，PTQ量化比QAT量化方法效果更好一些。**

**在小模型RepVGG_A0上的测试结果：**

1. PTQ量化方法：ImageNet分类任务最少掉点仅**0.16%**（“percentil”：-0.41% “mse”：-0.98% **“ce”：-0.16%**）；
2. QAT量化方法：ImageNet分类任务掉点>3%（没有仔细调参）；

#### 记录TensorRT量化工具的安装步骤和代码实现：

测试环境：Ubuntu 18.04 cuda11.2 cudnn8.0 TensorRT:8.2.3.0

##### 安装：TensorRT在8.0版本更新支持QAT，旧版本不支持QAT。

安装包从Nvidia官网下载TAR package，下图中红框。

![img](https://horizonrobotics.feishu.cn/space/api/box/stream/download/asynccode/?code=YmQ0NWJmNTk4ZmJiNTU4YTQyNTlmMDBiZDM5NzEyODdfM0FxUm9Tc0FmcXNRV2NnWG9HZElXTlFTSEtjblNhN21fVG9rZW46Ym94Y243UE5Za2EzMExzcjhHSXlZQmRYVTlQXzE2NDYwMzAwODc6MTY0NjAzMzY4N19WNA)

解压缩后，cd到TensorRT-8.2.3.0/python下，里面有不同python版本的wheel文件，选择对应本地python版本的wheel文件pip install即可。

#### TensorRT量化测试：

```
#安装pytorch-quantization包
pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com

#修改quant_repvgg.py中的数据集路径 data_path = "path/to/your/imagenet"
vim quant_repvgg.py

#进行量化测试
python quant_repvgg.py
```

