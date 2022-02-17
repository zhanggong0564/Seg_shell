# 炮弹分割项目

网络结构是resnet-unet，backbone是resnet18

## loss 函数：交叉熵

unet：0.556

Unet++:0.60

deeplabv3+:0.5716

BiseNet:0.769

## focal_loss

BiseNet:0.816

## lovasz_softmax

​	0.828  class_acc:0.912

在炮弹数据集中的miou map如下表：

| 图片分辨率 | class_acc | miou  | link |
| :--------: | :-------: | :---: | ---- |
| (1280,720) |     x     |   x   | x    |
| (640,368)  |    0.94   | 0.89  |      |
| (320,176)  |           |       |      |





这个项目用的网络结构是BiSeNetV1，backbone是resnet18；包含opencv,onnx，tensorrt（C++)部署代码。


C++部署时间测试：

| 部署方式 | time | link |
| -------- | ---- | ---- |
| opencv   |      |      |
| onnx     |      |      |
| tensorrt |      |      |

## deploy trained models

代码详情见[deploy](./CPP)

## platform

* ubnuntu18.04
* TX2
* cuda10.2
* cudnn8.0
* miniconda python 3.6.9
* pytorch 1.4

## get  start

用训练好的权重文件.pt,可以进行python版本的推理

```
python demo.py
```

## train

```
python train.py
```

## export onnx

```
cd tool
python export_onnx.py
```



