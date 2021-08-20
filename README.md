# 炮弹分割项目

这个项目用的网络结构是BiSeNetV1，backbone是resnet18；包含opencv,onnx，tensorrt（C++)部署代码。

在炮弹数据集中的miou map如下表：

| 图片分辨率 | map  | miou | link |
| :--------: | :--: | :--: | ---- |
| (1280,720) |      |      |      |
| (640,360)  |      |      |      |
| (320,180)  |      |      |      |

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



