from torchvision.models import resnet18,mobilenet_v2,shufflenetv2
from  torchvision.models.quantization.resnet import resnet18
from  torchvision.models import mobilenet
import torch
import torch.nn as nn

class shuffenet(nn.Module):
    def __init__(self):
        super(shuffenet, self).__init__()
        self.features =shufflenetv2.shufflenet_v2_x0_5()
        self.conv1 = self.features.conv1
        self.maxpool = self.features.maxpool
        self.stage2 = self.features.stage2
        self.stage3 = self.features.stage3
        self.stage4= self.features.stage4
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x1 = self.stage2(x)
        feature3= self.stage3(x1)
        feature4 = self.stage4(feature3)
        tail = self.avgpool(feature4)
        return feature3,feature4,tail




if __name__ == '__main__':
    x = torch.randn((4,3,224,224))
    model = resnet18()
    import time
    start = time.time()
    for i in range(20):
        y = model(x)
    end = time.time()
    print(end-start)#7.77
