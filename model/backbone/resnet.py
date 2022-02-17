import numpy as np
import torch
from torchvision.models import resnet50,resnet34,resnet18
import torch.nn as nn

class Resnet(nn.Module):
    def __init__(self,backbone):
        super(Resnet, self).__init__()
        assert backbone in ['resnet18','resnet34','resnet50']
        if backbone=='resnet18' or 'resnet34':
            self.expansion =1
        else:
            self.expansion =4
        model = self.get_model(backbone)
        self.inplanes =64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = model.bn1
        self.relu  = model.relu
        self.maxpool = model.maxpool


        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.relu(x)
        x =self.maxpool(x1)
        x2 =self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return x1,x2,x3,x4,x5
    def get_model(self,backbone):
        if backbone=='resnet50':
            model  = resnet50(pretrained=True)
        elif backbone=='resnet34':
            model = resnet34(pretrained=True)
        else:
            model = resnet18(pretrained=True)
        return model

if __name__ == '__main__':
    x = torch.randn((4, 3, 224, 224)).cuda()
    resnet = Resnet('resnet18').cuda()
    import time
    start = time.time()
    for i in range(100):
        y = resnet(x)
    end = time.time()
    print(end-start)#4.68