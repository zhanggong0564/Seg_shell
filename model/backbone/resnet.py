import numpy as np
import torch
from torchvision.models import resnet50,resnet34,resnet18
import torch.nn as nn

class Resnet(nn.Module):
    def __init__(self,backbone):
        super(Resnet, self).__init__()
        assert backbone in ['resnet18','resnet34','resnet50']
        model = self.get_model(backbone)
        self.conv1 = model.conv1
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
        x =self.maxpool(self.relu(x))
        x =self.layer1(x)
        x3 = self.layer2(x)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return x3,x4,x5
    def get_model(self,backbone):
        if backbone=='resnet50':
            model  = resnet50(pretrained=True)
        elif backbone=='resnet34':
            model = resnet34(pretrained=True)
        else:
            model = resnet18(pretrained=True)
        return model

if __name__ == '__main__':
    x = torch.randn((4, 3, 224, 224))
    resnet = Resnet('resnet18')
    import time
    start = time.time()
    for i in range(100):
        y = resnet(x)
    end = time.time()
    print(end-start)#4.68