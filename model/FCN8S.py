import numpy as np
import torch
from torchvision.models import resnet50
import torch.nn as nn
from model.backbone.resnet import Resnet
from model.backbone.efnet import Efnet

def bilinear_kernel(in_channels, out_channels, kernel_size):
    """Define a bilinear kernel according to in channels and out channels.
    Returns:
        return a bilinear filter tensor
    """
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    bilinear_filter = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float32)
    # weight[range(in_channels), range(out_channels), :, :] = bilinear_filter
    for w in range(weight.shape[0]):
        for h in range(weight.shape[1]):
            weight[w, h, :, :] = bilinear_filter
    return torch.from_numpy(weight)

class FCN8S(nn.Module):
    def __init__(self,backbone,num_classes):
        super(FCN8S, self).__init__()
        # self.features = Resnet(backbone)
        self.features = Efnet()
        self.conv3 = nn.Conv2d(112,40,1)
        self.conv2 = nn.Conv2d(40,num_classes,1)

        self.upsample_2x_1 = nn.ConvTranspose2d(1280,112,4,2,1,bias=False)
        self.upsample_2x_1.weight.data = bilinear_kernel(1280,112,4)

        self.upsample_2x_2 = nn.ConvTranspose2d(40,40,4,2,1,bias=False)
        self.upsample_2x_2.weight.data = bilinear_kernel(40,40,4)

        self.upsample_8x_1 = nn.ConvTranspose2d(num_classes,num_classes,16,8,4,bias=False)
        self.upsample_8x_1.weight.data = bilinear_kernel(num_classes,num_classes,16)


    def forward(self,x):
        x3,x4,x5 = self.features(x)
        #x5 2倍上采样
        x5 = self.upsample_2x_1(x5)
        add = x4+x5
        add = self.conv3(add)
        add = self.upsample_2x_2(add)

        add2 = add+x3

        add2 = self.conv2(add2)

        output = self.upsample_8x_1(add2)
        return  output
if __name__ == '__main__':
    x = torch.randn((4, 3, 224, 224))
    model =FCN8S('resnet18',3)
    y = model(x)
    print(y.shape)

    '''
    torch.Size([4, 512, 28, 28])
    torch.Size([4, 1024, 14, 14])
    torch.Size([4, 2048, 7, 7])
    '''
