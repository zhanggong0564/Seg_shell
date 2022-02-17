# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from model.backbone.resnet import Resnet






class BasicBlock(nn.Module):
    expansion =1
    def __init__(self,in_chans,out_chans,stride =1):
        super(BasicBlock, self).__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_chans,out_chans,kernel_size=3,stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chans,out_chans*BasicBlock.expansion,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(out_chans*BasicBlock.expansion)
        )
        self.shortcut = nn.Sequential()
        if stride!=1 or in_chans!=BasicBlock.expansion*out_chans:
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_chans,out_chans*BasicBlock.expansion,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_chans*BasicBlock.expansion)
            )
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        return self.relu(self.residual_function(x)+self.shortcut(x))

class UNetUpBlock(nn.Module):
    def __init__(self,in_chans,bridge_chans,out_chans,up_mode='upconv'):
        super(UNetUpBlock, self).__init__()
        if up_mode=='upconv':
            self.up = nn.ConvTranspose2d(in_chans,out_chans,kernel_size=2,stride=2)
        elif up_mode=='upsample':
            self.up  = nn.Sequential(
                nn.Upsample(mode='bilinear',scale_factor=2),
                nn.Conv2d(in_chans,out_chans,kernel_size=1)
            )
        self.conv_block = BasicBlock(out_chans+bridge_chans,out_chans)
    def forward(self,x,bridge):
        up = self.up(x)
        crop = self.center_crop(bridge,up.shape[2:])
        out = torch.cat([crop, up], dim=1)
        out = self.conv_block(out)
        return out
    def center_crop(self,layer,target_size):
        _, _, layer_height, layer_width = layer.size()
        offset_y = (layer_height-target_size[0])//2
        offset_x = (layer_width-target_size[1])//2
        return layer[:,:,offset_y:(offset_y+target_size[0]),offset_x:(offset_x+target_size[1])]



class UNet(nn.Module):
    def __init__(self,n_classes=4,backbone = 'resnet18',up_mode='upconv'):
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        assert backbone in ('resnet18','resnet34','resnet50')
        self.enchoder = Resnet(backbone)

        in_chans = 512*self.enchoder.expansion

        self.decoder = nn.ModuleList()
        for i in range(3):
            self.decoder.append(UNetUpBlock(in_chans,in_chans//2,in_chans//2,up_mode))
            in_chans//=2
        self.decoder.append(UNetUpBlock(in_chans,64,64,up_mode))

        self.cls_conv = nn.Conv2d(64, n_classes, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self,x):
        x1,x2,x3,x4,x5 = self.enchoder(x)
        bridges = [x1,x2,x3,x4]
        x = x5
        for i,decode_layer in enumerate(self.decoder):
            x = decode_layer(x,bridges[-i-1])
        score = self.cls_conv(x)
        return score
if __name__ == '__main__':
    print(torch.cuda.is_available())
    for i in range(350,370):
        x = torch.randn((1, 3, i, 640), dtype=torch.float32).cuda()
        unet = torch.nn.DataParallel(UNet(4)).cuda()
        # y = unet(x)
        try:
            y = unet(x)
        except:
            continue
        print(x.shape)
        print(y.shape)






