from torchvision import models
import torch
from torch import nn
from torchvision.models import shufflenetv2



class resnet18(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.features = models.resnet18(pretrained=pretrained)
        self.conv1 = self.features.conv1
        self.bn1 = self.features.bn1
        self.relu = self.features.relu
        self.maxpool1 = self.features.maxpool
        self.layer1 = self.features.layer1
        self.layer2 = self.features.layer2
        self.layer3 = self.features.layer3
        self.layer4 = self.features.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(self.bn1(x))
        x = self.maxpool1(x)
        feature1 = self.layer1(x)  # 1 / 4
        feature2 = self.layer2(feature1)  # 1 / 8
        feature3 = self.layer3(feature2)  # 1 / 16
        feature4 = self.layer4(feature3)  # 1 / 32
        # global average pooling to build tail
        tail = self.avgpool(feature4)
        return feature3, feature4, tail
class shuffenet(nn.Module):
    def __init__(self,pretrained):
        super(shuffenet, self).__init__()
        self.features =shufflenetv2.shufflenet_v2_x0_5(pretrained=pretrained)
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



class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2,padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))


class Spatial_path(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convblock1 = ConvBlock(in_channels=3, out_channels=64)
        self.convblock2 = ConvBlock(in_channels=64, out_channels=128)
        self.convblock3 = ConvBlock(in_channels=128, out_channels=256)

    def forward(self, input):
        x = self.convblock1(input)
        x = self.convblock2(x)
        x = self.convblock3(x)
        return x


class AttentionRefinementModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, input):
        # global average pooling
        x = self.avgpool(input)
        assert self.in_channels == x.size(1), 'in_channels and out_channels should all be {}'.format(x.size(1))
        x = self.conv(x)
        x = self.sigmoid(x)
        x = torch.mul(input, x)
        return x


class FeatureFusionModule(torch.nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.convblock = ConvBlock(in_channels=self.in_channels, out_channels=num_classes, stride=1)
        self.conv1 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, input_1, input_2):
        x = torch.cat((input_1, input_2), dim=1)
        assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.convblock(x)
        x = self.avgpool(feature)

        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = torch.mul(feature, x)
        x = torch.add(x, feature)
        return x


class BiSeNet(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.saptial_path = Spatial_path()
        # self.context_path = resnet18(pretrained=True)
        self.context_path = shuffenet(pretrained=True)

        self.attention_refinement_module1 = AttentionRefinementModule(96, 96)
        self.attention_refinement_module2 = AttentionRefinementModule(192, 192)

        self.supervision1 = nn.Conv2d(in_channels=96, out_channels=num_classes, kernel_size=1)
        self.supervision2 = nn.Conv2d(in_channels=192, out_channels=num_classes, kernel_size=1)

        self.feature_fusion_module = FeatureFusionModule(num_classes, 544)

        self.conv = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1)

    def forward(self, input):
        sp = self.saptial_path(input)
        cx1, cx2, tail = self.context_path(input)#1x256x28x28 1x512x28x28 1x512x1x1
        #96 192 192
        cx1 = self.attention_refinement_module1(cx1)
        cx2 = self.attention_refinement_module2(cx2)
        cx2 = torch.mul(cx2, tail)

        cx1 = torch.nn.functional.interpolate(cx1, size=sp.size()[-2:], mode='bilinear')
        cx2 = torch.nn.functional.interpolate(cx2, size=sp.size()[-2:], mode='bilinear')
        cx = torch.cat((cx1, cx2), dim=1)

        if self.training == True:
            cx1_sup = self.supervision1(cx1)
            cx2_sup = self.supervision2(cx2)
            cx1_sup = torch.nn.functional.interpolate(cx1_sup, size=input.size()[-2:], mode='bilinear')
            cx2_sup = torch.nn.functional.interpolate(cx2_sup, size=input.size()[-2:], mode='bilinear')

        result = self.feature_fusion_module(sp, cx)

        result = torch.nn.functional.interpolate(result, size=input.size()[-2:], mode='bilinear')

        result = self.conv(result)

        if self.training == True:
            return result, cx1_sup, cx2_sup
        #转模型使用
        # result = torch.softmax(result,dim=1);
        # result = result.squeeze().permute(1,2,0)
        # result = result.view(-1,3)
        return result


if __name__ == '__main__':
    import torch as t

    rgb = t.randn(1, 3, 448, 448)

    net = BiSeNet(3)

    out ,cx1_sup,cx2_sup= net(rgb)

    print(out.shape)
    print(cx1_sup.shape)
    print(cx2_sup.shape)