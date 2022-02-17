import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
}


def conv3x3(in_planes, out_planes, stride=1, atrous=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1 * atrous, dilation=atrous, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_chans, out_chans, stride=1, atrous=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_chans, out_chans, stride, atrous)
        self.bn1 = nn.BatchNorm2d(out_chans)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_chans, out_chans)
        self.bn2 = nn.BatchNorm2d(out_chans)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_chans, out_chans, stride=1, atrous=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=stride,
                               padding=1 * atrous, dilation=atrous, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_chans * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class ResNet_Atrous(nn.Module):

    def __init__(self, block, layers, atrous=None, os=16):
        super(ResNet_Atrous, self).__init__()
        self.block = block
        stride_list = None
        if os == 8:
            stride_list = [2, 1, 1]
        elif os == 16:
            stride_list = [2, 2, 1]
        else:
            raise ValueError('resnet_atrous.py: output stride=%d is not supported.' % os)

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, 64, layers[0])
        self.layer2 = self._make_layer(block, 64 * block.expansion, 128, layers[1], stride=stride_list[0])
        self.layer3 = self._make_layer(block, 128 * block.expansion, 256, layers[2], stride=stride_list[1], atrous=16 // os)
        self.layer4 = self._make_layer(block, 256 * block.expansion, 512, layers[3], stride=stride_list[2],
                                       atrous=[item * 16 // os for item in atrous])
        self.layer5 = self._make_layer(block, 512 * block.expansion, 512, layers[3], stride=1, atrous=[item*16//os for item in atrous])
        self.layer6 = self._make_layer(block, 512 * block.expansion, 512, layers[3], stride=1, atrous=[item*16//os for item in atrous])
        self.layers = []

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, in_chans, out_chans, blocks, stride=1, atrous=None):
        downsample = None
        if atrous == None:
            atrous = [1] * blocks
        elif isinstance(atrous, int):
            atrous_list = [atrous] * blocks
            atrous = atrous_list
        if stride != 1 or in_chans != out_chans * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_chans, out_chans * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chans * block.expansion),
            )

        layers = []
        layers.append(block(in_chans, out_chans, stride=stride, atrous=atrous[0], downsample=downsample))
        in_chans = out_chans * block.expansion
        for i in range(1, blocks):
            layers.append(block(in_chans, out_chans, stride=1, atrous=atrous[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        layers_list = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        layers_list.append(x)
        x = self.layer2(x)
        layers_list.append(x)
        x = self.layer3(x)
        layers_list.append(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        layers_list.append(x)

        return layers_list
def resnet18_atrous(pretrained=True, os=16, **kwargs):
    model = ResNet_Atrous(BasicBlock,[2,2,2,2],atrous=[1,2,1],os=os,**kwargs)
    if pretrained:
        old_dict = model_zoo.load_url(model_urls['resnet18'])
        model_dict = model.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model

def resnet34_atrous(pretrained=True, os=16, **kwargs):
    """Constructs a atrous ResNet-34 model."""
    model = ResNet_Atrous(BasicBlock, [3, 4, 6, 3], atrous=[1, 2, 1], os=os, **kwargs)
    if pretrained:
        old_dict = model_zoo.load_url(model_urls['resnet34'])
        model_dict = model.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model


def resnet50_atrous(pretrained=True, os=16, **kwargs):
    """Constructs a atrous ResNet-50 model."""
    model = ResNet_Atrous(Bottleneck, [3, 4, 6, 3], atrous=[1, 2, 1], os=os, **kwargs)
    if pretrained:
        old_dict = model_zoo.load_url(model_urls['resnet50'])
        model_dict = model.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model


def resnet101_atrous(pretrained=True, os=16, **kwargs):
    """Constructs a atrous ResNet-101 model."""
    model = ResNet_Atrous(Bottleneck, [3, 4, 23, 3], atrous=[1, 2, 1], os=os, **kwargs)
    if pretrained:
        old_dict = model_zoo.load_url(model_urls['resnet101'])
        model_dict = model.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model


class ASPP(nn.Module):

    def __init__(self, in_chans, out_chans, rate=1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True),
        )
        self.branch5_avg = nn.AdaptiveAvgPool2d(1)
        self.branch5_conv = nn.Conv2d(in_chans, out_chans, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(out_chans)
        self.branch5_relu = nn.ReLU(inplace=True)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(out_chans * 5, out_chans, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True))

    def forward(self, x):
        b, c, h, w = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        global_feature = self.branch5_avg(x)
        global_feature = self.branch5_relu(self.branch5_bn(self.branch5_conv(global_feature)))
        global_feature = F.interpolate(global_feature, (h, w), None, 'bilinear', True)

        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result


class Config(object):
    OUTPUT_STRIDE = 16
    ASPP_OUTDIM = 256
    SHORTCUT_DIM = 48
    SHORTCUT_KERNEL = 1
    NUM_CLASSES = 4

class DeeplabV3Plus(nn.Module):
    def __init__(self, cfg, backbone=resnet18_atrous):
        super(DeeplabV3Plus, self).__init__()
        self.backbone = backbone(pretrained=False, os=cfg.OUTPUT_STRIDE)

        input_channel = 512 * self.backbone.block.expansion

        self.aspp = ASPP(in_chans=input_channel, out_chans=cfg.ASPP_OUTDIM, rate=16 // cfg.OUTPUT_STRIDE)

        self.dropout1 = nn.Dropout(0.5)

        self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=cfg.OUTPUT_STRIDE // 4)

        indim = 64 * self.backbone.block.expansion

        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(indim, cfg.SHORTCUT_DIM, cfg.SHORTCUT_KERNEL, 1, padding=cfg.SHORTCUT_KERNEL // 2, bias=False),
            nn.BatchNorm2d(cfg.SHORTCUT_DIM),
            nn.ReLU(inplace=True),
        )

        self.cat_conv = nn.Sequential(
            nn.Conv2d(cfg.ASPP_OUTDIM + cfg.SHORTCUT_DIM, cfg.ASPP_OUTDIM, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(cfg.ASPP_OUTDIM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(cfg.ASPP_OUTDIM, cfg.ASPP_OUTDIM, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(cfg.ASPP_OUTDIM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        self.cls_conv = nn.Conv2d(cfg.ASPP_OUTDIM, cfg.NUM_CLASSES, 1, 1, padding=0)

        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        layers = self.backbone(x)
        feature_aspp = self.aspp(layers[-1])
        feature_aspp = self.dropout1(feature_aspp)
        feature_aspp = self.upsample_sub(feature_aspp)

        feature_shallow = self.shortcut_conv(layers[0])
        feature_cat = torch.cat([feature_aspp, feature_shallow], 1)

        result = self.cat_conv(feature_cat)
        result = self.cls_conv(result)

        result = self.upsample4(result)
        return result





if __name__ == '__main__':
    # x = torch.randn((4, 3, 224, 224))
    # resnet = resnet18_atrous()
    # y = resnet(x)
    # print(y)

    cfg = Config()
    model = DeeplabV3Plus(cfg, backbone=resnet50_atrous)
    x = torch.randn((2, 3, 128, 128), dtype=torch.float32)
    y = model(x)
    print(y.shape)
