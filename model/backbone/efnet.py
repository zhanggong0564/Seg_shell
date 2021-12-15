# from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torch




class Efnet(nn.Module):
    def __init__(self,backbone = 'efficientnet-b1'):
        super(Efnet, self).__init__()
        model= EfficientNet.from_pretrained(backbone)
        self._blocks = model._blocks
        self._swish =model._swish
        self._bn0 =model._bn0
        self._conv_stem =model._conv_stem
        self._global_params = model._global_params
        self._bn1 = model._bn1
        self._conv_head = model._conv_head
    def extract_features(self, inputs):
        """use convolution layer to extract feature .
        Args:
            inputs (tensor): Input tensor.
        Returns:
            Output of the final convolution
            layer in the efficientnet model.
        """
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx== 7:
                x1 = x
            if idx==12:
                x2  = x



        # Head
        x3 = self._swish(self._bn1(self._conv_head(x)))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        return x1,x2,x3
    def forward(self,inputs):
        x1,x2,x3 = self.extract_features(inputs)
        return x1,x2,x3

if __name__ == '__main__':
    x = torch.randn((4,3,224,224))
    model = Efnet()
    y1,y2,y3 = model(x)
    print(y1.shape,y2.shape,y3.shape)