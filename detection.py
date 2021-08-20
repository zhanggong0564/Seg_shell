import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from model.FCN8S import FCN8S

from albumentations.pytorch import ToTensorV2
import albumentations as A
import cv2
import glob
from model.BiSeNet import BiSeNet


class Detect(object):
    def __init__(self,weights):
        super(Detect, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # self.net = FCN8S('resnet18', 3)
        self.net = BiSeNet(3)
        self.load_weights(weights)
        self.net.to(self.device)
        self.net.eval()
        self.colormap = [(0,0,0),(128,0,0),(0,128,0)]
        self.cm = np.array(self.colormap).astype('uint8')
        self.val_transform = self.val_transfor()
    def detect(self,img):
        img_src = cv2.resize(img, (448,448))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_tens = self.val_transform(image=img_rgb)["image"]

        image_tens1 = image_tens.unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.net(image_tens1)
        # out = F.log_softmax(out, dim=1)
        # pre_label = out.max(1)[1].squeeze().cpu().data.numpy()
        pre_label = out.max(1)[1].cpu().data.numpy()
        pre_label = pre_label.reshape((448, 448))
        pre = self.cm[pre_label]
        return pre,img_src

    def val_transfor(self):
        val_transform = A.Compose(
            [A.Resize(448, 448), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
        )
        return val_transform

    def load_weights(self,weights):
        state_dict = torch.load(weights)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k,v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        self.net.load_state_dict(new_state_dict)
if __name__ == '__main__':
    Det = Detect('./model_map0.78_Bisenet_.pth')
    image_list = glob.glob('../data/imgs/33.jpg')
    for image_path in image_list:
        image = cv2.imread(image_path)
        pre = Det.detect(image)
        cv2.imshow('image', pre[0])
        cv2.waitKey()
        cv2.destroyAllWindows()