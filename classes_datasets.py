'''
paodan竖直分类
datasets 返回 1x28x28,c(int)
'''
import torch
from  torch.utils.data import Dataset
import pandas as pd
import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DatasetsCL(Dataset):
    def __init__(self,roo_dir,csv_file,transform=None):
        super(DatasetsCL, self).__init__()
        self.transform =transform
        self.data_info = pd.read_csv(os.path.join(roo_dir,csv_file))
        self.image = self.data_info['image'].values
        self.label = self.data_info['label'].values
    def __len__(self):
        return self.image.shape[0]
    def __getitem__(self, index):
        image = cv2.imread(self.image[index])
        label =int(self.label[index])
        if self.transform:
            image =self.transform(image)
        return image,label
class imagaug(object):
    def __call__(self, image):
        transform =  A.Compose([
            A.ToGray(p=1),
            A.GaussianBlur(p=0.5),
            A.IAASharpen(alpha=(0.0,0.3),lightness=(0.7,1.3)),
            A.Cutout(),
            ToTensorV2()
        ])
        image = transform(image=image)["image"]
        return image


