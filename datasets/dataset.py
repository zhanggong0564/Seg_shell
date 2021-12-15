import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


'''
RGB
128,0,0 shell
0,128,0 grenade,
128,128,0 shell1
'''


class ShellDateset(Dataset):
    def __init__(self,data_root,imagelist,mask_list,transforms = None):
        super(ShellDateset, self).__init__()
        self.rootdir = data_root
        self.transform = transforms
        self.imgs_list = imagelist
        self.mask_list = mask_list
        self.colormap = [(0,0,0),(128,0,0),(0,128,0),(128,128,0)]
        self.cm2lbl = self.encode_labels()
        self.norm = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.to_tensor = ToTensorV2()

    def __len__(self):
        return len(self.imgs_list)
    def __getitem__(self, index):
        img_path = os.path.join(self.rootdir,'imgs',self.imgs_list[index])
        mask_path = os.path.join(self.rootdir,'mask',self.mask_list[index])


        img = self.read_img(img_path)
        or_mask = self.read_img(mask_path)
        if img.shape == (480, 640, 3):
            img = cv2.resize(img, (1280, 720))
            or_mask = cv2.resize(or_mask, (1280, 720))
        mask = self.encode_label_img(or_mask)
        # sample = {'image':img.copy(),'mask':mask.copy()}


        if self.transform:
            transformed  = self.transform(image =img,mask = mask)
            img = transformed["image"]
            mask = transformed["mask"]
            img = self.norm(image=img)["image"]
            Totensor = self.to_tensor(image=img, mask=mask)
            img,mask = Totensor["image"],Totensor["mask"]
        return img,mask.to(torch.long)
    def read_img(self,img_path):
        img = cv2.imread(img_path)
        if img is None:
            print()
        img  = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        return img
    def encode_labels(self):
        cm2lbl = np.zeros(256**3)
        for i,cm in enumerate(self.colormap):
            cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i

        return cm2lbl
    def encode_label_img(self,img):
        data = np.array(img,dtype='int32')
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        return np.array(self.cm2lbl[idx], dtype='int64')


class VOCDateset(Dataset):
    def __init__(self,data_root,imagelist,mask_list,transforms = None,image_size=(416,416)):
        super(VOCDateset, self).__init__()
        self.rootdir = data_root
        self.transform = transforms
        self.imgs_list = imagelist
        self.mask_list = mask_list
        self.colormap = [(0,0,0),(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128),(0,128,128),(128,128,128),
                         (64,0,0),(192,0,0),(64,128,0),(192,128,0),(64,0,128),(192,0,128),(64,128,128),
                         (192,128,128),(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)
                         ]
        self.cm2lbl = self.encode_labels()
        self.norm = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.to_tensor = ToTensorV2()
        self.image_size =image_size

    def __len__(self):
        return len(self.imgs_list)
    def __getitem__(self, index):
        img_path = os.path.join(self.rootdir,'JPEGImages',self.imgs_list[index])
        mask_path = os.path.join(self.rootdir,'SegmentationClass',self.mask_list[index])



        img = self.read_img(img_path)
        or_mask = self.read_img(mask_path)
        h, w, c =img.shape
        if h<self.image_size[1] or w<self.image_size[0]:
            img  =cv2.resize(img,self.image_size)
            or_mask = cv2.resize(or_mask, self.image_size)
        mask = self.encode_label_img(or_mask)

        # sample = {'image':img.copy(),'mask':mask.copy()}
        if self.transform:
            transformed  = self.transform(image =img,mask = mask)
            img = transformed["image"]
            mask = transformed["mask"]
            img = self.norm(image=img)["image"]
            Totensor = self.to_tensor(image=img, mask=mask)
            img, mask = Totensor["image"], Totensor["mask"]
        return img,mask.to(torch.long)
    def read_img(self,img_path):
        img = cv2.imread(img_path)
        img  = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        return img
    def encode_labels(self):
        cm2lbl = np.zeros(256**3)
        for i,cm in enumerate(self.colormap):
            cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
        return cm2lbl
    def encode_label_img(self,img):
        data = np.array(img,dtype='int32')
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        return np.array(self.cm2lbl[idx], dtype='int64')

if __name__ == '__main__':
    rootdir = '/home/zhanggong/disk/Extern/AllData/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/'
    # dataset = ShellDateset(rootdir)
    seg_path ="/home/zhanggong/disk/Extern/AllData/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/SegmentationClass/";
    seg_name = os.listdir(seg_path)
    image_name = [name.replace('.png','.jpg') for name in seg_name]
    dataset = VOCDateset(rootdir,image_name,seg_name,None)
    print(np.unique(dataset[0][1]))
    # image = cv2.imread("/home/zhanggong/disk/Elements/data/paoyuan/train_dataset/mask/1167.png")
    # cv2.imshow('im',image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # print()


