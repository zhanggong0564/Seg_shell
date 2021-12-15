from torch.utils.data import DataLoader
import torch
from datasets.dataset import  ShellDateset,VOCDateset
import os
from sklearn.model_selection import train_test_split


def prepare_dataloader(data_root,  bs, n_job,trn_transform=None,val_transform=None,image_size=(416,416)):
    '''多进程数据生成器
    Args:
        df : DataFrame , 样本图片的文件名和标签
        trn_idx : ndarray , 训练集索引列表
        val_idx : ndarray , 验证集索引列表
        data_root : str , 图片文件所在路径
        trn_transform : object , 训练集图像增强器
        val_transform : object , 验证集图像增强器
        bs : int , 每次 batchsize 个数
        n_job : int , 使用进程数量
    Returns:
        train_loader, val_loader , 训练集和验证集的数据生成器
    '''
    #
    imgs_list = list(sorted(os.listdir(os.path.join(data_root, 'imgs'))))
    mask_list = [name.replace('.jpg', '.png') for name in imgs_list]
    # mask_list = list(sorted(os.listdir(os.path.join(data_root, 'mask'))))
    # mask_list = list(sorted(os.listdir(os.path.join(data_root, 'SegmentationClass'))))
    # imgs_list = [name.replace('png','jpg') for name in mask_list]
    # seg_path = "/home/zhanggong/disk/Extern/AllData/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/SegmentationClass/";
    # mask_list = os.listdir(seg_path)
    # imgs_list = [name.replace('.png', '.jpg') for name in mask_list]


    X_train, X_test, Y_train, Y_test = train_test_split(imgs_list,mask_list,test_size=0.1,shuffle=True)

    train_ds = ShellDateset(data_root,X_train,Y_train,transforms=trn_transform)
    valid_ds = ShellDateset(data_root,X_test,Y_test,transforms=val_transform)
    # train_ds = VOCDateset(data_root,X_train,Y_train,transforms=trn_transform,image_size=image_size)
    # valid_ds = VOCDateset(data_root,X_test,Y_test,transforms=val_transform,image_size=image_size)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=bs,
        pin_memory=False,
        drop_last=False,
        shuffle=True,
        num_workers=n_job,
    )
    val_loader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=bs,
        pin_memory=False,
        drop_last=False,
        shuffle=False,
        num_workers=n_job,
    )
    return train_loader, val_loader
if __name__ == '__main__':
    rootdir = '../../data/'
    train_loader, val_loader = prepare_dataloader(rootdir,4,0)
    for data in train_loader:
        print(data)