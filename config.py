from easydict import EasyDict
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


cfg = EasyDict()
cfg.BATCH_SIZE = 4*4
cfg.rootdir = '../data/'
cfg.rootdir_ex = '/home/zhanggong/disk/Extern/AllData/paoyuan/yls_data/'
# cfg.rootdir = '/home/zhanggong/disk/Extern/AllData/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/'
cfg.NUMCLASS =4

cfg.lr=1e-3
cfg.weight_decay=1e-3

cfg.pre_model = 'model_best_mutisize_shuff.pth'

cfg.T_0=5
cfg.T_mult = 1
cfg.eta_min = 1e-6

cfg.EPOCHES = 300
cfg.momentum = 0.9
cfg.weight_decay = 1e-4

cfg.factor = 0.1
cfg.milestones = [30, 100,]

# cfg.image_size =[(1280,720),(640,360),(320,180)]
cfg.image_size =[(640,360),(320,180)]


class Config(object):
    OUTPUT_STRIDE = 16
    ASPP_OUTDIM = 256
    SHORTCUT_DIM = 48
    SHORTCUT_KERNEL = 1
    NUM_CLASSES = 4
cfg.deepalab = Config()