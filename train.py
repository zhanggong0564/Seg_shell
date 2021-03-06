from torch import nn
import torch
from torch.utils.data import ConcatDataset
from datasets.dataaug import train_transfor,val_transfor
# from model.FCN import get_model
from imp import reload
from utils import utils
from datasets.dataloader import prepare_dataloader
# from model.FCN8S import FCN8S
from model.BiSeNet import BiSeNet
from model.Unet import UNet
from model.Unetpp import ResNetUNetpp
from model.deeplabv3Plus import DeeplabV3Plus
from utils.ohem_celoss import OhemCELoss
from losses.losses import *
from losses.lovasz_losses import lovasz_softmax
# import random
from config import cfg
from datasets.dataset import  ShellDateset
import os
from sklearn.model_selection import train_test_split

reload(utils)
rand_seed = 666
utils.set_seeds(rand_seed)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")






if __name__ == '__main__':
    val_transform = val_transfor()

    image_size = (640, 368)
    train_transform = train_transfor(image_size)


    imgs_list = list(sorted(os.listdir(os.path.join(cfg.rootdir, 'imgs'))))
    mask_list = [name.replace('.jpg', '.png') for name in imgs_list]
    X_train, X_test, Y_train, Y_test = train_test_split(imgs_list,mask_list,test_size=0.1,shuffle=True)

    X_train2 =list(sorted(os.listdir(os.path.join(cfg.rootdir_ex, 'imgs'))))
    Y_train2 = [name.replace('.jpg', '.png') for name in X_train2]

    train_ds = ShellDateset(cfg.rootdir,X_train,Y_train,transforms=train_transform)
    print(len(train_ds))
    train_ds_ex = ShellDateset(cfg.rootdir_ex,X_train2,Y_train2,transforms=train_transform)

    train_ds = ConcatDataset([train_ds,train_ds_ex])
    print(len(train_ds))


    valid_ds = ShellDateset(cfg.rootdir,X_test,Y_test,transforms=val_transform)
    # train_ds = VOCDateset(data_root,X_train,Y_train,transforms=trn_transform,image_size=image_size)
    # valid_ds = VOCDateset(data_root,X_test,Y_test,transforms=val_transform,image_size=image_size)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        pin_memory=False,
        drop_last=False,
        shuffle=True,
        num_workers=8,
    )
    val_loader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=cfg.BATCH_SIZE,
        pin_memory=False,
        drop_last=False,
        shuffle=False,
        num_workers=8,
    )

    # train_loader, val_loader = prepare_dataloader(cfg.rootdir, cfg.BATCH_SIZE, 8, trn_transform=train_transform,
    #                                               val_transform=val_transform, image_size=image_size)

    # model = get_model()
    # model = FCN8S('resnet18', 3)
    # model = UNet(cfg.NUMCLASS)
    # model = ResNetUNetpp(cfg.NUMCLASS)
    # model = DeeplabV3Plus(cfg.deepalab)
    model = BiSeNet(cfg.NUMCLASS)

    # state = torch.load('model_final_v1_city_new.pth')
    # fine_tune_net_dict = model.state_dict()
    # update_dict = {k: v for k, v in fine_tune_net_dict.items() if k in state.keys()}
    # model.load_state_dict(update_dict,False)
    # model.load_state_dict(torch.load('model_final_v1_city_new.pth'),strict=False)
    # torch.save(model.state_dict(), 'model_final_v1_city_new.pth',_use_new_zipfile_serialization=False)

    model = torch.nn.DataParallel(model)
    # model.load_state_dict(torch.load('../UNet++_best.pth'))

    model.to(DEVICE)

    # optimizer = torch.optim.Adam(model.parameters(),lr=cfg.lr, weight_decay=cfg.weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=10,
    #     T_mult=1,
    #     eta_min=1e-6,
    #     last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=cfg.factor, milestones=cfg.milestones)
    # scheduler.step()

    bce_fn = nn.CrossEntropyLoss()
    ohem = OhemCELoss(0.5)
    focal_fn = FacalLoss()
    dice_fn =DiceLoss()
    focal_dice = Focal_Dice(0.5)

    best_loss = 0
    for epoch in range(1, cfg.EPOCHES + 1):
        # image_size = random.choice([(1280,720),(640,360),(320,180)])
        # train_transform = train_transfor(image_size)
        # train_loader, val_loader = prepare_dataloader(cfg.rootdir, cfg.BATCH_SIZE, 8, trn_transform=train_transform,
        #                                               val_transform=val_transform,image_size=image_size)
        miou= utils.ModelTrainer.train_one_epoch(epoch,model,loss_fn=lovasz_softmax,
                              optimizer=optimizer,
                              train_loader=train_loader,
                              vloader=val_loader,
                              device=DEVICE,
                              scheduler =scheduler,image_size = image_size)


        if best_loss<miou:
            best_loss = miou
            torch.save(model.state_dict(), f'../Bisnet_best.pth')
            print( f'saved Bisnet_best_{best_loss}.pth')






