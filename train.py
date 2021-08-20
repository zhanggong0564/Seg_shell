from torch import nn
import torch
from datasets.dataaug import train_transfor,val_transfor
from model.FCN import get_model
from imp import reload
from utils import utils
from datasets.dataloader import prepare_dataloader
from model.FCN8S import FCN8S
from model.BiSeNet import BiSeNet
from utils.ohem_celoss import OhemCELoss
import random

reload(utils)
rand_seed = 666
utils.set_seeds(rand_seed)






if __name__ == '__main__':
    BATCH_SIZE = 8
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    rootdir = '../data/'
    # rootdir = '/home/zhanggong/disk/Elements/VOC2012/'
    val_transform = val_transfor()
    # model = get_model()
    # model = FCN8S('resnet18', 3)
    model = BiSeNet(3)

    # state = torch.load('model_final_v1_city_new1.4.pth')
    # fine_tune_net_dict = model.state_dict()
    # update_dict = {k: v for k, v in fine_tune_net_dict.items() if k in state.keys()}
    # model.load_state_dict(update_dict,False)
    # model.load_state_dict(torch.load('model_final_v1_city_new1.5.pth'),strict=False)
    # torch.save(model.state_dict(), 'model_final_v1_city_new1.5.pth',_use_new_zipfile_serialization=False)

    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load('model_best_mutisize_shuff.pth'),strict=False)

    model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=1,
        eta_min=1e-6,
        last_epoch=-1)

    bce_fn = nn.CrossEntropyLoss()
    ohem = OhemCELoss(0.5)

    best_loss = 0
    EPOCHES = 500
    for epoch in range(1, EPOCHES + 1):
        image_size = random.choice([(1280,720),(640,360),(320,180)])
        train_transform = train_transfor(image_size)
        train_loader, val_loader = prepare_dataloader(rootdir, BATCH_SIZE, 0, trn_transform=train_transform,
                                                      val_transform=val_transform)
        miou= utils.train_one_epoch(epoch,model,loss_fn=ohem,
                              optimizer=optimizer,
                              train_loader=train_loader,
                              vloader=val_loader,
                              device=DEVICE,
                              scheduler =scheduler,image_size = image_size)


        if best_loss<miou:
            best_loss = miou
            torch.save(model.state_dict(), 'model_best_mutisize_shuff.pth')






