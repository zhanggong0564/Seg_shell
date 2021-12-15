from torch import nn
import torch
from datasets.dataaug import train_transfor,val_transfor
# from model.FCN import get_model
from imp import reload
from utils import utils
from datasets.dataloader import prepare_dataloader
# from model.FCN8S import FCN8S
from model.BiSeNet import BiSeNet
from utils.ohem_celoss import OhemCELoss
# import random
from config import cfg

reload(utils)
rand_seed = 666
utils.set_seeds(rand_seed)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")






if __name__ == '__main__':
    val_transform = val_transfor()

    image_size = (640, 360)
    train_transform = train_transfor(image_size)
    train_loader, val_loader = prepare_dataloader(cfg.rootdir, cfg.BATCH_SIZE, 8, trn_transform=train_transform,
                                                  val_transform=val_transform, image_size=image_size)

    # model = get_model()
    # model = FCN8S('resnet18', 3)
    model = BiSeNet(cfg.NUMCLASS)

    # state = torch.load('model_final_v1_city_new.pth')
    # fine_tune_net_dict = model.state_dict()
    # update_dict = {k: v for k, v in fine_tune_net_dict.items() if k in state.keys()}
    # model.load_state_dict(update_dict,False)
    # model.load_state_dict(torch.load('model_final_v1_city_new.pth'),strict=False)
    # torch.save(model.state_dict(), 'model_final_v1_city_new.pth',_use_new_zipfile_serialization=False)

    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load('../Bisnet_best_0.8831062157654177.pth'))

    model.to(DEVICE)

    # optimizer = torch.optim.Adam(model.parameters(),lr=cfg.lr, weight_decay=cfg.weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=1,
        eta_min=1e-6,
        last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=cfg.factor, milestones=cfg.milestones)

    bce_fn = nn.CrossEntropyLoss()
    ohem = OhemCELoss(0.5)

    best_loss = 0
    EPOCHES = 500
    for epoch in range(1, EPOCHES + 1):
        # image_size = random.choice([(1280,720),(640,360),(320,180)])
        # train_transform = train_transfor(image_size)
        # train_loader, val_loader = prepare_dataloader(cfg.rootdir, cfg.BATCH_SIZE, 8, trn_transform=train_transform,
        #                                               val_transform=val_transform,image_size=image_size)
        miou= utils.ModelTrainer.train_one_epoch(epoch,model,loss_fn=ohem,
                              optimizer=optimizer,
                              train_loader=train_loader,
                              vloader=val_loader,
                              device=DEVICE,
                              scheduler =scheduler,image_size = image_size)


        if best_loss<miou:
            best_loss = miou
            torch.save(model.state_dict(), f'../Bisnet_best.pth')
            print( f'saved Bisnet_best_{best_loss}.pth')






