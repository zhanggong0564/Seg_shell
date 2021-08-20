import torch
import os
import numpy as np
import random
from tqdm import tqdm
import torch.nn.functional as F
from utils.metrics import *
from datetime import datetime


train_metrics = AccScores(3)
val_metrics = AccScores(3)



def set_seeds(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def train_one_epoch(epoch,model,
                    loss_fn,
                    optimizer,
                    train_loader,
                    vloader,
                    device,scheduler,image_size):
    '''训练集每个epoch训练函数
    Args:
        epoch : int , 训练到第几个 epoch
        model : object, 需要训练的模型
        loss_fn : object, 损失函数
        optimizer : object, 优化方法
        train_loader : object, 训练集数据生成器
        scaler : object, 梯度放大器
        device : str , 使用的训练设备 e.g 'cuda:0'
        scheduler : object , 学习率调整策略
        schd_batch_update : bool, 如果是 true 则每一个 batch 都调整，否则等一个 epoch 结束后再调整
        accum_iter : int , 梯度累加
    '''

    model.train()  # 开启训练模式
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))  # 构造进度条
    losses = []
    for step, (imgs, mask) in pbar:  # 遍历每个 batch
        optimizer.zero_grad()
        imgs = imgs.to(device).float()
        target = mask.to(device).long()
        # output = model(imgs)['out']
        output,cx1_sup,cx2_sup = model(imgs)
        # output = model(imgs)
        out = F.log_softmax(output, dim=1)
        loss0 = loss_fn(output, target)#这里是交叉熵损失函数 不是nll loss
        loss1 = loss_fn(cx1_sup, target)
        loss2 = loss_fn(cx2_sup, target)
        loss = loss0+loss1+loss2
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        #指标计算
        pre_label = out.max(dim=1)[1].data.cpu().numpy()
        true_label = target.data.cpu().numpy()
        train_metrics.update(true_label,pre_label)
        mec_dict = train_metrics.get_scores()

        description = f'epoch {epoch} loss: {loss:.4f}'
        pbar.set_description(description)
        pbar.set_postfix(pixel_acc=mec_dict[0]['pixel_acc'],
                         class_acc = mec_dict[0]['class_acc'],
                         mIou = mec_dict[0]['mIou'],
                         fwIou = mec_dict[0]['fwIou']
                         )
    scheduler.step()
    print("validation")
    miou = validation(model, vloader, image_size,device)
    # mean_loss =np.array(losses).mean()
    # print(f"validation{mean_loss:.4f}")
    return miou


@torch.no_grad()
def validation(model, loader, image_size,DEVICE):
    model.eval()
    prec_time = datetime.now()
    for image, target in loader:
        image, target = image.to(DEVICE), target.long().to(DEVICE)
        # output = model(image)['out']
        output = model(image)
        out = F.log_softmax(output, dim=1)
        pre_label = out.max(dim=1)[1].data.cpu().numpy()
        true_label = target.data.cpu().numpy()
        val_metrics.update(true_label, pre_label)
    metrics = val_metrics.get_scores()
    for k, v in metrics[0].items():
        print(k, v,end=' ')
    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prec_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
    print(time_str)
    return metrics[0]['mIou']