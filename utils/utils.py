import torch
import os
import numpy as np
import random
from tqdm import tqdm
import torch.nn.functional as F
from utils.metrics import *
from datetime import datetime
from config import cfg
import logging
from model.BiSeNet import BiSeNet



# val_model = BiSeNet(cfg.NUMCLASS, mode='infer').cuda()



def set_seeds(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class ModelTrainer(object):
    train_metrics = AccScores(cfg.NUMCLASS)
    val_metrics = AccScores(cfg.NUMCLASS)
    @staticmethod
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
        ModelTrainer.train_metrics.reset()
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
            ModelTrainer.train_metrics.update(true_label,pre_label)
            mec_dict = ModelTrainer.train_metrics.get_scores()

            description = f'epoch {epoch} loss: {loss:.4f}'
            pbar.set_description(description)
            pbar.set_postfix(pixel_acc=mec_dict[0]['pixel_acc'],
                             class_acc = mec_dict[0]['class_acc'],
                             mIou = mec_dict[0]['mIou'],
                             fwIou = mec_dict[0]['fwIou']
                             )
        scheduler.step()
        print("validation")
        miou = ModelTrainer.validation(model, vloader, image_size,device)
        # mean_loss =np.array(losses).mean()
        # print(f"validation{mean_loss:.4f}")
        return miou
    @staticmethod
    @torch.no_grad()
    def validation(model, loader, image_size, DEVICE):
        model.eval()
        ModelTrainer.val_metrics.reset()
        prec_time = datetime.now()
        for image, target in loader:
            image, target = image.to(DEVICE), target.long().to(DEVICE)
            # output = model(image)['out']
            output = model(image)
            out = F.log_softmax(output, dim=1)
            pre_label = out.max(dim=1)[1].data.cpu().numpy()
            true_label = target.data.cpu().numpy()
            ModelTrainer.val_metrics.update(true_label, pre_label)
        metrics = ModelTrainer.val_metrics.get_scores()
        for k, v in metrics[0].items():
            print(k, v, end=' ')
        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prec_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
        print(time_str)
        return metrics[0]['mIou']



class Logger(object):
    def __init__(self, path_log):
        log_name = os.path.basename(path_log)#log.log
        self.log_name = log_name if log_name else "root"
        self.out_path = path_log

        log_dir = os.path.dirname(self.out_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def init_logger(self):
        logger = logging.getLogger(self.log_name)
        logger.setLevel(level=logging.INFO)

        # 配置文件Handler
        file_handler = logging.FileHandler(self.out_path, 'w')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # 配置屏幕Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        # console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

        # 添加handler
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger