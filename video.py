import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from model.FCN8S import FCN8S
from model.shuffBiseNet import BiSeNet

from datasets.dataloader import prepare_dataloader
from albumentations.pytorch import ToTensorV2
import albumentations as A
import matplotlib.pyplot as plt
import cv2
import glob
# from model.BiSeNet import BiSeNet




def get_model():
    model = torchvision.models.segmentation.fcn_resnet50(False)
    model.classifier[4] = nn.Conv2d(512, 3, kernel_size=(1, 1), stride=(1, 1))
    return model
def val_transfor():
    val_transform = A.Compose(
        [A.Resize(512, 512), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
    )
    return val_transform
if __name__ == '__main__':
    # image_list = glob.glob('../data/imgs/*.jpg')
    cap = cv2.VideoCapture('/home/zhanggong/disk/Elements/data/paoyuan/source/data3.avi')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_class = 3
    BATCH_SIZE = 4
    rootdir = '../data'
    val_transform = val_transfor()



    # net = torch.nn.DataParallel(FCN8S('resnet18', 3)).to(device)
    net = torch.nn.DataParallel(BiSeNet(3)).to(device)
    # print(torch.load("./model_best.pth"))
    # net.load_state_dict(torch.load("./model_best_mutisize.pth"))
    net.load_state_dict(torch.load("./model_best_mutisize_shuff.pth"))
    net.eval()
    colormap = [(0,0,0),(128,0,0),(0,128,0)]
    cm = np.array(colormap).astype('uint8')
    while True:
        res, img_src = cap.read()
        if res:
            show_image = cv2.resize(img_src,(512,512))
            img_rgb= cv2.cvtColor(img_src,cv2.COLOR_BGR2RGB)
            image_tens = val_transform(image = img_rgb)["image"]

            image_tens1 = image_tens.unsqueeze(0).to(device)
            with torch.no_grad():
                import time
                start = time.time()
                out = net(image_tens1)#0.03-0.04
                end = time.time()
                print(end-start)
            out = F.log_softmax(out, dim=1)
            pre_label = out.max(1)[1].squeeze().cpu().data.numpy()
            # pre_label = out.max(1)[1].cpu().data.numpy()
            # pre_label = pre_label.reshape((448,448))
            pre = cm[pre_label]
            add_image = cv2.addWeighted(show_image, 0.7, pre, 0.3, 0)
            # plt.imshow(add_image)
            # plt.show()
            # if i==10:
            #     break
            add_image = cv2.hconcat([add_image,pre])
            cv2.imshow('image',add_image)
            k = cv2.waitKey(20)
            if k==27:
                break
        else:
            cap.release()
            cv2.destroyAllWindows()
            break


        # pre1 = Image.fromarray(pre_label)
        # pre1.save(dir + str(i) + '.png')
        # print('Done')