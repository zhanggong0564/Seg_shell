import glob
import cv2
import tqdm
import numpy as np


data_root = "/home/zhanggong/disk/Elements/ubuntu18.04/paoyuan/data/mask/"


mask_list = glob.glob(data_root+"*.png")

w_h = []
area = []
lengths = []
for  mask_path in tqdm.tqdm(mask_list):
    mask = cv2.imread(mask_path)
    binary = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    binary[binary > 0] = 255

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, )
    for cnt in contours:
        contours_area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        length = cv2.arcLength(cnt, True)
        scale_w_h = w / h

        w_h.append(scale_w_h)
        area.append(contours_area)
        lengths.append(length)

max_wh = max(w_h)
max_area = max(area)
max_len = max(lengths)

min_wh = min(w_h)
min_area = min(area)
min_len = min(lengths)

info = f"min_wh: {min_wh:2f} max_wh: {max_wh:2f} min_area: {min_area:2f} max_area: {max_area:2f} min_len: {min_len:2f} max_len: {max_len:2f}"
with open('../logs/feature','w') as f:
    f.write(info)

print(info)
