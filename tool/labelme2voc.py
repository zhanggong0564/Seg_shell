import os
import numpy as np
import codecs
import json
from glob import glob
import cv2
import shutil
from sklearn.model_selection import train_test_split



def get_xy(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x_min = np.min(contours[0][..., 0])
    y_min = np.min(contours[0][..., 1])
    x_max = np.max(contours[0][..., 0])
    y_max = np.max(contours[0][..., 1])
    return x_min,y_min,x_max,y_max

def get_masks(im_src):
    gray = cv2.cvtColor(im_src, cv2.COLOR_BGR2GRAY)
    mask_grenade = cv2.inRange(gray, 70, 80)
    mask_shell = cv2.inRange(gray, 30, 45)
    mask_shell1 = cv2.inRange(gray, 105, 120)
    return mask_grenade, mask_shell, mask_shell1

# 1.标签路径
labelme_path = "./total_data/"  # 原始labelme标注数据路径
saved_path = "./VOC2007/"  # 保存路径

# 2.创建要求文件夹
if not os.path.exists(saved_path + "Annotations"):
    os.makedirs(saved_path + "Annotations")
if not os.path.exists(saved_path + "JPEGImages/"):
    os.makedirs(saved_path + "JPEGImages/")
if not os.path.exists(saved_path + "ImageSets/Main/"):
    os.makedirs(saved_path + "ImageSets/Main/")

# 3.获取待处理文件
files = glob(labelme_path + "*.png")
# 4.读取标注信息并写入 xml
for image_path in files:
    json_file_ = image_path.split('/')[-1].split('.')[0]
    # json_filename = labelme_path + json_file_ + ".json"
    # json_file = json.load(open(json_filename, "r", encoding="utf-8"))
    im_src = cv2.imread(image_path.replace('mask','image'))
    height, width, channels = im_src.shape
    masks = get_masks(im_src)
    label = ['grenade', 'shell', 'shell1']
    with codecs.open(saved_path + "Annotations/" + json_file_ + ".xml", "w", "utf-8") as xml:
        xml.write('<annotation>\n')
        xml.write('\t<folder>' + 'paoyuan_data' + '</folder>\n')
        xml.write('\t<filename>' + json_file_ + ".jpg" + '</filename>\n')
        xml.write('\t<source>\n')
        xml.write('\t\t<database>The paoyuan Databases</database>\n')
        xml.write('\t\t<annotation>paoyuan AutoLanding</annotation>\n')
        xml.write('\t\t<image>flickr</image>\n')
        xml.write('\t\t<flickrid>NULL</flickrid>\n')
        xml.write('\t</source>\n')
        xml.write('\t<owner>\n')
        xml.write('\t\t<flickrid>NULL</flickrid>\n')
        xml.write('\t\t<name>zhanggong</name>\n')
        xml.write('\t</owner>\n')
        xml.write('\t<size>\n')
        xml.write('\t\t<width>' + str(width) + '</width>\n')
        xml.write('\t\t<height>' + str(height) + '</height>\n')
        xml.write('\t\t<depth>' + str(channels) + '</depth>\n')
        xml.write('\t</size>\n')
        xml.write('\t\t<segmented>0</segmented>\n')
        for i,mask in enumerate(masks):
            if mask.any()==0:
                continue
            xmin,ymin,xmax,ymax = get_xy(mask)
            if xmax <= xmin:
                pass
            elif ymax <= ymin:
                pass
            else:
                xml.write('\t<object>\n')
                xml.write('\t\t<name>' + label[i] + '</name>\n')
                xml.write('\t\t<pose>Unspecified</pose>\n')
                xml.write('\t\t<truncated>1</truncated>\n')
                xml.write('\t\t<difficult>0</difficult>\n')
                xml.write('\t\t<bndbox>\n')
                xml.write('\t\t\t<xmin>' + str(xmin) + '</xmin>\n')
                xml.write('\t\t\t<ymin>' + str(ymin) + '</ymin>\n')
                xml.write('\t\t\t<xmax>' + str(xmax) + '</xmax>\n')
                xml.write('\t\t\t<ymax>' + str(ymax) + '</ymax>\n')
                xml.write('\t\t</bndbox>\n')
                xml.write('\t</object>\n')
                print(image_path, xmin, ymin, xmax, ymax, label[i])
        xml.write('</annotation>')
    cv2.imwrite(saved_path + "JPEGImages/"+json_file_+".jpg",im_src)
# 6.split files for txt
txtsavepath = saved_path + "ImageSets/Main/"
ftrainval = open(txtsavepath + '/trainval.txt', 'w')
ftest = open(txtsavepath + '/test.txt', 'w')
ftrain = open(txtsavepath + '/train.txt', 'w')
fval = open(txtsavepath + '/val.txt', 'w')
total_files = glob("./VOC2007/Annotations/*.xml")
total_files = [i.split("/")[-1].split(".xml")[0] for i in total_files]
# test_filepath = ""
for file in total_files:
    ftrainval.write(file + "\n")
# test
# for file in os.listdir(test_filepath):
#    ftest.write(file.split(".jpg")[0] + "\n")
# split
train_files, val_files = train_test_split(total_files, test_size=0.15, random_state=42)
# train
for file in train_files:
    ftrain.write(file + "\n")
# val
for file in val_files:
    fval.write(file + "\n")

ftrainval.close()
ftrain.close()
fval.close()

# # ftest.close()