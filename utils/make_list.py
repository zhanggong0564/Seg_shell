import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

# image_0 = os.listdir('/home/zhanggong/disk/Extern/workspace/HOG+SVM/paodan_src/data/0')
# image_1 = os.listdir('/home/zhanggong/disk/Extern/workspace/HOG+SVM/paodan_src/data/1')
#
# root_dir ='/home/zhanggong/disk/Extern/workspace/HOG+SVM/paodan_src/data'

# def make_data(image_name_list,dst_dir):
#     X_train,X_test = train_test_split(image_name_list,test_size=0.2)
#     train_dst = os.path.join(root_dir, 'train')+'/'+str(dst_dir)
#     if not os.path.exists(train_dst):
#         os.makedirs(train_dst)
#     for  x_train in X_train:
#         path_trian = os.path.join(root_dir,str(dst_dir),x_train)
#         post_train = os.path.join(train_dst,x_train)
#         shutil.move(path_trian,post_train)
# make_data(image_0,0)
# make_data(image_1,1)


train_image_list = []
train_label_list = []
val_image_list=[]
val_label_list = []

train_image_dir = '/home/zhanggong/disk/Extern/workspace/HOG+SVM/paodan_src/data/train'
data_list_dir = '../logs'
val_image_dir = '/home/zhanggong/disk/Extern/workspace/HOG+SVM/paodan_src/data/val'

for s1 in os.listdir(train_image_dir):
    if s1 == '0':
        image_sub_dir = os.path.join(train_image_dir,s1)
        for s2 in os.listdir(image_sub_dir):
            image_sub_dir1 = os.path.join(image_sub_dir,s2)
            train_image_list.append(image_sub_dir1)
            train_label_list.append(0)
    else:
        image_sub_dir = os.path.join(train_image_dir, s1)
        for s2 in os.listdir(image_sub_dir):
            image_sub_dir1 = os.path.join(image_sub_dir, s2)
            train_image_list.append(image_sub_dir1)
            train_label_list.append(1)
for s1 in os.listdir(val_image_dir):
    if s1 == '0':
        image_sub_dir = os.path.join(val_image_dir,s1)
        for s2 in os.listdir(image_sub_dir):
            image_sub_dir1 = os.path.join(image_sub_dir,s2)
            val_image_list.append(image_sub_dir1)
            val_label_list.append(0)
    else:
        image_sub_dir = os.path.join(val_image_dir, s1)
        for s2 in os.listdir(image_sub_dir):
            image_sub_dir1 = os.path.join(image_sub_dir, s2)
            val_image_list.append(image_sub_dir1)
            val_label_list.append(1)
train = pd.DataFrame({'image':train_image_list,'label':train_label_list})
val = pd.DataFrame({'image':val_image_list,'label':val_label_list})
if not os.path.exists(data_list_dir):
    os.makedirs(data_list_dir)
train_info_dir = os.path.join(data_list_dir,'train.csv')
val_info_dir = os.path.join(data_list_dir,'val.csv')
train.to_csv(train_info_dir,index=False)
val.to_csv(val_info_dir,index=False)
print('Finish')
