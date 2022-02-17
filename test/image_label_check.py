import os
import cv2



if __name__ == '__main__':
    rootdir='/home/zhanggong/disk/Extern/AllData/paoyuan/yls_data'
    imgs_list = list(sorted(os.listdir(os.path.join(rootdir, 'imgs'))))
    mask_list = [name.replace('.jpg', '.png') for name in imgs_list]

    for image_path,mask_path in zip(imgs_list,mask_list):
        image_path1 = os.path.join(rootdir,'imgs',image_path)
        mask_path1 = os.path.join(rootdir, 'mask', mask_path)

        image = cv2.imread(image_path1)
        mask =cv2.imread(mask_path1)

        add_image = cv2.addWeighted(image,0.6,mask,0.4,0)
        add_image = cv2.resize(add_image,None,fx=0.5,fy=0.5)
        check = image_path1.replace('imgs','check')
        cv2.imwrite(check,add_image)

