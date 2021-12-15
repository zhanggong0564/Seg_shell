from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np
import math


def train_transfor(image_size):
    val_transform = A.Compose(
        [
            A.CLAHE(),
            A.HorizontalFlip(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1,scale_limit=0.2,rotate_limit=20,p=0.5),
            A.RandomCrop(image_size[1],image_size[0],p=0.5),
            A.VerticalFlip(p=0.5),
            A.Blur(blur_limit=3),
            A.OneOf([
                A.RandomRotate90(p=0.5),
                A.RandomContrast(p=1),
                A.RandomGamma(p=1),
                A.RandomBrightness(p=1),
            ], p=0.5),
            A.Resize(image_size[1], image_size[0]),
        ]
    )
    return val_transform
def val_transfor():
    val_transform = A.Compose(
        [A.Resize(360,640)]
    )
    return val_transform

class RandomResizedCrop(object):
    '''
    size should be a tuple of (H, W)
    '''
    def __init__(self, scales=(0.5, 1.), size=(384, 384)):
        self.scales = scales
        self.size = size

    def __call__(self, im_lb):
        if self.size is None:
            return im_lb

        im, lb = im_lb['im'], im_lb['lb']
        assert im.shape[:2] == lb.shape[:2]

        crop_h, crop_w = self.size
        scale = np.random.uniform(min(self.scales), max(self.scales))
        im_h, im_w = [math.ceil(el * scale) for el in im.shape[:2]]
        im = cv2.resize(im, (im_w, im_h))
        lb = cv2.resize(lb, (im_w, im_h), interpolation=cv2.INTER_NEAREST)

        if (im_h, im_w) == (crop_h, crop_w): return dict(im=im, lb=lb)
        pad_h, pad_w = 0, 0
        if im_h < crop_h:
            pad_h = (crop_h - im_h) // 2 + 1
        if im_w < crop_w:
            pad_w = (crop_w - im_w) // 2 + 1
        if pad_h > 0 or pad_w > 0:
            im = np.pad(im, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)))
            lb = np.pad(lb, ((pad_h, pad_h), (pad_w, pad_w)), 'constant', constant_values=255)

        im_h, im_w, _ = im.shape
        sh, sw = np.random.random(2)
        sh, sw = int(sh * (im_h - crop_h)), int(sw * (im_w - crop_w))
        return dict(
            im=im[sh:sh+crop_h, sw:sw+crop_w, :].copy(),
            lb=lb[sh:sh+crop_h, sw:sw+crop_w].copy()
        )

if __name__ == '__main__':
    import cv2
    import glob

    crp = RandomResizedCrop()
    image_list = glob.glob("/home/zhanggong/disk/Elements/data/paoyuan/train_dataset/imgs/*.jpg")
    mask_list = glob.glob("/home/zhanggong/disk/Elements/data/paoyuan/train_dataset/mask/*.png")
    for image_path in zip(image_list,mask_list):
        image = cv2.imread(image_path[0])
        mask = cv2.imread(image_path[1],1)
        # ROI = image[120:,40:1240,:]
        # ROI = cv2.resize(ROI,(600,300))
        im_dict = {"im":image,"lb":mask}
        m = crp(im_dict)
        cv2.imshow("tets",m["im"])
        cv2.imshow("tets", m["lb"])
        cv2.waitKey()
        cv2.destroyAllWindows()