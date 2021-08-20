from albumentations.pytorch import ToTensorV2
import albumentations as A


def train_transfor(image_size):
    val_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf([
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
        [A.Resize(180,320)]
    )
    return val_transform