import torch
import torchvision
import torch.nn as nn

def get_model():
    model = torchvision.models.segmentation.fcn_resnet50(False)

    pth = torch.load("./pretrain-coco-weights-pytorch/fcn_resnet50_coco-1167a1af.pth")
    for key in ["aux_classifier.0.weight", "aux_classifier.1.weight", "aux_classifier.1.bias",
                "aux_classifier.1.running_mean", "aux_classifier.1.running_var", "aux_classifier.1.num_batches_tracked",
                "aux_classifier.4.weight", "aux_classifier.4.bias"]:
        del pth[key]

    model.classifier[4] = nn.Conv2d(512, 3, kernel_size=(1, 1), stride=(1, 1))
    return model
if __name__ == '__main__':
    get_model()