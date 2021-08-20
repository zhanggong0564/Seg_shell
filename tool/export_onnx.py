import sys
import onnx
import os
import argparse
import numpy as np
import cv2
import onnxruntime
import torch
from model.BiSeNet import BiSeNet

from model.FCN8S import FCN8S


def load_weights(weights,model):
    state_dict = torch.load(weights)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model


def transform_to_onnx(weight_file, batch_size, n_classes, IN_IMAGE_H, IN_IMAGE_W):
    model = BiSeNet(n_classes)

    model = load_weights(weight_file,model)

    input_names = ["input"]
    output_names = ["score"]

    dynamic = False
    if batch_size <= 0:
        dynamic = True

    if dynamic:
        x = torch.randn((1, 3, IN_IMAGE_H, IN_IMAGE_W), requires_grad=True)
        onnx_file_name = "FCN8S-1_3_{}_{}_dynamic.onnx".format(IN_IMAGE_H, IN_IMAGE_W)
        dynamic_axes = {"input": {0: "batch_size"}, "score": {0: "batch_size"}}
        # Export the model
        print('Export the onnx model ...')
        torch.onnx.export(model,
                          x,
                          onnx_file_name,
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=dynamic_axes)

        print('Onnx model exporting done')
        return onnx_file_name

    else:
        x = torch.randn((batch_size, 3, IN_IMAGE_H, IN_IMAGE_W), requires_grad=True)
        onnx_file_name = "FCN8S_{}_3_{}_{}_static.onnx".format(batch_size, IN_IMAGE_H, IN_IMAGE_W)
        # Export the model
        print('Export the onnx model ...')
        torch.onnx.export(model,
                          x,
                          onnx_file_name,
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=None)

        print('Onnx model exporting done')
        return onnx_file_name


def main(weight_file, batch_size, n_classes, IN_IMAGE_H, IN_IMAGE_W):
    if batch_size <= 0:
        onnx_path_demo = transform_to_onnx(weight_file, batch_size, n_classes, IN_IMAGE_H, IN_IMAGE_W)
    else:
        # Transform to onnx as specified batch size
        # transform_to_onnx(weight_file, batch_size, n_classes, IN_IMAGE_H, IN_IMAGE_W)
        # Transform to onnx for demo
        onnx_path_demo = transform_to_onnx(weight_file, 1, n_classes, IN_IMAGE_H, IN_IMAGE_W)

    # session = onnxruntime.InferenceSession(onnx_path_demo)
    # # session = onnx.load(onnx_path)
    # print("The model expects input shape: ", session.get_inputs()[0].shape)
    #
    # image_src = cv2.imread(image_path)


if __name__ == '__main__':
    main('../model_best_mutisize_shuff.pth',1,3,360,640)
    # (1280, 720)
    #640,360