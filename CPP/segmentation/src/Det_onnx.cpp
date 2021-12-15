//
// Created by zhanggong on 2021/8/9.
//

#include "Det_onnx.h"
#include <stdlib.h>
#include <vector>
Det_onnx::Det_onnx(input_init input) {
    in =input;
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    _env = Ort::Env(ORT_LOGGING_LEVEL_WARNING,"onnx");
    _session = new Ort::Session(_env,in.model_path,session_options);
    _input_node_dims = {1,in.channel,in.net_input_row,in.net_input_col};
}
int Det_onnx::Detecting(cv::Mat &image, cv::Mat &outputimage) {
    cv::Mat net_image;
    cv::resize(image,net_image,cv::Size(in.net_input_col,in.net_input_row));
    input_tensor_size = in.channel*in.net_input_row*in.net_input_col;//创建tensor 传入memory_info 输入values的地址，values的总个数，输入的形状（n,c,w,h) 地址传入，shape的长度
    vector<float> _input_tensor(input_tensor_size);
    for (int c = 0; c <in.channel ; ++c) {
        for (int i = 0; i <in.net_input_row ; ++i) {
            for (int j = 0; j <in.net_input_col ; ++j) {
                if (c==0){
                    _input_tensor[c*in.net_input_row*in.net_input_col+i*in.net_input_col+i] = ((net_image.ptr<uchar>(i)[j*3+c])/255.0-0.485)/0.225;
                }
                if (c==1){
                    _input_tensor[c*in.net_input_row*in.net_input_col+i*in.net_input_col+i] = ((net_image.ptr<uchar>(i)[j*3+c])/255.0-0.456)/0.224;
                }
                if (c==2){
                    _input_tensor[c*in.net_input_row*in.net_input_col+i*in.net_input_col+i] = ((net_image.ptr<uchar>(i)[j*3+c])/255.0-0.406)/0.229;
                }
            }
        }
    }
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info,_input_tensor.data(),input_tensor_size,_input_node_dims.data(),4);
    auto output_tensors = _session->Run(Ort::RunOptions{nullptr},in.input_layer_names.data(),&input_tensor,1,in.output_node_name.data(),1);
    float * results = output_tensors.front().GetTensorMutableData<float>();
    PostProcess(results,outputimage);
}
void Det_onnx::PostProcess(float *results, cv::Mat &outputimage) {
    std::array<float, 3> results_extra{};
    int result_[in.net_input_col*in.net_input_row];
    int i,j;
    for (i = 0,j=0;i <in.net_input_col*in.net_input_row*in.channel&&j<in.net_input_row*in.net_input_col ; i+=3,j++) {
        results_extra[0] = results[i];
        results_extra[1] = results[i + 1];
        results_extra[2] = results[i + 2];
        result_[j] = distance(results_extra.begin(),max_element(results_extra.begin(),results_extra.end()));
    }
    int *result = result_;
    for (int i = 0; i < in.net_input_col; i++) {
        uchar *p = outputimage.ptr<uchar>(i);
        for (int j = 0; j <  in.net_input_row; j++) {
            if (result[i *  in.net_input_row + j] == 0) {
                p[j*3+0] =0;
                p[j*3+1] = 0;
                p[j*3+2] = 0;
            }
            if (result[i *  in.net_input_row + j] == 1) {
                p[j*3+0] =0;
                p[j*3+1] = 255;
                p[j*3+2] = 0;
            }
            if (result[i *  in.net_input_row + j] == 2) {
                p[j*3+0] =0;
                p[j*3+1] = 0;
                p[j*3+2] = 255;
            }
        }
    }
}
