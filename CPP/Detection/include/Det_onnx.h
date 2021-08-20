//
// Created by zhanggong on 2021/8/9.
//

#ifndef PAOYUAN_DET_ONNX_H
#define PAOYUAN_DET_ONNX_H
#include "onnxruntime_cxx_api.h"
#include "opencv2/opencv.hpp"
#include<iostream>
using  namespace std;
struct input_init{
    const char* model_path;
    int channel;
    int net_input_row;
    int net_input_col;
    vector<const char *>input_layer_names;
    vector<const char*>output_node_name;
};


class Det_onnx {
public:
    explicit Det_onnx(input_init init);
    int Detecting(cv::Mat &image,cv::Mat &mask);

private:
    void PostProcess(float *results, cv::Mat &mask);
    Ort::Env _env;
    Ort::Session* _session{nullptr};
    input_init in;
    vector<int64_t> _input_node_dims;
    size_t input_tensor_size;
};


#endif //PAOYUAN_DET_ONNX_H
