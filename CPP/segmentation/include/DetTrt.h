//
// Created by zhanggong on 2021/8/19.
//

#ifndef PAOYUAN_DETTRT_H
#define PAOYUAN_DETTRT_H
#include<iostream>
#include<opencv2/opencv.hpp>
#include"NvInfer.h"
#include"cuda_runtime.h"
#include"logging.h"
#include<fstream>
#include<map>
#include<chrono>
#include <iostream>

using namespace std;
using namespace nvinfer1;
struct input_init{
    const string model_path;
    const char* INPUT_BLOB_NAME;
    const char* OUTPUT_BLOB_NAME;
    int channel;
    int INPUT_H;
    int INPUT_W;
    size_t NumClass;
};


class DetTrt {
public:
    DetTrt(input_init inputs);
    ~DetTrt();
    int Detecting(cv::Mat &image,cv::Mat &mask);
private:
    void preprocess(cv::Mat image,float *data);
    void deserialize();
    void PostProcess(float* prob, cv::Mat &mask);
    void doInference(IExecutionContext& context, float* input, float* output, int batchSize);
    int m_width;
    int m_height;
    IRuntime* runtime;
    const char* INPUT_BLOB_NAME;
    const char* OUTPUT_BLOB_NAME;
    int channels;
    ifstream trt_file;
    int OUTPUT_SIZE;
    ICudaEngine* engine;
    IExecutionContext* context;
    size_t NumClass;

};


#endif //PAOYUAN_DETTRT_H
