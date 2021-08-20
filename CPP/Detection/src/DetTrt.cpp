//
// Created by zhanggong on 2021/8/19.
//

#include "DetTrt.h"
#include<fstream>
#include <iostream>
using namespace std;
using namespace nvinfer1;
static Logger gLogger;
#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)
DetTrt::DetTrt(input_init inputs) :trt_file(inputs.model_path){
    this->m_width = inputs.INPUT_W;
    this->m_height = inputs.INPUT_H;
    this->INPUT_BLOB_NAME = inputs.INPUT_BLOB_NAME;
    this->OUTPUT_BLOB_NAME = inputs.OUTPUT_BLOB_NAME;
    this->channels = inputs.channel;
    this->runtime = createInferRuntime(gLogger);
    this->OUTPUT_SIZE = inputs.INPUT_W*inputs.INPUT_H*inputs.channel;
    if(!trt_file) { //打开失败
        cout << "error opening source file." << endl;
    }
    deserialize();
}
DetTrt::~DetTrt(){
    trt_file.close();
    context->destroy();
    engine->destroy();
    runtime->destroy();
}


void DetTrt::deserialize(){
    string cached_engine = "";
    while(trt_file.peek()!=EOF){
        stringstream buffer;
        buffer<<trt_file.rdbuf();
        cached_engine.append(buffer.str());
    }
    assert(runtime!=nullptr);
    engine = runtime->deserializeCudaEngine(cached_engine.data(),cached_engine.size(),nullptr);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context!=nullptr);
}
int DetTrt::Detecting(cv::Mat &image, cv::Mat &mask) {
    float data[m_height * m_width * channels];
    float prob[m_width*m_height];
    preprocess(image,data);
    doInference(*context, data, prob, 1);
    PostProcess(prob,mask);

}
void DetTrt::doInference(IExecutionContext &context, float *input, float *output, int batchSize) {
    const ICudaEngine& engine =context.getEngine();//引用
    assert(engine.getNbBindings() == 2);
    void * buffers[2];

    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * m_height * m_width *3* sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize *  OUTPUT_SIZE * sizeof(float)));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize *  m_height * m_width *3* sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize,buffers,stream,nullptr);

    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize *  OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));

}
void DetTrt::PostProcess(float* prob, cv::Mat &mask) {
    int i,j;
    std::array<float, 3> results_extra{};
    int result_[m_height * m_width];
    for (i = 0,j=0;i <OUTPUT_SIZE&&j<(m_height*m_width) ; i+=3,j++) {
        results_extra[0] = prob[i];
        results_extra[1] = prob[i + 1];
        results_extra[2] = prob[i + 2];
        result_[j] = distance(results_extra.begin(),max_element(results_extra.begin(),results_extra.end()));
    }
    int *result = result_;
    for (int i = 0; i < m_height; i++) {
        uchar *p = mask.ptr<uchar>(i);
        for (int j = 0; j <  m_width; j++) {
            if (result[i *  m_width + j] == 0) {
                p[j] =0;
            }
            if (result[i *  m_width + j] == 1) {
                p[j] =255;
            }
            if (result[i *  m_width + j] == 2) {
                p[j] =255;
            }
        }
    }
}
void DetTrt::preprocess(cv::Mat image,float *data) {
    cv::resize(image,image,cv::Size(m_width,m_height));
    for (int i = 0; i <m_height * m_width ; ++i) {
        data[i] =(image.at<cv::Vec3b>(i)[2] / 255.0-0.485)/0.225 ;
        data[i + m_height * m_width] = (image.at<cv::Vec3b>(i)[1] / 255.0-0.456)/0.224;
        data[ i + 2 * m_height * m_width] = (image.at<cv::Vec3b>(i)[0] / 255.0-0.406)/0.229;
    }
}