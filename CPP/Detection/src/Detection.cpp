//
// Created by zhanggong on 2021/7/31.
//

#include "Detection.h"
#include<opencv2/opencv.hpp>
#include<opencv2/dnn/dnn.hpp>
#include<iostream>
#include<time.h>
using namespace std;

Detection::Detection(int width,int height,string model_name,bool is_gpu){
    this->m_width = width;
    this->m_height =height;
    this->m_net = cv::dnn::readNetFromONNX(model_name);
    if (is_gpu){
//        this->m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
//        this->m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    }else{
        this->m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        this->m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    this->outNames = this->m_net.getUnconnectedOutLayersNames();
}
//Detection::~Detection(){
//    this->m_net.empty();
//}
int Detection::Detecting(cv::Mat &image,cv::Mat &mask){
    cv::Mat rgb_image;
    cv::cvtColor(image,rgb_image,cv::COLOR_BGR2RGB);
    cv::Mat frame;
    cv::resize(rgb_image,frame,cv::Size(m_width,m_height));
    cv::Mat inputblob = cv::dnn::blobFromImage(frame, 1/57.375,
                                               cv::Size(m_width, m_height),
                                               cv::Scalar(123.675,116.28,103.53));
    m_net.setInput(inputblob);
    vector<cv::Mat>outs;
    m_net.forward(outs,outNames);
    PostProcess(outs[0],mask);
    return 0;
}
void Detection::PostProcess(cv::Mat &out,cv::Mat &mask){
    vector<uchar> mask_flatten;
    array<float, 3>result_extra{};
    int result_[m_width*m_height];
//    clock_t start = clock();
    for (int i = 0; i <out.rows ; ++i) {
        float* ptr = out.ptr<float>(i);
        for (int j = 0; j <out.cols ; ++j) {
            result_extra[j] = ptr[j];
        }
        result_[i] = distance(result_extra.begin(),max_element(result_extra.begin(),result_extra.end()));
    }
    int *result_ptr = result_;
    for (int i = 0; i <m_height ; ++i) {
        uchar *p = mask.ptr<uchar>(i);
        for (int j = 0; j <m_width ; ++j) {
            if (result_ptr[i*m_width+j]==0){
                p[j] = 0;
            }if (result_ptr[i*m_width+j]==1 or result_ptr[i*m_width+j]==2){
                p[j] = 255;
            }
        }
    }
//    clock_t end = clock();
//    double dur = (double)(end - start);
//    cout<<"demo1 speed time: "<<dur/CLOCKS_PER_SEC<<endl;
//
//    clock_t start1 = clock();
//    for (int i = 0; i <out.rows ; ++i) {
//        cv::Mat scores = out.row(i).colRange(0, out.cols);
//        cv::Point index;
//        double confidence;
//        minMaxLoc(scores, 0, &confidence, 0, &index);
//        mask_flatten.push_back(index.x*255);
//    }
//    cv::Mat mask_mat = cv::Mat(mask_flatten);
//    cout<<mask_mat.data<<endl;
//    mask = mask_mat.reshape(0,448).clone();
//    clock_t end1 = clock();
//    double dur1 = (double)(end1 - start1);
//    cout<<"demo2 speed time: "<<dur1/CLOCKS_PER_SEC<<endl;
}
