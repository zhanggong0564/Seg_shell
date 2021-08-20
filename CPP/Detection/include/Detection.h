//
// Created by zhanggong on 2021/7/31.
//

#ifndef YOLOV4_PYTORCH_DETECTION_H
#define YOLOV4_PYTORCH_DETECTION_H
#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/dnn/dnn.hpp>
using namespace std;


class Detection {
public:
    Detection(int width,int height,string model_name,bool is_gpu);
//    ~Detection();
    int Detecting(cv::Mat &image,cv::Mat &mask);

private:
    void PostProcess(cv::Mat &out,cv::Mat &mask);
    int m_width;
    int m_height;
    cv::dnn::Net m_net;
    vector<string>outNames;
};


#endif //YOLOV4_PYTORCH_DETECTION_H
