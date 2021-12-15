//
// Created by zhanggong on 2021/9/2.
//

#ifndef PAOYUAN_DARKNETDET_H
#define PAOYUAN_DARKNETDET_H
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include"Detect_manager.h"
using namespace std;

class DarknetDet:public detector{
public:
    DarknetDet(int w,int h,cv::String modelConfiguration,cv::String modelWeights);
    cv::dnn::Net initnet(cv::String modelConfiguration,cv::String modelWeights);
    int detect(unsigned char *image,vector<DetObjectStr>& boundRect) override;

private:
    vector<cv::String> getOutputsNames(const cv::dnn::Net& net);
    void postProcess(cv::Mat& frame,const vector<cv::Mat>&out,vector<DetObjectStr>&boundRect);
    vector<cv::String >class_names;
    float confThreshold;
    float nmsThreshold;
    int inpWidth;
    int inpHeight;
    int _image_height;
    int _image_width;
    cv::dnn::Net _net;

};


#endif //PAOYUAN_DARKNETDET_H
