//
// Created by zhanggong on 2021/9/2.
//

#ifndef PAOYUAN_DETECT_MANAGER_H
#define PAOYUAN_DETECT_MANAGER_H
#include<opencv2/opencv.hpp>
#include<iostream>
using namespace std;
struct DetObjectStr
{
    cv::Rect_<float> rect;
    int label;
    float score;
};

class detector{
public:
    virtual int detect(unsigned char *image,vector<DetObjectStr>& boundRect)=0;
};

#endif //PAOYUAN_DETECT_MANAGER_H
