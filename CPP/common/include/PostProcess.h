//
// Created by zhanggong on 2021/7/31.
//

#ifndef YOLOV4_PYTORCH_POSTPROCESS_H
#define YOLOV4_PYTORCH_POSTPROCESS_H
#include <iostream>
#include <opencv2/opencv.hpp>
#include "Detect_manager.h"
using namespace std;

struct message{
    double shiftx;
    double shiftv;
    double arm_x;
    double arm_y;
    double arm_theta;
    bool is_ob;
    bool is_up;
};



class PostProcess {
public:
    message get_car_info(vector<DetObjectStr>& boundRect,cv::Mat frame);
    message get_arm_info(cv::Mat &mask);
    vector<vector<cv::Point2f>> get_box();
    vector<cv::Point2f> get_centers();
    void empty_info();
private:
    bool filter_feature();
    vector<vector<cv::Point>> new_contour;
    vector<vector<cv::Point>> m_contour;
    vector<cv::Rect> Rec;
    vector<vector<cv::Point2f>>box;
    vector<cv::Point2f> centers;
    vector<bool>upright;
//    vector<double> angs;
};


#endif //YOLOV4_PYTORCH_POSTPROCESS_H
