//
// Created by zhanggong on 2021/7/31.
//

#ifndef YOLOV4_PYTORCH_POSTPROCESS_H
#define YOLOV4_PYTORCH_POSTPROCESS_H
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;

struct message{
    double shiftx;
    double shiftv;
};



class PostProcess {
public:
    message get_info(cv::Mat mask);
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
    message info;
};


#endif //YOLOV4_PYTORCH_POSTPROCESS_H
