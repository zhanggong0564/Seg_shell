//
// Created by zhanggong on 2021/8/2.
//

#ifndef PAOYUAN_VIDEO_H
#define PAOYUAN_VIDEO_H
#include<opencv2/opencv.hpp>
#include <iostream>
#include"PostProcess.h"
using namespace std;



class Video {
public:
    Video(string video_name,string calib_file);
    bool get_frame(cv::Mat &output);
    int get_fps();
    void show(cv::Mat bgr,cv::Mat mask,PostProcess P);
    int height;
    int width;



private:
    cv::VideoCapture video;
    int fps;
    string video_name;
    cv::FileStorage fs;
    cv::Mat map1, map2;
    cv::Mat intrinsic_matrix, distortion_coeffs;
    int image_width{0}, image_height{0};
};


#endif //PAOYUAN_VIDEO_H
