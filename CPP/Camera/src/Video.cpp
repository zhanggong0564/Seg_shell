//
// Created by zhanggong on 2021/8/2.
//

#include "../include/Video.h"
#include<opencv2/opencv.hpp>
#include <iostream>
#include "PostProcess.h"
using namespace std;

Video::Video(string video_names){
    this->video_name = video_names;
    video.open(video_name);
    this->fps = video.get(cv::CAP_PROP_FPS);
    if (video.isOpened()){
        cout<<"视频中的图像宽度: "<<video.get(cv::CAP_PROP_FRAME_WIDTH)<<endl;
        cout<<"视频中的图像高度: "<<video.get(cv::CAP_PROP_FRAME_HEIGHT)<<endl;
        cout<<"视频帧率: "<<fps<<endl;
//        cout<<"视频总帧数: "<<video.get(cv::CAP_PROP_FRAME_COUNT)<<endl;
    }
}

bool Video::get_frame(cv::Mat &output) {
    video>>output;
    while (output.empty()){
        cout<<"rebuild cam"<<endl;
        video.open(video_name);
        video>>output;
    }
    return true;
}
int Video::get_fps() {
    return fps;
}
void Video::show(cv::Mat bgr,cv::Mat mask,PostProcess P ) {
    cv::Mat masks[3];
    cv::Mat merge_image,result;
    for (int i = 0; i <3 ; ++i) {
        masks[i] = mask;
    }
    vector<vector<cv::Point2f>> box = P.get_box();
    vector<cv::Point2f> centers =  P.get_centers();
    for (int i = 0; i <box.size() ; ++i) {
        for (int j = 0; j <box[i].size() ; ++j) {
            if (j==3){
                cv::line(bgr,box[i][j],box[i][0],cv::Scalar(0,255,0),1,cv::LINE_AA,0);
                break;
            }
            line(bgr,box[i][j],box[i][j+1],cv::Scalar(0,255,0),1,cv::LINE_AA,0);
        }
        cv::circle(bgr,centers[i],2,cv::Scalar(0,255,0),1,cv::LINE_AA,0);
    }
    cv::merge(masks,3,merge_image);
    cv::addWeighted(bgr,0.7,merge_image,0.3,0,bgr);
    cv::hconcat(bgr,merge_image,result);
    cv::imshow("concat",result.clone());
}