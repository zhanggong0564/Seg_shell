//
// Created by zhanggong on 2021/8/2.
//

#include "Video.h"
#include<opencv2/opencv.hpp>
#include <iostream>
//#include "PostProcess.h"
using namespace std;

Video::Video(string video_names,string calib_file):fs(calib_file,cv::FileStorage::READ){
    if(!fs.isOpened()){
        cout<<"标定文件不存在"<<endl;
    }
    fs["image_width"] >> image_width;
    fs["image_height"] >> image_height;
    cv::Size image_size = cv::Size(image_width, image_height);
    fs["cameraMatrix"] >> intrinsic_matrix;
    fs["distCoeffs"] >> distortion_coeffs;
    cv::initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, cv::Mat(),
                                    intrinsic_matrix, image_size, CV_16SC2, map1, map2);
    this->video_name = video_names;
    video.open(video_name);
    this->fps = video.get(cv::CAP_PROP_FPS);
    width = video.get(cv::CAP_PROP_FRAME_WIDTH);
    height = video.get(cv::CAP_PROP_FRAME_HEIGHT);
    if (video.isOpened()){
        cout<<"视频中的图像宽度: "<<width<<endl;
        cout<<"视频中的图像高度: "<<height<<endl;
        cout<<"视频帧率: "<<fps<<endl;
//        cout<<"视频总帧数: "<<video.get(cv::CAP_PROP_FRAME_COUNT)<<endl;
    }
}

bool Video::get_frame(cv::Mat &output) {
    cv::Mat remap_iamge;
    video>>remap_iamge;

    while (remap_iamge.empty()){
        cout<<"rebuild cam"<<endl;
        video.open(video_name);
        video>>remap_iamge;
    }
    remap(remap_iamge, output, map1, map2,
          cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());
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