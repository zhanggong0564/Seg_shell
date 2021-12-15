//
// Created by zhanggong on 2021/7/31.
//

//#include"segmentation.h"
#include"DetTrt.h"
#include<opencv2/opencv.hpp>
#include<iostream>
#include<time.h>
#include "PostProcess.h"
#include"Video.h"
#include <unistd.h>
using namespace std;


int main(){
    int width= 640;
    int height= 360;
//    string model_name = "/home/zhanggong/disk/Elements/ubuntu18.04/paoyuan/seg_src/tool/FCN8S 1_3_180_360_static.onnx";
//    string model_name = "/home/zhanggong/disk/Elements/ubuntu18.04/paoyuan/seg_src/tool/export_360.trt";
    string model_name = "/home/zhanggong/disk/Elements/ubuntu18.04/paoyuan/seg_src/tool/new_export.trt";
    string video_name = "/home/zhanggong/disk/Elements/data/paoyuan/source/data3.avi";
    const char* INPUT_BLOB_NAME = "input";
    const char* OUTPUT_BLOB_NAME = "score";
    string calib_file ="../model/calibration_in_params2.yml";
//    string video_name = "rtsp://admin:hf62843295@192.168.166.7/Streaming/Channels/101";

//    segmentation Det(width,height,model_name,true);
    input_init in_{model_name,INPUT_BLOB_NAME,OUTPUT_BLOB_NAME,3,height,width,4};
    DetTrt Det(in_);
//    Det_onnx Det(in_);
    PostProcess ppro;
    Video video(video_name,calib_file);
//    cv::Mat img = cv::imread("/home/zhanggong/disk/Elements/ubuntu18.04/paoyuan/data/imgs/33.jpg");
    cv::Mat masks(height,width,CV_8UC1,cv::Scalar(0));
    cv::Mat frame(height,width,CV_8UC3,cv::Scalar(0,0,0));

    while(1){
//        clock_t start = clock();

//        video.get_frame(frame);
        cv::Mat frame = cv::imread("/home/zhanggong/disk/Elements/ubuntu18.04/paoyuan/paoyuan/shell/6.jpg");
        double time0 = static_cast<double>(cv::getTickCount());
        Det.Detecting(frame,masks);//检测时间0.006秒 6ms

        cv::imshow("mask",masks);
        time0 = ((double)cv::getTickCount()-time0)/cv::getTickFrequency();
        cout<<"speed time: "<<time0<<endl;
//        Det.PostProcess(out,masks);//后处理时间0.03s 30ms

        message info;
        info = ppro.get_arm_info(masks);
//        cout<<info.arm_x<<endl;
//        cout<<info.arm_y<<endl;
        cout<<info.is_up<<endl;
        cout<<"arm_theta"<<info.arm_theta<<endl;
        cv::resize(frame,frame,cv::Size(width,height));
        video.show(frame,masks,ppro);

        ppro.empty_info();

        int k = cv::waitKey(video.get_fps());
        if (k==27){
            break;
        }
//        clock_t end = clock();
//        double dur = (double)(end - start);

//        cout<<"speed time: "<<dur/CLOCKS_PER_SEC<<endl;

    }
    cv::destroyAllWindows();
    return 0;

}