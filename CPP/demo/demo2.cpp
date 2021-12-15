//
// Created by zhanggong on 2021/9/2.
//
//
// Created by zhanggong on 2021/7/31.
//

#include<opencv2/opencv.hpp>
#include<iostream>
#include<time.h>
#include "PostProcess.h"
#include"Video.h"
#include <unistd.h>
#include"Detect_manager.h"
#include"DarknetDet.h"
#define _GLIBCXX_USE_CXX11_ABI 0
using namespace std;


int main(){
    string video_name = "/home/zhanggong/disk/Elements/data/paoyuan/source/data3.avi";
    cv::String modelConfiguration ="../model/paoyuan_yolov4tiny.cfg";
    cv::String modelWeights = "../model/paoyuan_yolov4tiny_last.weights";
    string calib_file = "../model/calibration_in_params2.yml";
    Video video(video_name,calib_file);
    cv::Mat frame;
    video.get_frame(frame);
    PostProcess ppro;
    int width = frame.cols;
    int height = frame.rows;
    detector *darknet = new DarknetDet(width,height,modelConfiguration,modelWeights);


    while(1){
//        clock_t start = clock();
        vector< DetObjectStr> Detobject;

        video.get_frame(frame);
        cv::Mat dst = frame.clone();
        double time0 = static_cast<double>(cv::getTickCount());
        darknet->detect(frame.data,Detobject);
        time0 = ((double)cv::getTickCount()-time0)/cv::getTickFrequency();
        cout<<"speed time: "<<time0<<endl;
//        Det.PostProcess(out,masks);//后处理时间0.03s 30ms

        message info;
        info = ppro.get_car_info(Detobject,frame);
        cout<<info.shiftx<<endl;
        cout<<info.shiftv<<endl;
//        video.show_Rect(Detobject,dst);
        for (int i = 0; i < Detobject.size(); i++){
            cv::rectangle(frame,  Detobject[i].rect, CV_RGB(0, 0, 255));
        }
        cv::imshow("det",frame);

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
    delete darknet;
    return 0;

}

