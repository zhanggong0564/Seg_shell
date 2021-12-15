//
// Created by zhanggong on 2021/9/2.
//

#include "DarknetDet.h"
#include<iostream>
using namespace std;
DarknetDet::DarknetDet(int w ,int h,cv::String modelConfiguration,cv::String modelWeights){
    confThreshold = 0.5;
    nmsThreshold = 0.4;
    inpWidth =416;
    inpHeight =416;
    _image_width = w;
    _image_height = h;
    _net = initnet(modelConfiguration,modelWeights);
}
cv::dnn::Net DarknetDet::initnet(cv::String modelConfiguration,cv::String modelWeights) {
    string class_names_string = "../../model/coco.names";
    ifstream class_name_file(class_names_string);
    if (class_name_file.is_open()){
        string name="";
        while(getline(class_name_file,name)){
            class_names.push_back(name);
        }
    }else{
        cout<<"don't open class_names_file,please check file path"<<endl;
    }
//    cv::String modelConfiguration = "../model/paoyuan_yolov4tiny.cfg";
//    cv::String modelWeights = "../model/paoyuan_yolov4tiny_last.weights";

    cv::dnn::Net net = cv::dnn::readNetFromDarknet(modelConfiguration,modelWeights);
    std::cout << "Read Darknet..." << std::endl;

    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);  //DNN_TARGET_OPENCL_FP16

//    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
//    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    return net;
}

int DarknetDet::detect(unsigned char *image, vector<DetObjectStr> &boundRect) {
    cv::Mat frame = cv::Mat(_image_height,_image_width,CV_8UC3,image,0);
    cv::Mat blob;
    blob = cv::dnn::blobFromImage(frame,1/255.0,cv::Size(inpWidth,inpHeight),cv::Scalar(0,0,0),true,false);
    _net.setInput(blob);
    vector<cv::Mat> outs;
    _net.forward(outs,getOutputsNames(_net));
    postProcess(frame,outs,boundRect);
    std::cout << "succeed!!!" << std::endl;
}

vector<cv::String> DarknetDet::getOutputsNames(const cv::dnn::Net &net) {
    static vector<cv::String> names;
    if(names.empty()){
        vector<int> out_layer_index = net.getUnconnectedOutLayers();
        vector<cv::String>all_layers_names = net.getLayerNames();
        names.resize(out_layer_index.size());
        for (int i = 0; i <out_layer_index.size() ; ++i) {
            names[i] = all_layers_names[out_layer_index[i]-1];
        }
    }
    return names;
}

void DarknetDet::postProcess(cv::Mat &frame, const vector<cv::Mat> &out, vector<DetObjectStr> &boundRect) {
    vector<float>confidences;
    vector<cv::Rect>boxes;
    vector<int> classIds;
    for (int num = 0; num <out.size() ; ++num) {
        double value;
        cv::Point Position;
        float *data = (float *)out[num].data;
        for (int i = 0; i <out[num].rows ; ++i,data+=out[num].cols) {
            cv::Mat sorces = out[num].row(i).colRange(5, out[num].cols);//第几行n*9 9表示cx,cy w,h,conf,classes
            minMaxLoc(sorces,0,&value,0,&Position);
            float conf = data[4];
            if (conf>confThreshold){
                int center_x = (int)(data[0] * frame.cols);
                int center_y = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int box_x = center_x - width / 2;
                int box_y = center_y - height / 2;

                classIds.push_back(Position.x);
                confidences.push_back((float)value);
                boxes.push_back(cv::Rect(box_x, box_y, width, height));
            }

        }
    }
    std::vector<int> perfect_indx;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, perfect_indx);

    for (int i = 0; i < perfect_indx.size(); i++)
    {
        DetObjectStr dout;
        int idx = perfect_indx[i];
        float conf = confidences[idx];
        dout.rect= boxes[idx];
        dout.score = confidences[idx];
        dout.label = classIds[idx];
        boundRect.push_back(dout);
    }
}