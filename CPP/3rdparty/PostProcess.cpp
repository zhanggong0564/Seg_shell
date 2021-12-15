//
// Created by zhanggong on 2021/7/31.
//

#include "PostProcess.h"
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
//对point排序得到离相机最近的点
static cv::Point sorted(vector<cv::Point2f> centers){
    int max_centery = centers[0].y;
    int index = 0;
    for (int i = 1; i <centers.size() ; ++i) {
        if (centers[i].y>max_centery){
            max_centery = centers[i].x;
            index = i;
        }
    }
    return centers[index];
}
message PostProcess::get_info(cv::Mat mask){
    int w = mask.cols;
    int h = mask.rows;
    cv::findContours(mask,m_contour,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
    if (filter_feature()){
        for (int i = 0; i <new_contour.size() ; ++i) {
            cv::RotatedRect rrect = cv::minAreaRect(new_contour[i]);
            cv::Point2f cpt  = rrect.center;
//            cv::Moments M;
//            M = cv::moments(new_contour[i],true);
//            cv::Point center;
//            center.x = M.m10/M.m00;
//            center.y = M.m01/M.m00;
            centers.push_back(cpt);
        }
        cv::Point shiftxy = sorted(centers);
        info.shiftx = ((double)shiftxy.x-(double)(w/2))/(w/2);
        info.shiftv = double(h) -(double )shiftxy.y;
        return info;
    }else{
        info.shiftx = 0.0;
        info.shiftv = 0.0;
        return info;
    }

}
 bool PostProcess::filter_feature() {
    for (int i = 0; i <m_contour.size() ; ++i) {
        double contours_area = cv::contourArea(m_contour[i]);
        cv::Rect rect = cv::boundingRect(m_contour[i]);
        cv::RotatedRect rrect = cv::minAreaRect(m_contour[i]);
        double length = cv::arcLength(m_contour[i],true);
//        vector<cv::Point2f> p;

//        vector<int>rect_points;
        cv::Point2f rect_points[4];
        vector<cv::Point2f> p;
//        cv::boxPoints(rrect,rect_points);//
        rrect.points(rect_points);
        for (int j = 0; j < 4; ++j) {
            p.push_back(rect_points[j]);
        }
        double scale_w_h =(double)rect.width/(double)rect.height;
//        cout<<"w: "<<rect.width<<"h"<<rect.height<<endl;
//        cout<<"scale_w_h: "<<scale_w_h<<"contours_area :"<<contours_area<<"length :"<<length<<endl;
        if(scale_w_h>0.3 && scale_w_h<4.9 && 25 < contours_area && contours_area < 29556 &&length>19 && length<1344 ){
            new_contour.push_back(m_contour[i]);
            Rec.push_back(rect);
            box.push_back(p);
        }
    }
     if(!new_contour.empty()){
         return true;
     }else{
//         cout<<"empty contour"<<endl;
         return false;
     }
}
vector<vector<cv::Point2f>> PostProcess::get_box() {
    return box;
}
void PostProcess::empty_info() {
    new_contour.clear();
    Rec.clear();
    box.clear();
    m_contour.clear();
    centers.clear();
}
vector<cv::Point2f> PostProcess::get_centers() {
    return centers;
}
