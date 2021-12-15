//
// Created by zhanggong on 2021/7/31.
//

#include "../include/PostProcess.h"
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
//对point排序得到离相机最近的点,(height)最大
static cv::Point sorted(vector<cv::Point2f> centers,double &index){
    int max_centery = centers[0].y;
    index = 0;
    for (int i = 1; i <centers.size() ; ++i) {
        if (centers[i].y>max_centery){
            max_centery = centers[i].x;
            index = i;
        }
    }
    return centers[index];
}
cv::Point static getCenterPoint(cv::Rect rect)
{
    cv::Point cpt;
    cpt.x = rect.x + cvRound(rect.width/2.0);
    cpt.y = rect.y + cvRound(rect.height/2.0);
    return cpt;
}

static double distanceBtwPoints(const cv::Point2f &a, const cv::Point2f &b)
{
    double xDiff = a.x - b.x;
    double yDiff = a.y - b.y;

    return std::sqrt((xDiff * xDiff) + (yDiff * yDiff));
}
static double get_angle(vector<cv::Point2f> pts){
    double dist0 = distanceBtwPoints(pts[0], pts[1]);
    double dist1 = distanceBtwPoints(pts[1], pts[2]);
    double angle = 0;
    if (dist0 > dist1){
        angle = atan2(pts[0].y - pts[1].y, pts[0].x - pts[1].x) * 180.0 / CV_PI;
    }else{
        angle = atan2(pts[1].y - pts[2].y, pts[1].x - pts[2].x) * 180.0 / CV_PI;
    }
    if (angle>0){
        angle = 180-angle;
    }else if(angle<0){
        angle =angle+90;
    }
//    angle = 180.-angle+90.;
//    if (angle>=170){
//        angle=0;
//    }
    return angle ;
}
message PostProcess::get_car_info(vector<DetObjectStr>& boundRect,cv::Mat frame){
    int w = frame.cols;
    int h = frame.rows;
    message info;
    cout<<"is empty"<<boundRect.empty()<<endl;
    if (!boundRect.empty()){
        for (int i = 0; i <boundRect.size() ; ++i) {
            cv::Point cpt = getCenterPoint(boundRect[i].rect);
            centers.push_back(cpt);
        }
        double ind;
        cv::Point shiftxy = sorted(centers,ind);
        cout<<shiftxy.x<<endl;
        cout<<shiftxy.y<<endl;
        info.shiftx = ((double)shiftxy.x-(double)(w/2))/(w/2);
        info.shiftv = double(h) -(double )shiftxy.y;
        return info;
    }else{
        info.shiftx = 0.0;
        info.shiftv = 0.0;
        return info;
    }


}


message PostProcess::get_arm_info(cv::Mat &mask){
    int w = mask.cols;
    int h = mask.rows;
    message info;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(20,1),cv::Point(-1,-1));
    cv::morphologyEx(mask,mask,cv::MORPH_OPEN,kernel);
    cv::findContours(mask,m_contour,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
    if (filter_feature()){
        for (int i = 0; i <new_contour.size() ; ++i) {
            cv::RotatedRect rrect = cv::minAreaRect(new_contour[i]);
            cv::Point2f cpt  = rrect.center;
            double epsilon = 0.25*max(Rec[i].width,Rec[i].height);
            vector<cv::Point> approx;
            cv::approxPolyDP(new_contour[i],approx,epsilon, true);
            double scale = Rec[i].width/Rec[i].height;
            if ( scale <= 1.26 and scale >= 0.95){
                upright.push_back(true);
            }else{
                if (approx.size()>3){
                    upright.push_back(true);
                } else{
                    upright.push_back(false);
                }
            }
//            double angle = rrect.angle;
//            angs.push_back(angle);
            centers.push_back(cpt);
        }
        double ind;
        cv::Point shiftxy = sorted(centers,ind);
        info.arm_x = (double)shiftxy.x-(double)(w/2);
        info.arm_y = (double)shiftxy.y-(double)(h/2);
        vector<cv::Point2f> rect =box[ind];
        info.arm_theta = get_angle(rect);
        info.is_ob=true;
        info.is_up = upright[ind];
        return info;
    }else{
        info.shiftx = 0.0;
        info.shiftv = 0.0;
        info.arm_x = 0;
        info.arm_y = 0;
        info.arm_theta = 0;
        info.is_ob = false;
        info.is_up = false;
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
//    angs.clear();
}
vector<cv::Point2f> PostProcess::get_centers() {
    return centers;
}



