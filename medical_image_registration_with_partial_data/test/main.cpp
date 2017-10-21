#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv/cv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <map>
#include <cmath>
#include <vector>
#include <cfloat>
#include <algorithm>
#include <utils.hpp>

void help()
{
    std::cout << "Usage: ./TEST source target" << std::endl; 
}

int main(int argc, char* argv[])
{
   if(argc <= 2)
   {
       help();
       return -1;
   }
   else
   {
       cv::Mat source, target, M, reg;
       float b, c;
       source = cv::imread(argv[1], 0);
       target = cv::imread(argv[2], 0);

       //rotate taget
       cv::Point2f center(target.cols / 2.0, target.rows / 2.0);
       cv::Mat rot = cv::getRotationMatrix2D(center, -30, 1);
       cv::warpAffine(target, target, rot, target.size());

       int iter = 20;
       get_affine_params(source, target, iter, M, b, c);
       reg = affine_warp(source, M);

       //scale
       double max_v, min_v;
       cv::minMaxIdx(reg, &min_v, &max_v);
       reg = (reg - min_v) / (max_v - min_v);
       reg.convertTo(reg, CV_8U, 255);

       cv::Mat canvas(source.rows, 3*source.cols, CV_8UC1, cv::Scalar::all(0));
       source.copyTo(canvas.colRange(0, source.cols));
       target.copyTo(canvas.colRange(source.cols, 2*source.cols));
       reg.copyTo(canvas.colRange(2*source.cols, 3*source.cols));

       cv::cvtColor(canvas, canvas, CV_GRAY2BGR);
       cv::putText(canvas, "source", cv::Point(source.cols / 2, 250), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
       cv::putText(canvas, "target", cv::Point(source.cols / 2 + source.cols, 250), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
       cv::putText(canvas, "reg", cv::Point(source.cols / 2 + 2*source.cols, 250), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
       cv::namedWindow("canvas");
       cv::imshow("canvas", canvas);
       cv::waitKey();
       cv::imwrite("canvas.jpg", canvas);
   }
}
