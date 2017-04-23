#include "tools.hpp"
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv/cv.hpp>
#include <iostream>

int main(int argc, char* argv[])
{
    std::string image_name;
    if(argc < 2)
        image_name = "test1.jpg";
    else
        image_name = argv[1];

    cv::Mat img = cv::imread(image_name, 1);
    cv::Mat_<uchar> dark_channel;
    calcDarkChannel(img, dark_channel);

    cv::Vec3b A;
    estimateAtmosphericLight(img, dark_channel, A);

    cv::Mat_<double> t, t_refine; cv::Mat t_show, t_refine_show;
    double min_value, max_value;
    initTransMap(img, A, t, 10);
    cv::Mat tMat = t;
    cv::minMaxIdx(t, &min_value, &max_value);
    tMat.convertTo(t_show, CV_8U, 255.0/(max_value-min_value), -255.0*min_value/(max_value- min_value));
    
    softMatting(img, t, t_refine);

    tMat = t_refine;
    cv::minMaxIdx(t_refine, &min_value, &max_value);
    tMat.convertTo(t_refine_show, CV_8U, 255.0/(max_value-min_value), -255.0*min_value/(max_value- min_value));
    
    cv::Mat_<cv::Vec3b> recoverImg;
    recoverSceneRadiance(img, recoverImg, t_refine, A);

    cv::namedWindow("source img");
    cv::imshow("source img", img);

    cv::namedWindow("dark_channel");
    cv::imshow("dark_channel", dark_channel);
    cv::imwrite("dark_channel.jpg", dark_channel);

    cv::namedWindow("transmission_init");
    cv::imshow("transmission_init", t_show);
    cv::imwrite("transmission_init.jpg", t_show);

    cv::namedWindow("transmission_refine");
    cv::imshow("transmission_refine", t_refine_show);
    cv::imwrite("transmission_refine.jpg", t_refine_show);

    cv::namedWindow("recoverImg");
    cv::imshow("recoverImg", recoverImg);
    cv::imwrite("recoverImg.jpg", recoverImg);
    cv::waitKey();

}