#include "tools.hpp"
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv/cv.hpp>

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

    /*
    cv::Vec3b A;
    estimateAtmosphericLight(img, dark_channel, A);

    cv::Mat t, t_show;
    double min_value, max_value;
    initTransMap(img, A, t);
    cv::minMaxIdx(t, &min_value, &max_value);
    t.convertTo(t_show, CV_8U, 255.0/(max_value-min_value), 255.0*min_value/(max_value- min_value));
    
    cv::Mat_<cv::Vec3b> recoverImg;
    recoverSceneRadiance(img, t, A, recoverImg);
    
    cv::namedWindow("source img");
    cv::imshow("source img", img);
    cv::namedWindow("dark_channel");
    cv::imshow("dark_channel", dark_channel);
    cv::namedWindow("transmission");
    cv::imshow("transmission", t_show);
    cv::namedWindow("recoverImg");
    cv::imshow("recoverImg", recoverImg);
    cv::waitKey();
    */
}
