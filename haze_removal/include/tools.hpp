#ifndef __TOOLS_HPP__
#define __TOOLS_HPP__

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <limits>

/** @brief Calculate Dark Channel*/
void calcDarkChannel(const cv::Mat_<cv::Vec3b>& src, cv::Mat_<uchar>& dst, const int s = 15);

/** @brief estimate Atmospheric Light */
void estimateAtmosphericLight(const cv::Mat_<cv::Vec3b>& src, const cv::Mat_<uchar>& dark_channel, cv::Vec3b& A);

/** @brief Estimate Initial Transmission Map*/
void initTransMap(const cv::Mat_<cv::Vec3b>& src, const cv::Vec3b A, cv::Mat& t, const int s = 15, const float om = 0.95);

#endif
