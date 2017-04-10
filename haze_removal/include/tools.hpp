#ifndef __TOOLS_HPP__
#define __TOOLS_HPP__

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <limits>

/** @brief Calculate Dark Channel*/
void calcDarkChannel(const cv::Mat_<cv::Vec3b>& src, cv::Mat_<uchar>& dst, const int s = 15);

/** @brief estimate Atmospheric Light */

void estimateAtmosphericLight();


#endif
