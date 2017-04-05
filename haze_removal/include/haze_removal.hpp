#ifndef __HAZE_REMOVAL_HPP__
#define __HAZE_REMOVAL_HPP__

#include <opencv2/core.hpp>

/** @brief Solver of Linear Equations
This is the solver of linear equations which can be expressed in the
following matrix equation
Lx = b

@param L the coefficient matrix which size is N\timesN
@param b the vector in the right side of above eqution which size is N\time s1
@param x the result of this solver
*/

void solver(const cv::Mat_<float> L, const cv::Mat_<float> b, cv::Mat_<float>& x);

/** @brief Compute The Dark Channel of Input Image
This function is used to compute the dark channel of input image

@param src input 3-channels image
@param s the window's size
@param dst dark channel of the input image
 */

void computeDarkChannel(const cv::Mat_<cv::Vec3b>& src, const cv::Size s, cv::Mat_<uchar>& dst);

#endif
