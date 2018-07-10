#ifndef __TOOLS_HPP__
#define __TOOLS_HPP__

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <limits>

/** @brief Calculate Dark Channel*/
void calcDarkChannel(const cv::Mat_<cv::Vec3b>& src, cv::Mat_<uchar>& dst, const int s = 15);

/** @brief Estimate Atmospheric Light */
void estimateAtmosphericLight(const cv::Mat_<cv::Vec3b>& src, const cv::Mat_<uchar>& dark_channel, cv::Vec3b& A);

/** @brief Estimate Initial Transmission Map*/
void initTransMap(const cv::Mat_<cv::Vec3b>& src, const cv::Vec3b A, cv::Mat_<double>& t, const int s = 15, const double om = 0.95);

/** @brief Recover the Scene Radiance*/
void recoverSceneRadiance(const cv::Mat_<cv::Vec3b>& src, cv::Mat_<cv::Vec3b>& dst, const cv::Mat_<double>& t,
	const cv::Vec3b A, const double t0 = 0.1);
	
/** @brief Solver Linear Equations*/
void linearEquationSolver(cv::SparseMat_<double>& A, cv::Mat_<double>& b, cv::Mat_<double>& X, cv::Size img_size,
    const int w = 3, const double omega = 1.23, const double T = 10e-6, unsigned int N = 10e6);

/** @brief Calculate Mean And Covariance of pixels in input window*/
void meanAndCovariance(const cv::Mat_<cv::Vec3b>& win, cv::Vec3d& m, cv::Mat_<double>& c);

/** @brief Soft Matting*/
void softMatting(const cv::Mat_<cv::Vec3b>& src, cv::Mat_<double>& t_hat, cv::Mat_<double>& t_refine,
    const double lambda = 10e-4, const int w = 3);


#endif