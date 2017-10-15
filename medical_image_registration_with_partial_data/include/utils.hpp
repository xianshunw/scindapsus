/*
 * Utility functions
 */

#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv/cv.hpp>
#include <cmath>
#include <vector>

struct affine_params
{
    cv::Mat fx, fy, ft, mask, nf, S, D, mout, pt, kt;
    int w, h;
};

struct RParams
{
    cv::Mat M, pt, kt;
    float c, b, r;
};

struct TempStatic
{
    cv::Mat mx, my, H, mout_def, minus_one;
};

void diffxyt(cv::Mat& frame1, cv::Mat& frame2, std::vector<cv::Mat>& fxyt);

void affine_find(affine_params& params, TempStatic& T);

void get_affine_params(cv::Mat& img1, cv::Mat& img2, int iters, cv::Mat& M, float& bnew, float& cnew);

RParams affine_find_api(cv::Mat& img1, cv::Mat& img2, cv::Mat& mask, TempStatic& T);

void affine_iter(cv::Mat& img1, cv::Mat& img2, int iters);

cv::Mat mask_compute(cv::Mat& img1, cv::Mat& img2, float sigma);

void reduce(cv::Mat& img, cv::Mat& output);

void interp2(cv::Mat& X, cv::Mat& Y, cv::Mat& V, cv::Mat& XI, cv::Mat& YI, cv::Mat& Z);

cv::Mat affine_warp(cv::Mat& img, cv::Mat& M);

#endif
