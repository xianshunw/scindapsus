#ifndef __OTHERS_HPP__
#define __OTHERS_HPP__

#include <opencv2/core.hpp>
#include <vector>
#include "kd_tree.hpp"

/** @brief Calculate Dark Channel*/
void calcDarkChannel(const cv::Mat_<cv::Vec3d>& src, cv::Mat_<double>& dst, const int s = 15);

void estimateAtmosphericLight(const cv::Mat_<cv::Vec3d>& src, 
    const cv::Mat_<double>& dark_channel, cv::Vec3d& A);

/** @brief convert cartesian coordinates to spherical coordinates and keep longitude and latitude*/
void spherical_coordinates(const std::vector<cv::Point3d>& vertex_table, std::vector<cv::Point2d>& sph_table);

/** @brief overload spherical_coordinates*/
void spherical_coordinates(const cv::Mat& img, std::vector<cv::Point2d>& img_sph,
    std::vector<double>& r, cv::Vec3d& A);

/** @brief clustering the haze image*/
void cluster_img(kd_node* root, const std::vector<cv::Point2d>& sph_table,
    const std::vector<cv::Point2d>& img_sph, std::vector<std::vector<int>>& cluster_result);

/** @brief estimate initial transmission and variance*/
void trans_stdde(const std::vector<std::vector<int>>& cluster_result, const std::vector<double>& r,
    const cv::Vec3d& A, std::vector<double>& t_init, std::vector<double>& stdde);

/** @brief transmission regularization*/
void regular_trans(const cv::Mat& img, const cv::Vec3d& A, std::vector<double>& t_init,
    std::vector<double>& stdde, const std::vector<std::vector<int>>& cluster_result,
    std::vector<double>& t_refine, double lambda = 0.1);

void dehaze(cv::Mat& img_scale, cv::Vec3d A, std::vector<double>& t, cv::Mat& haze_free);

#endif