#include "sphere_subdivision.hpp"
#include "kd_tree.hpp"
#include "others.hpp"
#include <iostream>
#include <cmath>
#include <gsl/gsl_sf.h>

#include <fstream>
#include <limits>


int main(int argc, char* argv[])
{
    cv::Mat haze_img = cv::imread(argv[1], 1);

    //subdivide the icosahedron and convert to spherical cooridinates
    icosahedron ic(1.0);
    polyhedron dst;
    subdivide(ic, dst, 1000);
    std::vector<cv::Point2d> sph_table;
    spherical_coordinates(dst.vertex_table, sph_table);

    //build kd tree
    std::vector<int> vertex_idx(dst.vertex_table.size());
    for(int i = 0; i != dst.vertex_table.size(); ++i)
    {
        vertex_idx[i] = i;
    }
    kd_node* root = build_kdTree(sph_table, nullptr, vertex_idx);

    //estimate atmospheric light
    cv::Vec3d A; cv::Mat_<uchar> dark_channel;
    calcDarkChannel(haze_img, dark_channel);
    estimateAtmosphericLight(haze_img, dark_channel, A);

    //find haze lines
    std::vector<cv::Point2d> img_sph; std::vector<std::vector<int>> cluster_result;
    std::vector<double> img_radius;
    spherical_coordinates(haze_img, img_sph, img_radius, A);
    cluster_img(root, sph_table, img_sph, cluster_result);

    /****************************************************
    cv::Vec3b color[] = {cv::Vec3b(0, 0, 255), cv::Vec3b(255, 0, 255), cv::Vec3b(255, 0, 0)};
    int num = 0;
    for(int i = 0; i != cluster_result.size(); ++i)
    {
        if(cluster_result[i].size() < 1000) continue;
        for(int k = 0; k != cluster_result[i].size(); ++k)
        {
            int rs = cluster_result[i][k]/haze_img.cols, cs = cluster_result[i][k]%haze_img.cols;
            haze_img.at<cv::Vec3b>(rs, cs) = color[num%3];
        }
        num += 1;
        
        cv::imshow("t_show", haze_img);
        cv::waitKey();
    }
    ****************************************************/

    //Estimating Initial Transmission and Regularization
    std::vector<double> t_estimate, variance;
    lowBound_variance(cluster_result, img_radius, haze_img, A, t_estimate, variance);


    double t_max =  *std::max_element(t_estimate.begin(), t_estimate.end()),
        t_min = *std::min_element(t_estimate.begin(), t_estimate.end());
    cv::Mat_<uchar> t_show(haze_img.size());
    for(int i = 0; i != t_estimate.size(); ++i)
    {
        int rs = i/haze_img.cols, cs = i%haze_img.cols;
        t_show(rs, cs) = t_estimate[i]*255;
    }
    cv::imshow("t_show", t_show);
    cv::waitKey();

    destory_kdTree(root);
}