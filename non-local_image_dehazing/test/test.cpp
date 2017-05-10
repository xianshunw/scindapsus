#include "sphere_subdivision.hpp"
#include "kd_tree.hpp"
#include "others.hpp"
#include <iostream>
#include <gsl/gsl_sf.h>

#include <fstream>


int main(int argc, char* argv[])
{
    cv::Mat haze_img = cv::imread(argv[1], 1);

    //subdivide the icosahedron and convert to spherical cooridinates
    icosahedron ic(1.0);
    polyhedron dst;
    subdivide(ic, dst, 500);
    std::vector<cv::Point2d> sph_table;
    spherical_coordinates(dst.vertex_table, sph_table);

    /*--------------test subdivide result-------------------*/
    /********************************************************
    std::ofstream output1("vertex_table.txt"), output2("plane_table.txt");
    if(output1.is_open()&&output2.is_open())
    {
        for(int i = 0; i != dst.vertex_table.size(); ++i)
        {
            output1 << dst.vertex_table[i].x << " "
                    << dst.vertex_table[i].y << " "
                    << dst.vertex_table[i].z << " "
                    << std::endl;
        }

        for(int i = 0; i != dst.plane_table.size(); ++i)
        {
            output2 << dst.plane_table[i][0] << " "
                    << dst.plane_table[i][1] << " "
                    << dst.plane_table[i][2] << std::endl;
        }
    }
    output1.close(); output2.close();
    **********************************************/

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

    //Estimating Initial Transmission and Regularization
    std::vector<double> t_estimate, variance;
    lowBound_variance(cluster_result, img_radius, haze_img, A, t_estimate, variance);

    //show initial transmission
    cv::Mat_<uchar> t_show(haze_img.size());
    for(int i = 0; i != t_show.rows; ++i)
    {
        for(int j = 0; j != t_show.cols; ++j)
        {
            t_show(i, j) = t_estimate[i*t_show.rows + j]*255;
        }
    }
    cv::imshow("t_show", t_show);
    cv::waitKey();

    destory_kdTree(root);
}