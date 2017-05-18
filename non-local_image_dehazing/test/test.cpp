#include "sphere_subdivision.hpp"
#include "kd_tree.hpp"
#include "others.hpp"
#include <iostream>

#include <fstream>
#include <sstream>


int main(int argc, char* argv[])
{
    cv::Mat haze_img = cv::imread(argv[1], 1);
    cv::Mat img_scale(haze_img.size(), CV_64FC3);
    for(int i = 0; i != img_scale.rows; ++i)
    {
        for(int j = 0; j != img_scale.cols; ++j)
        {
            cv::Vec3b pix = haze_img.at<cv::Vec3b>(i, j);
            img_scale.at<cv::Vec3d>(i, j) = cv::Vec3d(pix[0]/255.0, pix[1]/255.0, pix[2]/255.0);
        }
    }

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
    cv::Vec3d A; cv::Mat_<double> dark_channel;
    calcDarkChannel(img_scale, dark_channel);
    estimateAtmosphericLight(img_scale, dark_channel, A);

    //find haze lines
    std::vector<cv::Point2d> img_sph; std::vector<std::vector<int>> cluster_result;
    std::vector<double> img_radius;
    spherical_coordinates(img_scale, img_sph, img_radius, A);
    cluster_img(root, sph_table, img_sph, cluster_result);

    //Estimating Initial Transmission and Regularization
    std::vector<double> t_init, stdde, t_refine;
    trans_stdde(cluster_result, img_radius, A, t_init, stdde);
    regular_trans(img_scale, A, t_init, stdde, cluster_result, t_refine);


    //dehaze
    cv::Mat haze_free;
    dehaze(img_scale, A, t_refine, haze_free);
    cv::Mat_<uchar> t_show(haze_img.size());
    for(int i = 0; i != t_refine.size(); ++i)
    {
        int rs = i/haze_img.cols, cs = i%haze_img.cols;
        t_show(rs, cs) = t_refine[i]*255;
    }
    
    cv::imshow("t_show", t_show);
    cv::imshow("dehaze", haze_free);
    cv::waitKey();

    destory_kdTree(root);
}
