#include "sphere_subdivision.hpp"
#include "kd_tree.hpp"
#include "others.hpp"
#include <iostream>


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

    //Estimating Initial Transmission and Regularization
    std::vector<double> t_init, variance, t_refine;
    trans_variance(cluster_result, img_radius, A, t_init, variance);
    regular_trans(haze_img, A, t_init, variance, t_refine);

    cv::Mat_<uchar> t_show(haze_img.size());
    for(int i = 0; i != t_refine.size(); ++i)
    {
        int rs = i/haze_img.cols, cs = i%haze_img.cols;
        t_show(rs, cs) = t_refine[i]*255;
    }
    cv::imshow("t_show", t_show);
    cv::waitKey();

    destory_kdTree(root);
}
