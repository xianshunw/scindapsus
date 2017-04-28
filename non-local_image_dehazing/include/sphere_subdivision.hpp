#ifndef __SPHERE_SUBDIVISION_HPP__
#define __SPHERE_SUBDIVISION_HPP__

#include <opencv2/core.hpp>
#include <opencv/cv.hpp>
#include <vector>
#include <map>
#include <deque>

struct polyhedron
{
    std::vector<cv::Point3d> vertex_table;
    std::deque<std::vector<int>> plane_table;
};


struct icosahedron
{
    icosahedron() { set(1.0); }
    icosahedron(double r) { set(r); }

    void set(double r);

    polyhedron i;
    double radius;
};


struct kd_node
{
    int data;
    int dimension;
    kd_node *left, *right, *parent;
};

int dimension_choice(std::vector<cv::Point2d>& sph_table, std::vector<int>& subset);

int split(std::vector<cv::Point2d>& sph_table, std::vector<int> subset,
    std::vector<std::vector<int>>& split_subsets, int dimension);

kd_node* build_kdTree(std::vector<cv::Point2d>& sph_table, kd_node* p, std::vector<int> subset);

bool is_present(int idx1, int idx2, std::map<std::vector<int>, int>& mid_table, int& mid_idx);

void scale2unit(cv::Point3d& pt, double unit = 1.0);

void subdivide(icosahedron& src, polyhedron& dst, int num = 500);

void spherical_coordinates(std::vector<cv::Point3d>& vertex_table, std::vector<cv::Point2d>& sph_table);

#endif
