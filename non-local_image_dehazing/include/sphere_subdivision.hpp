#ifndef __SPHERE_SUBDIVISION_HPP__
#define __SPHERE_SUBDIVISION_HPP__

#include <opencv2/core.hpp>
#include <opencv/cv.hpp>
#include <vector>
#include <deque>
#include <map>

/** @brief struct to store polyhedron*/
struct polyhedron
{
    std::vector<cv::Point3d> vertex_table;
    std::deque<std::vector<int>> plane_table;
};

/** @brief struct to store icosahedron*/
struct icosahedron
{
    icosahedron() { set(1.0); }
    icosahedron(double r) { set(r); }


    void set(double r);

    polyhedron i;
    double radius;
};

/** @brief check if the mid-point of two points has calculated*/
bool is_present(int idx1, int idx2, std::map<std::vector<int>, int>& mid_table, int& mid_idx);

/** @brief scale the vector to unit*/
void scale2unit(cv::Point3d& pt, double unit = 1.0);

/** @brief  tessellate a sphere to give a geodesic sphere*/
void subdivide(icosahedron& src, polyhedron& dst, int num = 500);

#endif
