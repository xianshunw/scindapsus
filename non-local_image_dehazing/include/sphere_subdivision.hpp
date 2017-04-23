#ifndef __SPHERE_SUBDIVISION_HPP__
#define __SPHERE_SUBDIVISION_HPP__

#include <opencv2/core.hpp>
#include <opencv/cv.hpp>
#include <vector>
#include <utility>

struct polyhedron
{
    std::vector<cv::Point3d> vertex_table;
    std::vector<std::vector<int>> plane_table;
};


struct icosahedron
{
    icosahedron() { set(1.0); }
    icosahedron(double r) { set(r); }

    void set(double r);

    polyhedron i;
};

#endif
