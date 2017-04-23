#include "sphere_subdivision.hpp"
#include <opencv2/core.hpp>
#include <opencv/cv.hpp>
#include <cmath>

void icosahedron::set(double r)
{
    i.vertex_table.clear();
    i.plane_table.clear();

    double z = r*(sqrt(5.0) + 1.0)/2;
    i.vertex_table.emplace_back(-r, 0.0, z);
    i.vertex_table.emplace_back(r, 0.0, z);
    i.vertex_table.emplace_back(-r, 0.0, -z);
    i.vertex_table.emplace_back(r, 0.0, -z);
    i.vertex_table.emplace_back(0.0, z, r);
    i.vertex_table.emplace_back(0.0, z, -r);
    i.vertex_table.emplace_back(0.0, -z, r);
    i.vertex_table.emplace_back(0.0, -z, -r);
    i.vertex_table.emplace_back(z, r, 0.0);
    i.vertex_table.emplace_back(-z, r, 0.0);
    i.vertex_table.emplace_back(z, -r, 0.0);
    i.vertex_table.emplace_back(-z, -r, 0.0);

    
}
