#include "sphere_subdivision.hpp"
#include "kd_tree.hpp"
#include <iostream>


int main(int argc, char* argv[])
{
    icosahedron ic(1.0);
    polyhedron dst;
    subdivide(ic, dst, 30);

    std::vector<cv::Point2d> sph_table;
    spherical_coordinates(dst.vertex_table, sph_table);

    for(int i = 0; i != dst.vertex_table.size(); ++i)
    {
        std::cout<<dst.vertex_table[i]<<"   "<<sph_table[i]<<std::endl;
    }

    std::vector<int> vertex_idx(dst.vertex_table.size());
    for(int i = 0; i != dst.vertex_table.size(); ++i)
    {
        vertex_idx[i] = i;
    }

    kd_node* root = build_kdTree(sph_table, nullptr, vertex_idx);
}