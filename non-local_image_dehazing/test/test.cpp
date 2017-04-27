#include "sphere_subdivision.hpp"
#include <iostream>


int main(int argc, char* argv[])
{
    icosahedron ic(1.0);
    polyhedron dst;
    subdivide(ic, dst, 500);

    std::cout << dst.vertex_table.size() << std::endl;
}