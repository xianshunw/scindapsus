#include "sphere_subdivision.hpp"
#include <opencv2/core.hpp>
#include <opencv/cv.hpp>
#include <cmath>

void icosahedron::set(double r)
{
    i.vertex_table.clear();
    i.plane_table.clear();
    radius = r;

    double x = 1.0, z = x*(std::sqrt(5.0) + 1.0)/2, m = std::sqrt(x*x + z*z);
    x = r*x/m; z = r*z/m;
    i.vertex_table.emplace_back(-x, 0.0, z);
    i.vertex_table.emplace_back(x, 0.0, z);
    i.vertex_table.emplace_back(-x, 0.0, -z);
    i.vertex_table.emplace_back(x, 0.0, -z);
    i.vertex_table.emplace_back(0.0, z, x);
    i.vertex_table.emplace_back(0.0, z, -x);
    i.vertex_table.emplace_back(0.0, -z, x);
    i.vertex_table.emplace_back(0.0, -z, -x);
    i.vertex_table.emplace_back(z, x, 0.0);
    i.vertex_table.emplace_back(-z, x, 0.0);
    i.vertex_table.emplace_back(z, -x, 0.0);
    i.vertex_table.emplace_back(-z, -x, 0.0);
	
    std::vector<int> t = {0,1,6}; i.plane_table.push_back(t);
    t[0] = 0; t[1] = 1; t[2] = 4; i.plane_table.push_back(t);
    t[0] = 4; t[1] = 8; t[2] = 1; i.plane_table.push_back(t);
    t[0] = 10; t[1] = 1; t[2] = 8; i.plane_table.push_back(t);
    t[0] = 10; t[1] = 1; t[2] = 6; i.plane_table.push_back(t);
    t[0] = 9; t[1] = 0; t[2] = 4; i.plane_table.push_back(t);
    t[0] = 0; t[1] = 11; t[2] = 9; i.plane_table.push_back(t);
    t[0] = 0; t[1] = 11; t[2] = 6; i.plane_table.push_back(t);
    t[0] = 4; t[1] = 9; t[2] = 5; i.plane_table.push_back(t);
    t[0] = 4; t[1] = 8; t[2] = 5; i.plane_table.push_back(t);
    t[0] = 6; t[1] = 10; t[2] = 7; i.plane_table.push_back(t);
    t[0] = 6; t[1] = 11; t[2] = 7; i.plane_table.push_back(t);
    t[0] = 11; t[1] = 7; t[2] = 2; i.plane_table.push_back(t);
    t[0] = 10; t[1] = 7; t[2] = 3; i.plane_table.push_back(t);
    t[0] = 7; t[1] = 2; t[2] = 3; i.plane_table.push_back(t);
    t[0] = 8; t[1] = 5; t[2] = 3; i.plane_table.push_back(t);
    t[0] = 9; t[1] = 5; t[2] = 2; i.plane_table.push_back(t);
    t[0] = 5; t[1] = 3; t[2] = 2; i.plane_table.push_back(t);
    t[0] = 9; t[1] = 11; t[2] = 2; i.plane_table.push_back(t);
    t[0] = 10; t[1] = 8; t[2] = 3; i.plane_table.push_back(t);
}

bool is_present(int idx1, int idx2, std::map<std::vector<int>, int>& mid_table, int& mid_idx)
{
    std::vector<int> pt_pair1 = { idx1, idx2 }, pt_pair2 = { idx2, idx1 };
    if(mid_table.find(pt_pair1) != mid_table.end())
    {
        mid_idx = mid_table.find(pt_pair1)->second;
        return true;
    }

    if(mid_table.find(pt_pair2) != mid_table.end())
    {
        mid_idx = mid_table.find(pt_pair2)->second;
        return true;
    }

    return false;
}

void scale2unit(cv::Point3d& pt, double unit)
{
    double m = std::sqrt(pt.x*pt.x + pt.y*pt.y + pt.z*pt.z);

    if(m < 10e-15) return;

    pt.x = pt.x*unit/m; pt.y = pt.y*unit/m; pt.z = pt.z*unit/m;
}

void subdivide(icosahedron& src, polyhedron& dst, int num)
{
    dst = src.i; double r = src.radius;
	
    while(dst.vertex_table.size() < num)
    {
        std::map<std::vector<int>, int> mid_table;
        const int size = dst.plane_table.size();
        for(int i = 0; i != size; ++i)
        {
            auto f = *(dst.plane_table.begin());
            dst.plane_table.pop_front();
            int mid_idx12, mid_idx13, mid_idx23;

            if(!is_present(f[0], f[1], mid_table, mid_idx12))
            {
                cv::Point3d pt1 = dst.vertex_table[f[0]], pt2 = dst.vertex_table[f[1]];
                dst.vertex_table.emplace_back((pt1.x + pt2.x)/2, (pt1.y + pt2.y)/2, (pt1.z + pt2.z)/2);
                mid_idx12 = static_cast<int>(dst.vertex_table.size()) - 1;
                scale2unit(dst.vertex_table[mid_idx12], r);
                std::vector<int> temp = { f[0], f[1] };
                mid_table[temp] = mid_idx12;
            }

            if(!is_present(f[0], f[2], mid_table, mid_idx13))
            {
                cv::Point3d pt1 = dst.vertex_table[f[0]], pt2 = dst.vertex_table[f[2]];
                dst.vertex_table.emplace_back((pt1.x + pt2.x)/2, (pt1.y + pt2.y)/2, (pt1.z + pt2.z)/2);
                mid_idx13 = static_cast<int>(dst.vertex_table.size()) - 1;
                scale2unit(dst.vertex_table[mid_idx13], r);
                std::vector<int> temp = { f[0], f[2] };
                mid_table[temp] = mid_idx13;
            }

            if(!is_present(f[1], f[2], mid_table, mid_idx23))
            {
                cv::Point3d pt1 = dst.vertex_table[f[1]], pt2 = dst.vertex_table[f[2]];
                dst.vertex_table.emplace_back((pt1.x + pt2.x)/2, (pt1.y + pt2.y)/2, (pt1.z + pt2.z)/2);
                mid_idx23 = static_cast<int>(dst.vertex_table.size()) - 1;
                scale2unit(dst.vertex_table[mid_idx23], r);
                std::vector<int> temp = { f[1], f[2] };
                mid_table[temp] = mid_idx23;
            }
			
            std::vector<int> t = { f[0], mid_idx12, mid_idx13 };
            dst.plane_table.push_back(t);
            t[0] = mid_idx23;
            dst.plane_table.push_back(t);
            t[2] = f[1];
            dst.plane_table.push_back(t);
            t[1] = mid_idx13; t[2] = f[2];
            dst.plane_table.push_back(t);
        }
    }
}