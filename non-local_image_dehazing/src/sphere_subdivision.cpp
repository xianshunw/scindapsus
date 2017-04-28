#include "sphere_subdivision.hpp"
#include <opencv2/core.hpp>
#include <opencv/cv.hpp>
#include <cmath>
#include <map>


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

int dimension_choice(std::vector<cv::Point2d>& sph_table, std::vector<int>& subset)
{
    double mean_d1 = 0.0, mean_d2 = 0.0;
    for(int i = 0; i != subset.size(); ++i)
    {
        mean_d1 += sph_table[subset[i]].x;
        mean_d2 += sph_table[subset[i]].y;
    }
    mean_d1 /= sph_table.size();
    mean_d2 /= sph_table.size();

    double square_error1 = 0.0, square_error2 = 0.0;
    for(int i = 0; i != subset.size(); ++i)
    {
        square_error1 += (sph_table[subset[i]].x - mean_d1)*(sph_table[subset[i]].x - mean_d1);
        square_error2 += (sph_table[subset[i]].y - mean_d2)*(sph_table[subset[i]].y - mean_d2);
    }

    return square_error1 > square_error2 ? 0 : 1;
}

int split(std::vector<cv::Point2d>& sph_table, std::vector<int> subset,
    std::vector<std::vector<int>>& split_subsets, int dimension)
{
    double *value = new double[sph_table.size()];
    for(int i = 0; i != subset.size(); ++i)
    {
        value[i] = dimension == 0 ? sph_table[subset[i]].x : sph_table[subset[i]].y;
    }

    for(int i = 1; i != subset.size(); ++i)
    {
        for(int j = 0; j != subset.size() - i; ++j)
        {
            if(value[j] > value[j + 1])
            {
                double t1 = value[j];
                value[j] = value[j + 1]; value[j + 1] = t1;
                int t2 = subset[j];
                subset[j] = subset[j + 1]; subset[j + 1] = t2;
            }
        }
    }
    delete[] value;

    int r = subset.size()/2; split_subsets.clear();
    std::vector<int> subset1, subset2;
    for(auto iter = subset.cbegin(); iter != subset.cbegin() + r; ++iter)
    {
        subset1.push_back(*iter);
    }
    for(auto iter = subset.cbegin() + r + 1; iter != subset.cend(); ++iter)
    {
        subset2.push_back(*iter);
    }
    split_subsets.push_back(subset1); split_subsets.push_back(subset2);

    return r;
}

kd_node* build_kdTree(std::vector<cv::Point2d>& sph_table, kd_node* p, std::vector<int> subset)
{
    kd_node* r = new kd_node; r->parent = p;
    if(subset.size() == 1)
    {
        r->data = subset[0];
        r->dimension = 0;
        r->left = r->right = nullptr;
        return r;
    }

    std::vector<std::vector<int>> subsets;
    int d = dimension_choice(sph_table, subset);
    r->data = split(sph_table, subset, subsets, d);


    r->left = subsets[0].size() != 0 ? build_kdTree(sph_table, r, subsets[0]) : nullptr;
    r->right = subsets[1].size() != 0 ? build_kdTree(sph_table, r, subsets[1]) : nullptr;

    return r;
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

void spherical_coordinates(std::vector<cv::Point3d>& vertex_table, std::vector<cv::Point2d>& sph_table)
{
	sph_table.clear();
    for(int i = 0; i != vertex_table.size(); ++i)
    {
        double l = std::sqrt(vertex_table[i].x*vertex_table[i].x + vertex_table[i].y*vertex_table[i].y), 
            theta, phi;
        if(vertex_table[i].y > 0.0)
        {
            theta = std::acos(vertex_table[i].x/l);
        }

        if(vertex_table[i].y < 0.0)
        {
            theta = 2*CV_PI - std::acos(vertex_table[i].x/l);
        }

        if(vertex_table[i].x >= 0.0&&vertex_table[i].y == 0.0)
        {
            theta = 0.0;
        }

        if(vertex_table[i].x < 0.0&&vertex_table[i].y == 0.0)
        {
            theta = CV_PI;
        }

        phi = std::acos(vertex_table[i].z);

        sph_table.emplace_back(theta, phi);
    }
}