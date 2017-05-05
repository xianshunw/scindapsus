#include <deque>
#include "kd_tree.hpp"
#include <iostream>

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

int split(std::vector<cv::Point2d>& sph_table, std::vector<int>& subset,
    std::vector<std::vector<int>>& split_subsets, int dimension)
{
    double *value = new double[subset.size()];
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

    int idx = subset.size()/2, r = subset[idx]; split_subsets.clear();
    std::vector<int> subset1, subset2;
    for(auto iter = subset.cbegin(); iter != subset.cbegin() + idx; ++iter)
    {
        subset1.push_back(*iter);
    }
    for(auto iter = subset.cbegin() + idx + 1; iter != subset.cend(); ++iter)
    {
        subset2.push_back(*iter);
    }
    split_subsets.push_back(subset1); split_subsets.push_back(subset2);

    return r;
}

double points_distance(cv::Point2d pt1, cv::Point2d pt2)
{
    return std::sqrt((pt1.x - pt2.x)*(pt1.x - pt2.x) + (pt1.y - pt2.y)*(pt1.y - pt2.y));
}

kd_node* build_kdTree(std::vector<cv::Point2d>& sph_table, kd_node* p, std::vector<int>& subset)
{
    kd_node* r = new kd_node; r->parent = p;
    if(subset.size() == 1)
    {
        r->data = subset[0];
        r->dimension = 0;
        r->left = r->right = nullptr;
        r->is_leaf = true;
        return r;
    }

    std::vector<std::vector<int>> subsets;
    r->dimension = dimension_choice(sph_table, subset);
    r->data = split(sph_table, subset, subsets, r->dimension);
    r->is_leaf = false;

    r->left = subsets[0].size() != 0 ? build_kdTree(sph_table, r, subsets[0]) : nullptr;
    r->right = subsets[1].size() != 0 ? build_kdTree(sph_table, r, subsets[1]) : nullptr;

    return r;
}

void print_kdTree(std::vector<cv::Point2d>& sph_table, kd_node* root, int blk)
{
    for(int i = 0; i != blk; ++i) std::cout<<"      "<<std::flush;
    std::cout<<"|--("<<sph_table[root->data].x<<","
             <<sph_table[root->data].y
             <<","<<root->dimension<<")"<<std::endl;
    if(root->right != nullptr) print_kdTree(sph_table, root->right, blk + 1);
    if(root->left != nullptr) print_kdTree(sph_table, root->left, blk + 1);
}

void destory_kdTree(kd_node* root)
{
    std::deque<kd_node*> q;
    q.push_back(root);
    while(q.size() != 0)
    {
        kd_node* t = *(q.begin());
        q.pop_front();

        if(t->left != nullptr) q.push_back(t->left);
        if(t->right != nullptr) q.push_back(t->right);

        delete t;
    }
}

kd_node* search_kdTree(std::vector<cv::Point2d>& sph_table, cv::Point2d pt, kd_node* root)
{
    if(root->is_leaf) return root;

    kd_node* branch_nearest = root;
    while(!branch_nearest->is_leaf)
    {
        if(branch_nearest->left == nullptr)
        {
            branch_nearest = branch_nearest->right;
            continue;
        }
        if(branch_nearest->right == nullptr)
        {
            branch_nearest = branch_nearest->left;
            continue;
        }

        if(branch_nearest->dimension == 0)
        {
            branch_nearest = pt.x <= sph_table[branch_nearest->data].x ? branch_nearest->left :
                branch_nearest->right;
        }
        else
        {
            branch_nearest = pt.y <= sph_table[branch_nearest->data].y ? branch_nearest->right :
                branch_nearest->right;
        }
    }

    double curr_dist = points_distance(sph_table[branch_nearest->data], pt);
    kd_node* search_node = curr_nearest->parent;

    //backtrack
    while(search_node->parent != root->parent)
    {
        //if another branch is empty
        if(search_node->left == nullptr||search_node->right == nullptr)
        {
            search_node = search_node->parent;
            continue;
        }

        double c = search_node->dimension == 0 ? std::abs(pt.x - sph_table[search_node->data].x) :
            std::abs(pt.y - sph_table[search_node->data].y);
        if(c <= curr_nearest)
        {
            kd_node* another_subtree = curr_nearest == search_node->left ? search_node->right :
                search_node->left;
            curr_dist = search_dist;
            curr_nearest = search_node;

            kd_node* nearest_another = search_kdTree(sph_table, pt, another_substree);
            double another_dist = points_distance(sph_table[nearest_another->data], pt);

            if(another_dist < curr_nearest)
            {
                curr_nearest = nearest_another;
                curr_dist = another_dist;
                search_node = curr_nearest->parent;
            }
            continue;
        }


        double search_dist = points_distance(sph_table[search_node->data], pt);
        if(search_dist < curr_dist)
        {
            curr_dist = search_dist;
            curr_nearest = search_node;
        }

        search_node = curr_nearest->parent;
    }
}
