#ifndef __KD_TREE_HPP__
#define __KD_TREE_HPP__

#include <opencv2/core.hpp>
#include <vector>

struct kd_node
{
    int data, dimension;
    kd_node *left, *right, *parent;
	bool is_leaf;
};

int dimension_choice(std::vector<cv::Point2d>& sph_table, std::vector<int>& subset);

int split(std::vector<cv::Point2d>& sph_table, std::vector<int>& subset,
    std::vector<std::vector<int>>& split_subsets, int dimension);

double points_distance(cv::Point2d pt1, cv::Point2d pt2);

kd_node* build_kdTree(std::vector<cv::Point2d>& sph_table, kd_node* p, std::vector<int>& subset);

void print_kdTree(std::vector<cv::Point2d>& sph_table, kd_node* root, int blk = 0);

void destory_kdTree(kd_node* root);

kd_node* search_kdTree(const std::vector<cv::Point2d>& sph_table, const cv::Point2d pt, kd_node* root);

#endif