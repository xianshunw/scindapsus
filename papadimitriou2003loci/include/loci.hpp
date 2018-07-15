#ifndef __LOCI_HPP__
#define __LOCI_HPP__

/*
 * This headfile define the core class of this program.
 * Because we hope this program can be used in high dimension
 * dataset, so we use cv::Mat not cv::Point to store the input
 * data points. Every single row of the dataset matrix store a
 * data point.
 * Note that, we just implement the fast version(aLOCI) of this paper not the
 * accurate version.
 */

#include <opencv2/core/core.hpp>
#include <vector>
#include <fstream>

class LOCI
{
public:
    LOCI(cv::Mat& _dataset);
    void detect(std::vector<bool>& result);
private:
    cv::Mat dataset;
    int g_num, max_level, child_num, dim;
    cv::Mat child_dir;

    /*
     * k dimension quad-tree
     * Mind that k = 2^dim which dim is the dimension of the dataset
     */
    struct KQuadTreeNode
    {
        KQuadTreeNode(int dim, int child_num);

        KQuadTreeNode* parent;

        std::vector<KQuadTreeNode*> children;
        cv::Mat center;                            //one row

        int level, count;
        float radii;

        bool is_leaf;
    };

    // insert path 
    std::vector<std::vector<KQuadTreeNode*>> pts_leaf;

    /*
     * Build a k dimension quad-tree using the input dataset
     * The return value is the vector of the pointers of the non-empty leafs.
     */
    KQuadTreeNode* buildKQuadTree(float radii, int l, cv::Mat center, KQuadTreeNode* parent);

    /*
     * Insert a point in the quad tree
     */
    KQuadTreeNode* insertPoint(KQuadTreeNode* node, cv::Mat pt);

    /*
     * Distance between point and node's center
     */
    float disPtNode(cv::Mat pt, KQuadTreeNode*);

    /*
     * Destory quad-tree
     */
    void destoryKQuadTree(KQuadTreeNode* root);
};

/*
 * Load dataset from a text file
 */
void loadData(std::string file, cv::Mat& data);

#endif