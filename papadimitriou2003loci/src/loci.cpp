#include <loci.hpp>
#include <queue>

#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <random>

#include <iostream>

LOCI::LOCI(cv::Mat& _dataset)
{
    dataset = _dataset;
    g_num = 2;
    max_level = 9;
    dim = dataset.cols;

    child_num = 1 << dim;

    //find the child positions related to the center
    int fl[2] = {1, -1};
    child_dir.create(child_num, dim, CV_32SC1);
    for(int j = 0; j != child_num; ++j)
    {
        int mask = j, *ptr_child_dir = child_dir.ptr<int>(j);
        for(int i = 0; i != dim; ++i)
        {
            ptr_child_dir[i] = fl[(mask >> i) & 1];
        }
    }
}

void LOCI::detect(std::vector<bool>& result)
{
    result.resize(dataset.rows, false);

    // collect the dataset's basic infomations
    cv::Mat center(1, dim, CV_32FC1), range(1, dim, CV_32FC1);
    cv::Mat min_vec(1, dim, CV_32FC1), max_vec(1, dim, CV_32FC1);
    double min_value, max_value;
    auto *ptr_center = center.ptr<float>();
    auto *ptr_range = range.ptr<float>();
    auto *ptr_min = min_vec.ptr<float>();
    auto *ptr_max = max_vec.ptr<float>();
    float diameter = std::numeric_limits<float>::min();
    for(int i = 0; i != dim; ++i)
    {
        cv::minMaxIdx(dataset.col(i), &min_value, &max_value);
        ptr_min[i] = min_value;
        ptr_max[i] = max_value;
        ptr_range[i] = max_value - min_value;
        ptr_center[i] = ptr_range[i] * 0.5f + min_value;

        // diameter defined as the max range among all the dimension
        if(diameter < ptr_range[i]) diameter = ptr_range[i];
    }

    /*******************************************/
    /*           Initialization Step           */
    /*******************************************/
    // generate shifts
    cv::Mat shifts(g_num, dim, CV_32FC1);
    std::memset(shifts.ptr<float>(), 0, dim * sizeof(float));             //s0 should be 0

    std::default_random_engine gen(19210817);                             //the elder can do it
    auto *ptr_shift = shifts.ptr<float>();
    for(int i = 0; i != dim; ++i)
    {
        float half_range = ptr_range[i] / 2.0f;
        std::uniform_real_distribution<float> dis(-half_range, half_range);
        for(int j = 1; j != g_num; ++j)
        {
            ptr_shift[j * dim + i] = dis(gen);
        }
    }

    // build trees according to the random shifts
    std::vector<KQuadTreeNode*> roots;
    for(int i = 0; i != g_num; ++i)
    {
        // shift center
        cv::Mat shift_center(1, dim, CV_32FC1);
        auto *ptr_sc = shift_center.ptr<float>();
        for(int j = 0; j != dim; ++j)
        {
            ptr_sc[j] = ptr_center[j] + ptr_shift[i * dim + j];
        }

        KQuadTreeNode* root = buildKQuadTree(diameter / 2.0f, 0, shift_center, nullptr);
        roots.push_back(root);
    }

    /*******************************************/
    /*          Pre-processing stage           */
    /*******************************************/
    pts_leaf.resize(dataset.rows);
    for(int i = 0; i != dataset.rows; ++i)
    {
        for(int j = 0; j != g_num; ++j)
        {
            pts_leaf[i].push_back(insertPoint(roots[j], dataset.row(i)));
        }
    }

    /*******************************************/
    /*          Post-processing stage          */
    /*******************************************/
    for(int i = 0; i != dataset.rows; ++i)
    {
        std::vector<KQuadTreeNode*> g_nodes = pts_leaf[i];
        for(int l = 0; l != max_level - 1; ++l)
        {
            // select Ci
            float dis = std::numeric_limits<float>::max();
            KQuadTreeNode *Ci = nullptr;
            for(int j = 0; j != g_num; ++j)
            {
                float curr_dis = disPtNode(dataset.row(i), g_nodes[j]);
                if(curr_dis < dis) Ci = g_nodes[j];
            }

            // go upside of the trees
            for(int j = 0; j != g_num; ++j)
            {
                g_nodes[j] = g_nodes[j]->parent;
            }

            // select Cj
            dis = std::numeric_limits<float>::max();
            KQuadTreeNode *Cj = nullptr;
            for(int j = 0; j != g_num; ++j)
            {
                float curr_dis = disPtNode(Ci->center, g_nodes[j]);
                if(curr_dis < dis) Cj = g_nodes[j];
            }

            // well, let's compute the two very important values
            float S1 = 0.f, S2 = 0.f, S3 = 0.f;
            for(int j = 0; j != child_num; ++j)
            {
                S1 += Cj->children[j]->count;
                S2 += Cj->children[j]->count * Cj->children[j]->count;
                S3 += Cj->children[j]->count * Cj->children[j]->count * Cj->children[j]->count;
            }

            float avg_n = S2 / S1, delta_avg_n = std::sqrt(S3 / S1 - S2 * S2 / (S1 * S1));
            float mdef = 1 - Ci->count / avg_n, delta_mdef = delta_avg_n / avg_n;

            // flag
            if(mdef > 3 * delta_mdef)
            {
                result[i] = true;
                continue;
            }
        }
    }


    // clear the trees
    for(int i = 0; i != roots.size(); ++i)
    {
        destoryKQuadTree(roots[i]);
    }
}

LOCI::KQuadTreeNode::KQuadTreeNode(int dim, int child_num)
{
    parent = nullptr;
    children.resize(child_num, nullptr);
    center.create(1, dim, CV_32FC1);
    count = 0;
}

LOCI::KQuadTreeNode* LOCI::buildKQuadTree(float radii, int l, cv::Mat center, KQuadTreeNode* parent)
{
    // recursive terminate
    if(l >= max_level) return nullptr;

    // construct current node
    auto* node = new KQuadTreeNode(dim, child_num);
    node->parent = parent;
    node->center = center;
    node->radii = radii;
    node->level = l;

    // papare children's information
    float child_radii = radii * 0.5f;
    int child_l = l + 1;
    cv::Mat children_center(child_num, dim, CV_32FC1);
    auto *ptr_cen = center.ptr<float>();

    for(int j = 0; j != child_num; ++j)
    {
        auto *ptr_child_cen = children_center.ptr<float>(j);
        auto *ptr_child_dir = child_dir.ptr<int>(j);
        for(int i = 0; i != dim; ++i)
        {
            ptr_child_cen[i] = ptr_cen[i] + ptr_child_dir[i] * child_radii;
        }
    }

    for(int i = 0; i != child_num; ++i)
    {
        node->children[i] = buildKQuadTree(child_radii, child_l, children_center.row(i), node);
    }

    // if this node is leaf or not
    node->is_leaf = true;
    for(int i = 0; i != dim; ++i)
    {
        if(node->children[i])
        {
            node->is_leaf = false;
            break;
        }
    }

    return node;
}

LOCI::KQuadTreeNode* LOCI::insertPoint(LOCI::KQuadTreeNode *node, cv::Mat pt)
{
    cv::Mat dir(1, dim, CV_32SC1);
    auto *ptr_dir = dir.ptr<int>();
    auto *ptr_pt = pt.ptr<float>();

    KQuadTreeNode* curr_node = node;
    while(!curr_node->is_leaf)
    {
        ++curr_node->count;

        // determine which child will in the insert path
        auto *ptr_curr_cen = curr_node->center.ptr<float>();
        for(int i = 0; i != dim; ++i)
        {
            ptr_dir[i] = ptr_pt[i] - ptr_curr_cen[i] < 0 ? -1 : 1;
        }

        for(int i = 0; i != child_num; ++i)
        {
            auto *ptr_child_dir = child_dir.ptr<int>(i);
            if(!std::memcmp(ptr_dir, ptr_child_dir, sizeof(int) * dim))
            {
                curr_node = curr_node->children[i];
                break;
            }
        }
    }
    ++curr_node->count;

    return curr_node;
}

float LOCI::disPtNode(cv::Mat pt, LOCI::KQuadTreeNode* node)
{
    return cv::norm(pt - node->center);
}

void LOCI::destoryKQuadTree(LOCI::KQuadTreeNode *root)
{
    // This can be done just by deleting the root node
    // because we define a vector of pointers point to
    // children's node, which make all the nodes in the 
    // tree will be destoried recursively.
    delete root;
}

void loadData(std::string file, cv::Mat& data)
{
    std::ifstream f(file);

   std::vector<std::vector<float>> data_mat;
   std::vector<float> data_vec;
   std::string line;
   std::istringstream str;
   float tmp;
   while(std::getline(f, line))
   {
       str.str(line);

       while(str >> tmp)
       {
           data_vec.push_back(tmp);
       }

       data_mat.push_back(data_vec);
       data_vec.clear();
       str.clear();
   }

   f.close();

   data.create(data_mat.size(), data_mat[0].size(), CV_32FC1);
   for(int i = 0; i != data.rows; ++i)
   {
       auto *ptr_row = data.ptr<float>(i);
       for(int j = 0; j != data.cols; ++j)
       {
           ptr_row[j] = data_mat[i][j];
       }
   }
}