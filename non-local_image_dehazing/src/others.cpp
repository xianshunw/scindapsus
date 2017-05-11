#include "others.hpp"
#include "kd_tree.hpp"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <algorithm>

void calcDarkChannel(const cv::Mat_<cv::Vec3b>& src, cv::Mat_<uchar>& dst, const int s)
{
    dst.create(src.size());

    for(int i = 0; i != src.rows; ++i)
    {
        for(int j = 0; j != src.cols; ++j)
        {
            uchar min_value = 255;
            int bi = i - (s >> 1), ei = bi + s,
            bj = j - (s >> 1), ej = bj + s;
            for(int p = bi; p != ei; ++p)
            {
                if(p < 0||p >= src.rows) continue;
                for(int q = bj; q != ej; ++q)
                {
                    if(q < 0||q >= src.cols) continue;
                    if(min_value > src(p, q)[0]) min_value = src(p, q)[0];
                    if(min_value > src(p, q)[1]) min_value = src(p, q)[1];
                    if(min_value > src(p, q)[2]) min_value = src(p, q)[2];
                }
            }

        dst(i, j) = min_value;
        }
    }
}

void estimateAtmosphericLight(const cv::Mat_<cv::Vec3b>& src, 
    const cv::Mat_<uchar>& dark_channel, cv::Vec3d& A)
{
    int table[256] = { 0 };
    for(int i = 0; i != dark_channel.rows; ++i)
    {
        for(int j = 0; j != dark_channel.cols; ++j)
        {
            table[dark_channel(i, j)] += 1;
        }
    }

    double n = dark_channel.rows*dark_channel.cols, cum = 0;
    uchar threshold = 0;
    for(int i = 0; i != 256; ++i)
    {
        cum += table[i]/n;
        if(cum >= 0.9)
        {
            threshold = i;
            break;
        }
    }

    cv::Mat_<uchar> mask(dark_channel.size());
    for(int i = 0; i != dark_channel.rows; ++i)
    {
        for(int j = 0; j != dark_channel.cols; ++j)
        {
            mask(i, j) = dark_channel(i, j) > threshold ? 1 : 0;
        }
    }

    cv::Mat_<uchar> gray;
    uchar max_value = 0;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    for(int i = 0; i != gray.rows; ++i)
    {
        for(int j = 0; j != gray.cols; ++j)
        {
            if(max_value < gray(i, j)&&mask(i, j) == 1)
            {
                max_value = gray(i, j);
                A[0] = src(i, j)[0];
                A[1] = src(i, j)[1];
                A[2] = src(i, j)[2];
            }
        }
    }
}

void spherical_coordinates(const std::vector<cv::Point3d>& vertex_table, std::vector<cv::Point2d>& sph_table)
{
    sph_table.clear();
    for(int i = 0; i != vertex_table.size(); ++i)
    {
        double l = std::sqrt(vertex_table[i].x*vertex_table[i].x + vertex_table[i].y*vertex_table[i].y), 
            theta, phi;
        if(l == 0.0)
        {
            theta = 0.0; phi = 0.0;
            sph_table.emplace_back(theta, phi);
            continue;
        }
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

        l = std::sqrt(vertex_table[i].x*vertex_table[i].x + vertex_table[i].y*vertex_table[i].y +
            vertex_table[i].z*vertex_table[i].z);

        phi = std::acos(vertex_table[i].z/l);

        sph_table.emplace_back(theta, phi);
    }
}

void spherical_coordinates(const cv::Mat& img, std::vector<cv::Point2d>& img_sph, std::vector<double>& r,
    cv::Vec3d& A)
{
    std::vector<cv::Point3d> vertex_table;
    r.clear();
    for(int i = 0; i != img.rows; ++i)
    {
        for(int j = 0; j != img.cols; ++j)
        {
            const cv::Vec3b &t = img.at<cv::Vec3b>(i, j);
            vertex_table.emplace_back(t[0] - A[0], t[1] - A[1], t[2] - A[2]);
            r.push_back(std::sqrt((t[0] - A[0])*(t[0] - A[0]) + (t[1] - A[1])*(t[1] - A[1]) +
                (t[2] - A[2])*(t[2] - A[2])));
        }
    }

    spherical_coordinates(vertex_table, img_sph);
}

void cluster_img(kd_node* root, const std::vector<cv::Point2d>& sph_table,
    const std::vector<cv::Point2d>& img_sph, std::vector<std::vector<int>>& cluster_result)
{
    cluster_result.clear(); cluster_result.resize(sph_table.size());
    for(int i = 0; i != img_sph.size(); ++i)
    {
        kd_node* t = search_kdTree(sph_table, img_sph[i], root);
        cluster_result[t->data].push_back(i);
    }
}

void lowBound_variance(const std::vector<std::vector<int>>& cluster_result, const std::vector<double>& r,
    const cv::Mat& img, const cv::Vec3d& A, std::vector<double>& t_estimate, std::vector<double>& variance)
{
    const cv::Mat img_col = img.reshape(1, img.cols*img.rows);
    t_estimate.resize(r.size()); variance.resize(r.size());
    std::vector<double> r_max(r.size(), 0.0);

    for(int i = 0; i != cluster_result.size(); ++i)
    {
        double cluster_rmax = 0.0, cluster_rmean = 0.0;
        for(int k = 0; k != cluster_result[i].size(); ++k)
        {
            if(r[cluster_result[i][k]] > cluster_rmax) cluster_rmax = r[cluster_result[i][k]];
            cluster_rmean += r[cluster_result[i][k]];
        }
        cluster_rmean /= cluster_result[i].size();

        double cluster_variance = 0.0;
        for(int k = 0; k != cluster_result[i].size(); ++k)
        {
            t_estimate[cluster_result[i][k]] = r[cluster_result[i][k]]/cluster_rmax;

            const cv::Vec3b& I = img_col.at<cv::Vec3b>(cluster_result[i][k], 0);
            double IA[] = { I[0]/A[0], I[1]/A[1], I[2]/A[2] };
            double t_LB =  IA[0] < IA[1] ? IA[0] : IA[1];
            if(t_LB > IA[2]) t_LB = IA[2]; t_LB = 1 - t_LB;

            if(t_LB > t_estimate[cluster_result[i][k]]) t_estimate[cluster_result[i][k]] = t_LB;

            cluster_variance += std::pow((r[cluster_result[i][k]] - cluster_rmean), 2);
        }
        cluster_variance /= cluster_result[i].size();

        for(int k = 0; k != cluster_result[i].size(); ++k)
        {
            variance[cluster_result[i][k]] = cluster_variance;
        }
    }

    //salce variance to [0, 1]
    double max_variance = *std::max_element(variance.begin(), variance.end());
    if(max_variance == 0.0) return;

    for(int i = 0; i != variance.size(); ++i)
    {
        variance[i] /= max_variance;
    }
}