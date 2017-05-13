#include "others.hpp"
#include "kd_tree.hpp"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <algorithm>
#include <gsl/gsl_spmatrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_splinalg.h>
#include <gsl/gsl_math.h>
#include <iostream>

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

void trans_variance(const std::vector<std::vector<int>>& cluster_result, const std::vector<double>& r,
    const cv::Vec3d& A, std::vector<double>& t_init, std::vector<double>& variance)
{
    t_init.resize(r.size()); variance.resize(r.size());
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
            if(cluster_rmax == 0.0) t_init[cluster_result[i][k]] = 0;
            else t_init[cluster_result[i][k]] = r[cluster_result[i][k]]/cluster_rmax;

            cluster_variance += std::pow((r[cluster_result[i][k]] - cluster_rmean), 2);
        }
        cluster_variance /= cluster_result[i].size();

        for(int k = 0; k != cluster_result[i].size(); ++k)
        {
            variance[cluster_result[i][k]] = cluster_variance;
        }
    }
}

void regular_trans(const cv::Mat& img, const cv::Vec3d& A, std::vector<double>& t_init,
    const std::vector<double>& variance, std::vector<double>& t_refine, double lambda)
{
    cv::Mat img_col = img.reshape(1, img.cols*img.rows);

    std::vector<double> recip_variance(variance.size());
    double max_variance = *std::max_element(variance.begin(), variance.end());
    for(int i = 0; i != variance.size(); ++i)
    {
        recip_variance[i] = variance[i] == 0.0 ? 1.0/max_variance : 1.0/variance[i];
    }

    double max_value = *std::max_element(recip_variance.begin(), recip_variance.end()),
        min_value = *std::min_element(recip_variance.begin(), recip_variance.end());
    for(int i = 0; i != variance.size(); ++i)
    {
        recip_variance[i] = (recip_variance[i] - min_value)/max_value;
    }

    //low bound constraint
    for(int i = 0; i != img_col.rows; ++i)
    {
        const cv::Vec3b& pixel = img_col.at<cv::Vec3b>(i, 0);
        double IA[] = { pixel[0]/A[0], pixel[1]/A[1], pixel[2]/A[2] };
        if(t_init[i] < IA[0]) t_init[i] = IA[0];
        if(t_init[i] < IA[1]) t_init[i] = IA[1];
        if(t_init[i] < IA[2]) t_init[i] = IA[2];
    }

    gsl_spmatrix *coeff = gsl_spmatrix_alloc_nzmax(img_col.rows, img_col.rows, 5*img_col.rows, GSL_SPMATRIX_TRIPLET);
    gsl_vector *b = gsl_vector_alloc(img_col.rows), *X = gsl_vector_alloc(img_col.rows);
    for(int i = 0; i != img.rows; ++i)
    {
        for(int j = 0; j != img.cols; ++j)
        {
            int idx = i*img.cols + j;

            //existed 4-neighbors
            std::vector<int> q;
            if(i - 1 >= 0) q.push_back(idx - img.cols);
            if(j - 1 >= 0) q.push_back(idx - 1);
            if(j + 1 < img.cols) q.push_back(idx + 1);
            if(i + 1 < img.rows) q.push_back(idx + img.cols);

            double temp_value = 0.0;
            const cv::Vec3b& pixel_idx = img_col.at<cv::Vec3b>(idx, 0);
            for(int k = 0; k != q.size(); ++k)
            {
                const cv::Vec3b& pixel_q = img_col.at<cv::Vec3b>(q[k], 0);
                double temp_norm = std::sqrt(std::pow(static_cast<double>(pixel_idx[0]) -
                    static_cast<double>(pixel_q[0]), 2) +
                    std::pow(static_cast<double>(pixel_idx[1]) - static_cast<double>(pixel_q[1]), 2) +
                    std::pow(static_cast<double>(pixel_idx[2]) - static_cast<double>(pixel_q[2]), 2));
                if(temp_norm == 0.0) temp_norm = 1;
                temp_value += 1/temp_norm;

                gsl_spmatrix_set(coeff, idx, q[k], -4.0*lambda/temp_norm);
            }
            temp_value *= (4*lambda);

            gsl_spmatrix_set(coeff, idx, idx, 2*recip_variance[idx] + temp_value);
            gsl_vector_set(b, idx, 2*recip_variance[idx]*t_init[idx]);
            gsl_vector_set(X, idx, t_init[idx]);
        }
    }

    //sovle the linear equations
    gsl_spmatrix *C = gsl_spmatrix_ccs(coeff);
    const double tol = 10e-6;
    size_t max_iter = 10;
    const gsl_splinalg_itersolve_type *T = gsl_splinalg_itersolve_gmres;
    gsl_splinalg_itersolve *work = gsl_splinalg_itersolve_alloc(T, img_col.rows, 0);

    int status, iter = 0; double residual;
    do
    {
        status = gsl_splinalg_itersolve_iterate(C, b, tol, X, work);
        residual = gsl_splinalg_itersolve_normr(work);

        std::cout << "iter:" << iter << "residual: " << residual <<std::endl;
    }
    while(status == GSL_CONTINUE&&++iter < max_iter);

    t_refine.resize(img_col.rows);
    for(int i = 0; i != t_refine.size(); ++i)
    {
        t_refine[i] = gsl_vector_get(X, i);
    }

    gsl_spmatrix_free(coeff);
    gsl_vector_free(b);
    gsl_vector_free(X);
}