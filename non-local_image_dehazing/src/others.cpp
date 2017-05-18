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

void calcDarkChannel(const cv::Mat_<cv::Vec3d>& src, cv::Mat_<double>& dst, const int s)
{
    dst.create(src.size());

    for(int i = 0; i != src.rows; ++i)
    {
        for(int j = 0; j != src.cols; ++j)
        {
            double min_value = 1;
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

void estimateAtmosphericLight(const cv::Mat_<cv::Vec3d>& src, 
    const cv::Mat_<double>& dark_channel, cv::Vec3d& A)
{
    int num = 0.001*src.rows*src.cols;
	std::vector<std::vector<double>> table(num, { 0, 0, 0 });
	
	//computer gray image
	cv::Mat_<double> gray(src.size());
	for(int i = 0; i != src.rows; ++i)
	{
		for(int j = 0; j != src.cols; ++j)
		{
			gray(i, j) = (src(i, j)[0] + src(i, j)[1] + src(i, j)[2])/3;
		}
	}
	
	for(int i = 0; i != gray.rows; ++i)
	{
		for(int j = 0; j != gray.cols; ++j)
		{
			int idx; double min_value = 1;
			for(int k = 0; k != table.size(); ++k)
			{
				if(min_value > table[k][2])
				{
					idx = k; min_value = table[k][2];
				}
			}
			
			if(gray(i, j) > min_value)
			{
				table[idx][0] = i; table[idx][1] = j;
				table[idx][2] = gray(i, j);
			}
		}
	}

    A[0] = A[1] = A[2] = 0.0;
	for(int i = 0; i != table.size(); ++i)
	{
		A[0] += src(table[i][0], table[i][1])[0];
		A[1] += src(table[i][0], table[i][1])[1];
		A[2] += src(table[i][0], table[i][1])[2];
	}
	A[0] /= table.size(); A[1] /= table.size(); A[2] /= table.size();
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
            const cv::Vec3d &t = img.at<cv::Vec3d>(i, j);
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

void trans_stdde(const std::vector<std::vector<int>>& cluster_result, const std::vector<double>& r,
    const cv::Vec3d& A, std::vector<double>& t_init, std::vector<double>& stdde)
{
    t_init.resize(r.size()); stdde.resize(r.size());
    std::vector<double> r_max(r.size(), 0.0);
    double trans_min = 0.1;

    for(int i = 0; i != cluster_result.size(); ++i)
    {
        if(cluster_result[i].size() == 0) continue;
        double cluster_rmax = 0.0, cluster_rmean = 0.0;
        for(int k = 0; k != cluster_result[i].size(); ++k)
        {
            if(r[cluster_result[i][k]] > cluster_rmax) cluster_rmax = r[cluster_result[i][k]];
            cluster_rmean += r[cluster_result[i][k]];
        }
        cluster_rmean /= cluster_result[i].size();

        double cluster_stdde = 0.0;
        for(int k = 0; k != cluster_result[i].size(); ++k)
        {
            if(cluster_rmax == 0.0) t_init[cluster_result[i][k]] = 0;
            else t_init[cluster_result[i][k]] = r[cluster_result[i][k]]/cluster_rmax;

            //bound to [trans_min, 1]
            if(t_init[cluster_result[i][k]] < trans_min) t_init[cluster_result[i][k]] = trans_min;

            cluster_stdde += std::pow((r[cluster_result[i][k]] - cluster_rmean), 2);
        }
        cluster_stdde = sqrt(cluster_stdde/cluster_result[i].size());

        for(int k = 0; k != cluster_result[i].size(); ++k)
        {
            stdde[cluster_result[i][k]] = cluster_stdde;
        }
    }
}

void regular_trans(const cv::Mat& img, const cv::Vec3d& A, std::vector<double>& t_init,
    std::vector<double>& stdde, const std::vector<std::vector<int>>& cluster_result,
    std::vector<double>& t_refine, double lambda)
{
    cv::Mat_<double> gray(img.size());
    cv::Mat img_col = gray.reshape(1, img.cols*img.rows);

    for(int i = 0; i != img.rows; ++i)
    {
        for(int j = 0; j != img.cols; ++j)
        {
            gray(i, j) = (img.at<cv::Vec3d>(i, j)[0] + img.at<cv::Vec3d>(i, j)[1] + img.at<cv::Vec3d>(i, j)[2])/3;
        }
    }
    cv::Mat_<double> gray_col = gray.reshape(1, img.cols*img.rows);

    /*
    std::vector<double> recip_variance(variance.size());
    double max_variance = *std::max_element(variance.begin(), variance.end());
    for(int i = 0; i != variance.size(); ++i)
    {
        recip_variance[i] = variance[i] <= 0.01 ? 1.0/max_variance : 1.0/variance[i];
    }

    double max_value = *std::max_element(recip_variance.begin(), recip_variance.end()),
        min_value = *std::min_element(recip_variance.begin(), recip_variance.end());
    for(int i = 0; i != variance.size(); ++i)
    {
        recip_variance[i] = (recip_variance[i] - min_value)/max_value;
    }
    */
    //bound stdde to [0, 1] and
    //do magic transform according to //radius_eval_fun = @(r) min(1, 3*max(0.001, r-0.1))//
    double max_stdde = *std::max_element(stdde.begin(), stdde.end());
    for(int i = 0; i != stdde.size(); ++i)
    {
        stdde[i] /= max_stdde;
        stdde[i] = stdde[i] - 0.1 > 0.001 ? stdde[i] - 0.1 : 0.001;
        stdde[i] = stdde[i]*3 > 1 ? 1 : stdde[i]*3;
    }

    //scale stdde accordding the number of pixels of the cluster
    for(int i = 0; i != cluster_result.size(); ++i)
    {
        if(cluster_result[i].size() == 0) continue;
        double s = cluster_result[i].size() < 50 ? 1 : cluster_result[i].size()/50.0;
        for(int k = 0; k != cluster_result[i].size(); ++k)
        {
            stdde[cluster_result[i][k]] *= s;
        } 
    }

    //normalize stdde
    double small_num = 0.00001;
    max_stdde = *std::max_element(stdde.begin(), stdde.end());
    double min_stdde = *std::min_element(stdde.begin(), stdde.end());
    for(int i = 0; i != stdde.size(); ++i)
    {
        stdde[i] -= min_stdde;
        stdde[i] /= (max_stdde + max_stdde);
    }

    //low bound constraint
    for(int i = 0; i != img_col.rows; ++i)
    {
        const cv::Vec3d& pixel = img_col.at<cv::Vec3d>(i, 0);
        double IA[] = { pixel[0]/A[0], pixel[1]/A[1], pixel[2]/A[2] }, t_LB = IA[0];
        if(t_LB > IA[1]) t_LB = IA[1]; if(t_LB > IA[2]) t_LB = IA[2];
        t_LB = 1 - t_LB;
        if(t_init[i] < t_LB) t_init[i] = t_LB;

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

            double temp_value = 0.0, pixel_idx = gray_col(idx, 0);
            //const cv::Vec3b& pixel_idx = img_col.at<cv::Vec3b>(idx, 0);
            for(int k = 0; k != q.size(); ++k)
            {
                //const cv::Vec3b& pixel_q = img_col.at<cv::Vec3b>(q[k], 0);
                /*
                double temp_norm = std::sqrt(std::pow(static_cast<double>(pixel_idx[0]) -
                    static_cast<double>(pixel_q[0]), 2) +
                    std::pow(static_cast<double>(pixel_idx[1]) - static_cast<double>(pixel_q[1]), 2) +
                    std::pow(static_cast<double>(pixel_idx[2]) - static_cast<double>(pixel_q[2]), 2));
                */
                double temp_norm = std::pow((pixel_idx - gray_col(q[k], 0)), 2) + small_num;
                temp_value += 1/temp_norm;

                gsl_spmatrix_set(coeff, idx, q[k], -lambda/temp_norm);
            }
            temp_value *= (lambda);

            gsl_spmatrix_set(coeff, idx, idx, stdde[idx] + temp_value);
            gsl_vector_set(b, idx, stdde[idx]*t_init[idx]);
            gsl_vector_set(X, idx, t_init[idx]);
        }
    }

    //sovle the linear equations
    gsl_spmatrix *C = gsl_spmatrix_ccs(coeff);
    const double tol = 10e-4;
    size_t max_iter = 500;
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

void dehaze(cv::Mat& img_scale, cv::Vec3d A, std::vector<double>& t, cv::Mat& haze_free)
{
    haze_free.create(img_scale.size(), CV_8UC3);
    for(int i = 0; i != img_scale.rows; ++i)
    {
        for(int j = 0; j != img_scale.cols; ++j)
        {
            int idx = i*img_scale.cols + j;;
            double r = t[idx] < 0.1 ? 0.1 : t[idx], t0 = (img_scale.at<cv::Vec3d>(i, j)[0] - (1 - t[idx])*A[0])/r,
                t1 = (img_scale.at<cv::Vec3d>(i, j)[1] - (1 - t[idx])*A[1])/r,
                t2 = (img_scale.at<cv::Vec3d>(i, j)[2] - (1 - t[idx])*A[2])/r;
            
            haze_free.at<cv::Vec3b>(i, j)[0] = t0 < 0 ? : t0 > 1 ? 255 : t0*255;
	        haze_free.at<cv::Vec3b>(i, j)[1] = t1 < 0 ? : t1 > 1 ? 255 : t1*255;
	        haze_free.at<cv::Vec3b>(i, j)[2] = t2 < 0 ? : t2 > 1 ? 255 : t2*255;
        }
    }
}