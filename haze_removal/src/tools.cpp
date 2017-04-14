#include "tools.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <limits>
#include <random>
#include <ctime>
#include <cmath>
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

void estimateAtmosphericLight(const cv::Mat_<cv::Vec3b>& src, const cv::Mat_<uchar>& dark_channel, cv::Vec3b& A)
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

void initTransMap(const cv::Mat_<cv::Vec3b>& src, const cv::Vec3b A, cv::Mat_<double>& t, const int s, const double om)
{
    t.create(src.size());
    cv::Vec3f Af(A[0], A[1], A[2]);

    for(int i = 0; i != src.rows; ++i)
    {
	    for(int j = 0; j != src.cols; ++j)
	    {
	        double min_value = std::numeric_limits<double>::max();
	        int bi = i - (s >> 1), ei = bi + s,
		    bj = j - (s >> 1), ej = bj + s;
	        for(int p = bi; p != ei; ++p)
	        {
		        if(p < 0||p >= src.rows) continue;
		        for(int q = bj; q != ej; ++q)
		        {
		            if(q < 0||q >= src.cols) continue;
		            if(min_value > src(p, q)[0]/Af[0]) min_value = src(p, q)[0]/Af[0];
		            if(min_value > src(p, q)[1]/Af[1]) min_value = src(p, q)[1]/Af[1];
		            if(min_value > src(p, q)[2]/Af[2]) min_value = src(p, q)[2]/Af[2];
		        } 
	        }

	    t(i, j) = 1 - om*min_value;
	    }
    }
}


void recoverSceneRadiance(const cv::Mat_<cv::Vec3b>& src, cv::Mat_<cv::Vec3b>& dst, const cv::Mat_<double>& t,
	const cv::Vec3b A, const double t0)
{
    CV_Assert(t.type() == CV_64F);
    cv::Vec3f Af(A[0], A[1], A[2]);

    dst.create(src.size());
    for(int i = 0; i != src.rows; ++i)
    {
	    for(int j = 0; j != src.cols; ++j)
	    {
	        double r = t.at<double>(i, j) > t0 ? t.at<double>(i, j) : t0;

	        dst(i, j)[0] = (src(i, j)[0] - Af[0])/r + A[0];
	        dst(i, j)[1] = (src(i, j)[1] - Af[1])/r + A[1];
	        dst(i, j)[2] = (src(i, j)[2] - Af[2])/r + A[2];
	    }
    }
}

void linearEquationSolver(cv::SparseMat_<double>& A, cv::Mat_<double>& b, cv::Mat_<double>& X,
    const double omega, const double T, unsigned int N)
{
	CV_Assert(b.cols == 1);

	//row elementary operation
	bool flag = true;
	for(int i = 0; i != b.rows; ++i)
	{
		if(A.ref(i, i) == 0.0f)
		{
			flag = false;
			for(int j = 0; j != b.rows; ++j)
			{
				if(i == j) continue;
				if(A.ref(i, j) != 0.0f&&A.ref(j, i) != 0.0f)
				{
					for(int k = 0; k != b.rows; ++k)
					{
						double temp_value = A.ref(i, k);
						if(A.ref(j, k) == 0.0f) A.erase(i, k); else A.ref(i, k) = A.ref(j, k);
						if(temp_value == 0.0f) A.erase(j, k); else A.ref(j, k) = temp_value;
					}

					flag = true;
					break;
				}
			}
		}
	}
	CV_Assert(flag);
	
	std::default_random_engine generator(time(NULL));
	std::uniform_real_distribution<double> distribution(0, 1);
	for(int i = 0; i != X.rows; ++i)
	{
		X(i, 0) = distribution(generator);
	}
	X.create(b.size());
	
	std::cout << "SOR Iterating..." << std::endl;
	while(--N)
	{
		cv::Mat_<double> pre_X = X.clone();
		for(int i = 0; i != X.rows; ++i)
		{
			CV_Assert(A(i, i) != 0.0f);
			X(i, 0) = 0;
			for(int j = 0; j != X.rows; ++j)
			{
				if(i == j) continue;
				X(i, 0) += A(i, j)*X(j, 0);
			}
			
			X(i, 0) = (omega/A(i, i)*(b(i, 0) - X(i, 0))) + (1 - omega)*pre_X(i, 0);
		}
		
		double max_value;
		cv::minMaxIdx(cv::abs(pre_X - X), 0, &max_value);
		if (max_value <= T) break;
		std::cout<< "error = " << max_value <<std::endl;
	}
}

void meanAndCovariance(const cv::Mat_<cv::Vec3b>& win, cv::Vec3f& m, cv::Mat_<double>& c)
{
	int N = win.rows*win.cols;

	cv::Mat_<double> bgrMat(N, 3);
	for(int i = 0; i != win.rows; ++i)
	{
		for(int j = 0; j != win.cols; ++j)
		{
			bgrMat(i*win.rows + j, 0) = win(i, j)[0];
			bgrMat(i*win.rows + j, 1) = win(i, j)[1];
			bgrMat(i*win.rows + j, 2) = win(i, j)[2];

			m[0] += win(i, j)[0];
			m[1] += win(i, j)[1];
			m[2] += win(i, j)[2];
		}
	}
	m[0] /= N; m[1] /= N; m[2] /= N;

	c.create(3, 3);
	for(int i = 0; i != c.rows; ++i)
	{
		for(int j = i; j != c.cols; ++j)
		{
			c(i, j) = (bgrMat.col(i) - m[i]).dot(bgrMat.col(j) - m[j]);
			c(j, i) = c(i, j);
		}
	}

}


void softMatting(const cv::Mat_<cv::Vec3b>& src, cv::Mat_<double>& t_hat, cv::Mat_<double>& t_refine,
    const double lambda, const int w)
{
	double eps = 5e-4;
	int N = src.rows*src.cols;
	int size[] = { N, N };
	cv::SparseMat_<double> L(2, size);

	std::cout<< "compute laplacian matrix... " << std::endl;
	for(int i = 0; i != N; ++i)
	{
		for(int j = 0; j != N; ++j)
		{
			int row_i = i/src.rows, col_i = i%src.rows,
			    row_j = j/src.rows, col_j = j%src.rows;

			if(std::abs(row_i - row_j) >= w||std::abs(col_i - col_j) >= w) continue;
			L.ref(i, j) = 0.0f;
			int br = row_i - w + 1, bc = col_i - w + 1;
			for(int p = br; p != row_i + 1; ++p)
			{
				if(p < 0||p + w - 1 >= src.rows) continue;
				for(int q = bc; q != col_i + 1; ++q)
				{
					if(q < 0||q + w - 1 >= src.cols) continue;
					cv::Rect temp_win(q, p, w, w);
					if(!temp_win.contains(cv::Point(col_j, row_j))) continue;

					cv::Vec3f mean_value;
					cv::Mat_<double> cov_mat;

					meanAndCovariance(src.rowRange(p, p + w).colRange(q, q + w), mean_value, cov_mat);
					cv::Mat_<double> I_i(1, 3), I_j(3, 1);
					I_i(0, 0) = src(row_i, col_i)[0] - mean_value[0];
					I_i(0, 1) = src(row_i, col_i)[1] - mean_value[1];
					I_i(0, 2) = src(row_i, col_i)[2] - mean_value[2];

					I_j(0, 0) = src(row_j, col_j)[0] - mean_value[0];
					I_j(1, 0) = src(row_j, col_j)[1] - mean_value[1];
					I_j(2, 0) = src(row_j, col_j)[2] - mean_value[2];

					cv::Mat_<double> temp_mat;
					cv::invert(cov_mat + cv::Mat_<double>::eye(3, 3)*eps/(w*w), temp_mat);
					temp_mat = I_i*temp_mat*I_j;
					L.ref(i, j) += i == j ? 1 - (1 + temp_mat(0, 0))/(w*w) :
					    0 - (1 + temp_mat(0, 0))/(w*w);
				}
			}
		}
	}

	t_hat = lambda*t_hat.reshape(1, N);

	for(int i = 0; i != N; ++i)
	{
		L.ref(i, i) += lambda;
	}

	linearEquationSolver(L, t_hat, t_refine);
	t_refine = t_refine.reshape(2, src.rows);
	t_refine = t_refine;
}
