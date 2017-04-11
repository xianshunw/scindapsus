#include "tools.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <limits>
#include <random>
#include <ctime>
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

    float n = dark_channel.rows*dark_channel.cols, cum = 0;
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

void initTransMap(const cv::Mat_<cv::Vec3b>& src, const cv::Vec3b A, cv::Mat& t, const int s, const float om)
{
    t.create(src.size(), CV_32F);
    cv::Vec3f Af(A[0], A[1], A[2]);

    for(int i = 0; i != src.rows; ++i)
    {
	    for(int j = 0; j != src.cols; ++j)
	    {
	        float min_value = std::numeric_limits<float>::max();
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

	    t.at<float>(i, j) = 1 - om*min_value;
	    }
    }
}


void recoverSceneRadiance(const cv::Mat_<cv::Vec3b>& src, cv::Mat_<cv::Vec3b>& dst, const cv::Mat& t,
	const cv::Vec3b A, const float t0)
{
    CV_Assert(t.type() == CV_32F);
    cv::Vec3f Af(A[0], A[1], A[2]);

    dst.create(src.size());
    for(int i = 0; i != src.rows; ++i)
    {
	    for(int j = 0; j != src.cols; ++j)
	    {
	        float r = t.at<float>(i, j) > t0 ? t.at<float>(i, j) : t0;

	        dst(i, j)[0] = (src(i, j)[0] - Af[0])/r + A[0];
	        dst(i, j)[1] = (src(i, j)[1] - Af[1])/r + A[1];
	        dst(i, j)[2] = (src(i, j)[2] - Af[2])/r + A[2];
	    }
    }
}

void linearEquationSolver(cv::Mat_<float>& A, cv::Mat_<float>& b, cv::Mat_<float>& X,
    const float omega, const float T, unsigned int N)
{
	CV_Assert(b.cols == 1);
	CV_Assert(A.rows == b.rows);
	CV_Assert(A.cols == A.rows);

	//row elementary operation
	bool flag = true;
	for(int i = 0; i != A.rows; ++i)
	{
		if(A(i, i) == 0.0f)
		{
			flag = false;
			for(int j = 0; j != A.cols; ++j)
			{
				if(i == j) continue;
				if(A(i, j) != 0.0f&&A(j, i) != 0.0f)
				{
					cv::Mat t;
					A.row(i).copyTo(t); A.row(j).copyTo(A.row(i));
					t.copyTo(A.row(j));
					b.row(i).copyTo(t); b.row(j).copyTo(b.row(i));
					t.copyTo(b.row(j));

					flag = true;
					break;
				}
			}
		}
	}
	CV_Assert(flag);
	
	X.create(b.size());
	
	std::default_random_engine generator(time(NULL));
	std::uniform_real_distribution<float> distribution(0, 1);
	for(int i = 0; i != X.rows; ++i)
	{
		X(i, 0) = distribution(generator);
	}
	
	while(--N)
	{
		cv::Mat_<float> pre_X = X.clone();
		for(int i = 0; i != X.rows; ++i)
		{
			CV_Assert(A(i, i) != 0.0f);
			X(i, 0) = 0;
			for(int j = 0; j != X.rows; ++j)
			{
				if(i == j) continue;
				X(i, 0) += A(i, j)*X(j, 0);
			}
			
			X(i, 0) = (1/A(i, i)*(b(i, 0) - X(i, 0)))*omega + (1 - omega)*pre_X(i, 0);
		}
		
		double max_value;
		cv::minMaxIdx(cv::abs(pre_X - X), 0, &max_value);
		if (max_value <= T) break;
	}
}

void meanAndCovariance(const cv::Mat_<cv::Vec3b>& win, cv::Vec3f& m, cv::Mat_<float>& c)
{
	int N = win.rows*win.cols;

	cv::Mat_<float> bgrMat(N, 3);
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


void softMatting(const cv::Mat_<cv::Vec3b>& src, const cv::Mat& t_hat, cv::Mat& t_refine,
    const float lambda, const int w)
{
	unsigned int N = src.rows*src.cols;
	cv::Mat_<float> L(N, N);

	for(int i = 0; i != N; ++i)
	{
		for(int j = 0; j != N; ++j)
		{
			
		}
	}
}