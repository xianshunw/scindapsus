#include "tools.hpp"
#include <opencv2/core.hpp>

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
		    min_value = src(i, j)[0] > src(i, j)[1] ? src(i, j)[1] :
			src(i, j)[0];
		    if(min_value > src(i, j)[2]) min_value = src(i, j)[2];
		}
		dst(i, j) = min_value;
	    }
	}
    }
}
