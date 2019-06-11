#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <cmath>
#include <utility>
#include <vector>
#include <unordered_set>

struct CorrelationPatch
{
    cv::Mat A, B, C, D;
};

struct Corners
{
    cv::Mat p;
    cv::Mat v1, v2;
	cv::Mat score;
};

struct ChessBoards
{
    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<cv::Size> boards_size;
};

class ChessBoardsDetector
{
public:
    ChessBoardsDetector();
	ChessBoards detect(cv::Mat& img);
	
private:
    int radius[3];
    cv::Mat du, dv;
    cv::Mat img_du, img_dv, img_angle, img_weight;
    cv::Mat template_props;
    cv::Mat img;
    cv::Mat img_corners;
    Corners corners;
    ChessBoards boards;
    std::vector<cv::Mat> boards_index;
    
    double tau;

    void filter();
    void refine(int r);
    void scoreCorners();
    void chessboardsFromCorners();
    
    // reproduce normpdf of MATLAB
    double normpdf(double x, double mu, double sigma);
    int rank(cv::Mat& m);
    cv::Mat nonMaximumSuppression(cv::Mat& _img, int n, double _tau, int margin);

    void findModesMeanShift(cv::Mat& hist, double sigma, cv::Mat& modes, 
        cv::Mat& hist_smoothed);
    void edgeOrientations(cv::Mat& img_angle_sub, cv::Mat& img_weight_sub, 
        cv::Mat& v1, cv::Mat& v2);
    double cornerCorrelationScore(cv::Mat& img_sub, cv::Mat& img_weight_sub,
        cv::Mat v1, cv::Mat v2);
    void convertAngleRange(cv::Mat& angle);
    void initChessBoard(int idx);
    void directionalNeighbor(int idx, cv::Mat v, cv::Mat& board_index, int* neigh, double *dist);
    double chessboardEnergy(cv::Mat& board_index);
    cv::Mat extractPoints(cv::Mat& point_list, cv::Mat& index);
    cv::Mat extractIndex(std::vector<int>& unused, cv::Mat& index);
    cv::Mat predictCorners(cv::Mat& p1, cv::Mat& p2, cv::Mat& p3);
    cv::Mat growChessboard(cv::Mat& board_index, int border_type);
    cv::Mat assignClosestCorners(cv::Mat& cand, cv::Mat& pred);
    cv::Mat MatAtan2(cv::Mat& x, cv::Mat& y);
    template <typename _T>
    cv::Mat findValue(cv::Mat_<_T>& mat, _T value);
    CorrelationPatch createCorrelationPatch(cv::Mat& template_class);
};

template <typename _T>
cv::Mat ChessBoardsDetector::findValue(cv::Mat_<_T>& mat, _T value)
{
    std::vector<int> index_list;
    for (int i = 0; i != mat.rows; ++i)
    {
        auto *ptr_mat = mat.ptr(i);
        if (ptr_mat[0] == value)
        {
            index_list.push_back(i);
        }
    }

    cv::Mat index = cv::Mat(1, index_list.size(), CV_32SC1);
    auto *ptr_index = index.ptr<int>();
    for (int i = 0; i != index_list.size(); ++i)
    {
        ptr_index[i] = index_list[i];
    }

    return index;
}
