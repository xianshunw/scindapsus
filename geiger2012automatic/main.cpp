#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "utils.hpp"

int main(int argc, char** argv)
{
    cv::Mat img = cv::imread(argv[1], 0);

    ChessBoardsDetector detector;
    auto ChessBoards = detector.detect(img);

    cv::Mat img_draw;
    cv::cvtColor(img, img_draw, cv::COLOR_GRAY2BGR);

    for (int i = 0; i != ChessBoards.corners.size(); ++i)
    {
        cv::drawChessboardCorners(img_draw, ChessBoards.boards_size[i], ChessBoards.corners[i], true);
    }

    cv::imshow("corners", img_draw);
    cv::waitKey();

    return 0;
}