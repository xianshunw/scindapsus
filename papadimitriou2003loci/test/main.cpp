#include <loci.hpp>
#include <opencv2/plot.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

int main(int argc, char** argv)
{
    //load data
    cv::Mat dataset;
    loadData(argv[1], dataset);

    // plot data, the type of input of plot tools must be CV_64F
    cv::Mat data_plot;
    dataset.convertTo(data_plot, CV_64FC1);
    cv::Mat data_img;
    cv::Ptr<cv::plot::Plot2d> plt = cv::plot::Plot2d::create(data_plot.col(0), data_plot.col(1));
    plt->setNeedPlotLine(false);
    plt->setPlotLineWidth(4);
    plt->setGridLinesNumber(0);
    plt->render(data_img);

    cv::imshow("data points", data_img);
    cv::waitKey();
    cv::destroyAllWindows();

    std::vector<bool> result;
    LOCI outlier_detecter(dataset);
    outlier_detecter.detect(result);
    return 0;
}