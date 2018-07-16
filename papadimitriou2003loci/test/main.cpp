#include <loci.hpp>
#include <opencv2/plot.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cvplot/cvplot.h>

int main(int argc, char** argv)
{
    //load data
    cv::Mat dataset;
    loadData(argv[1], dataset);

    // plot all the data points
    std::vector<std::pair<float, float>> data_pts;
    for(int i = 0; i != dataset.rows; ++i)
    {
        float *ptr_data = dataset.ptr<float>(i);
        data_pts.emplace_back(ptr_data[0], ptr_data[1]);
    }
    auto &figure_org = cvplot::figure("original data");
    figure_org.series("org").add(data_pts).type(cvplot::Dots).color(cvplot::Aqua);
    figure_org.show();
    char k = cv::waitKey();

    // detect outliers
    std::vector<bool> result;
    LOCI outlier_detecter(dataset);
    outlier_detecter.detect(result);

    std::vector<std::pair<float, float>> false_pts, true_pts;
    for(int i = 0; i != result.size(); ++i)
    {
        float *ptr_data = dataset.ptr<float>(i);
        result[i] ? false_pts.emplace_back(ptr_data[0], ptr_data[1]) : true_pts.emplace_back(ptr_data[0], ptr_data[1]);
    }
    auto &figure_detect = cvplot::figure("detect result");
    figure_detect.series("true").add(true_pts).type(cvplot::Dots).color(cvplot::Aqua);
    figure_detect.series("false").add(false_pts).type(cvplot::Dots).color(cvplot::Red);
    figure_detect.show();

    k = cv::waitKey();
    return 0;
}