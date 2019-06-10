#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "utils.hpp"
#include <utility>
#include <cmath>
#include <unordered_set>
#include <map>
#include <cmath>

#include <iostream>

ChessBoardsDetector::ChessBoardsDetector()
{
    // 3 scales
    radius[0] = 4; radius[1] = 8; radius[2] = 12;

    // sobel mask
    du = (cv::Mat_<double>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
    dv = du.t();

    // template properties
    template_props = (cv::Mat_<double>(6, 3) << 0,         CV_PI / 2, radius[0],  
                                              CV_PI / 4, -CV_PI / 4, radius[0],
                                              0,          CV_PI / 2, radius[1],
                                              CV_PI / 4, -CV_PI / 4, radius[1],
                                              0,          CV_PI / 2, radius[2],
                                              CV_PI / 4, -CV_PI / 4, radius[2]);

    tau = 0.01;
};

void ChessBoardsDetector::filter()
{
    img_corners = cv::Mat::zeros(img.size(), CV_64FC1);

    for(int i = 0; i != template_props.rows; ++i)
    {
        // create correlation template
        cv::Mat template_class = template_props.row(i).clone();
        CorrelationPatch template_patch = createCorrelationPatch(template_class);

        // filter image according with current template
        cv::Mat img_corners_A, img_corners_B, img_corners_C, img_corners_D;
        cv::Mat a1, a2, b1, b2;
        cv::flip(template_patch.A, a1, -1);
        cv::flip(template_patch.B, a2, -1);
        cv::flip(template_patch.C, b1, -1);
        cv::flip(template_patch.D, b2, -1);
        cv::Point anchor(a1.cols - a1.cols / 2 - 1, 
            a1.rows - a1.rows / 2 - 1);
        cv::filter2D(img, img_corners_A, CV_64F, a1, anchor, 0.0, cv::BORDER_CONSTANT);
        cv::filter2D(img, img_corners_B, CV_64F, a2, anchor, 0.0, cv::BORDER_CONSTANT);
        cv::filter2D(img, img_corners_C, CV_64F, b1, anchor, 0.0, cv::BORDER_CONSTANT);
        cv::filter2D(img, img_corners_D, CV_64F, b2, anchor, 0.0, cv::BORDER_CONSTANT);

        // compute mean
        cv::Mat img_corners_mu = (img_corners_A + img_corners_B + img_corners_C + img_corners_D) / 4.0;

        // case 1: a = white, b = black
        cv::Mat tmp1 = img_corners_A - img_corners_mu, tmp2 = img_corners_B - img_corners_mu;
        cv::Mat img_corners_a = cv::min(tmp1, tmp2);
        tmp1 = img_corners_mu - img_corners_C; tmp2 = img_corners_mu - img_corners_D;
        cv::Mat img_corners_b = cv::min(tmp1, tmp2);
        cv::Mat img_corners_1 = cv::min(img_corners_a, img_corners_b);

        // case 2: b=white, a=black
        tmp1 = img_corners_mu - img_corners_A; tmp2 = img_corners_mu - img_corners_B;
        img_corners_a = cv::min(tmp1, tmp2);
        tmp1 = img_corners_C - img_corners_mu; tmp2 = img_corners_D - img_corners_mu;
        img_corners_b = cv::min(tmp1, tmp2);
        cv::Mat img_corners_2 = cv::min(img_corners_a, img_corners_b);

        // update corner map
        img_corners = cv::max(img_corners, img_corners_1);
        img_corners = cv::max(img_corners, img_corners_2);
    }

    // extract corner candidates via non maximum suppression
    corners.p = nonMaximumSuppression(img_corners, 3, 0.025, 5);
}

void ChessBoardsDetector::refine(int r)
{
    int width = img_du.cols;
    int height = img_du.rows;

    corners.v1 = cv::Mat::zeros(corners.p.rows, 2, CV_64FC1);
    corners.v2 = cv::Mat::zeros(corners.p.rows, 2, CV_64FC1);

    // scan all corners
    auto *ptr_p = corners.p.ptr<double>();
    auto *ptr_du = img_du.ptr<double>();
    auto *ptr_dv = img_dv.ptr<double>();
    for(int i = 0; i != corners.p.rows; ++i)
    {
        // extract current corner location
        double cu = ptr_p[i * 2];
        double cv = ptr_p[i * 2 + 1];

        // estimate edge orientations
        cv::Mat img_angle_sub = img_angle.rowRange(std::max(int(cv - r), 0), 
            std::min(int(cv + r + 1), height - 1)).colRange(std::max(int(cu - r), 0), 
            std::min(int(cu + r + 1), width - 1)).clone();
        cv::Mat img_weight_sub = img_weight.rowRange(std::max(int(cv - r), 0), 
            std::min(int(cv + r + 1), height - 1)).colRange(std::max(int(cu - r), 0), 
            std::min(int(cu + r + 1), width - 1)).clone();

        cv::Mat v1, v2;
        edgeOrientations(img_angle_sub, img_weight_sub, v1, v2);
        v1.copyTo(corners.v1.row(i));
        v2.copyTo(corners.v2.row(i));

        // continue, if invalid edge orientations
        auto *ptr_v1 = v1.ptr<double>();
        auto *ptr_v2 = v2.ptr<double>();
        if(ptr_v1[0] == 0.0 && ptr_v1[1] == 0.0 || 
            ptr_v2[0] == 0.0 && ptr_v2[1] == 0.0)
        {
            continue;
        }

        /**********************************
         * corner orientation refinement *
         * ********************************/
        cv::Mat A1 = cv::Mat::zeros(2, 2, CV_64FC1);
        cv::Mat A2 = A1.clone();

        int ud = std::max<int>(cu - r, 0), uu = std::min<int>(cu + r, width - 1);
        int vd = std::max<int>(cv - r, 0), vu = std::min<int>(cv + r, height - 1);
        for(int u = ud; u <= uu; ++u)
        {
            for(int v = vd; v <= vu; ++v)
            {
                // pixel orientation vector
                cv::Mat o = (cv::Mat_<double>(1, 2) 
                    << ptr_du[v * width + u], ptr_dv[v * width + u]);

                if(cv::norm(o) < 0.1)
                {
                    continue;
                }

                o /= norm(o);

                // robust refinement of orientation
                if(abs(o.dot(v1)) < 0.25)
                {
                    A1.row(0) = A1.row(0) + ptr_du[v * width + u] * (cv::Mat_<double>(1, 2) 
                        << ptr_du[v * width + u], ptr_dv[v * width + u]);
                    A1.row(1) = A1.row(1) + ptr_dv[v * width + u] * (cv::Mat_<double>(1, 2) 
                        << ptr_du[v * width + u], ptr_dv[v * width + u]);
                }

                if(abs(o.dot(v2)) < 0.25)
                {
                    A2.row(0) = A2.row(0) + ptr_du[v * width + u] * (cv::Mat_<double>(1, 2) 
                        << ptr_du[v * width + u], ptr_dv[v * width + u]);
                    A2.row(1) = A2.row(1) + ptr_dv[v * width + u] * (cv::Mat_<double>(1, 2) 
                        << ptr_du[v * width + u], ptr_dv[v * width + u]);
                }
            }
        }

        // set new corner orientation
        cv::Mat tmp1;
        cv::eigen(A1, tmp1, v1);
        v1 = v1.row(v1.rows - 1).clone();
        v1.copyTo(corners.v1.row(i));
        cv::eigen(A2, tmp1, v2);
        v2 = v2.row(v2.rows - 1).clone();
        v2.copyTo(corners.v2.row(i));

        /**********************************
         *   corner location refinement   *
         * ********************************/
        cv::Mat G = cv::Mat::zeros(2, 2, CV_64FC1);
        cv::Mat b = cv::Mat::zeros(2, 1, CV_64FC1);

        for(int u = ud; u <= uu; ++u)
        {
            for(int v = vd; v <= vu; ++v)
            {
                // pixel orientation vector
                cv::Mat o = (cv::Mat_<double>(1, 2) 
                    << ptr_du[v * width + u], ptr_dv[v * width + u]);

                if(cv::norm(o) < 0.1)
                {
                    continue;
                }

                o /= norm(o);

                // robust subpixel corner estimation
                if(u != int(cu) || v != int(cv))
                {
                    // compute rel. position of pixel and distance to vectors
                    cv::Mat w = (cv::Mat_<double>(1, 2) << (u - cu), (v - cv));
                    double d1 = cv::norm(w - w*v1.t()*v1);
                    double d2 = cv::norm(w - w*v2.t()*v2);

                    if(d1 < 3 && std::abs(o.dot(v1)) < 0.25 || 
                        d2 < 3 && std::abs(o.dot(v2)) < 0.25)
                    {
                        double du_uv = ptr_du[v * width + u];
                        double dv_uv = ptr_dv[v * width + u];

                        cv::Mat H = (cv::Mat_<double>(2, 1) << du_uv, dv_uv) * 
                            (cv::Mat_<double>(1, 2) << du_uv, dv_uv);
                        G = G + H;
                        b = b + H*(cv::Mat_<double>(2, 1) << u, v);
                    }
                }
            }
        }

        int rG = rank(G);
        if(rG == 2)
        {
            cv::Mat corners_pos_old = corners.p.row(i).clone();
            cv::Mat corners_pos_new;
            cv::transpose(G.inv() * b, corners_pos_new);
            corners_pos_new.copyTo(corners.p.row(i));

            // set corner to invalid, if position update is very large
            if(cv::norm(corners_pos_new - corners_pos_old) > 4.0f)
            {
                cv::Mat tmp = cv::Mat::zeros(1, 2, CV_64FC1);
                tmp.copyTo(corners.v1.row(i));
                tmp.copyTo(corners.v2.row(i));
            } 
        }
        else    // otherwise: set corner to invalid
        {
            cv::Mat tmp = cv::Mat::zeros(1, 2, CV_64FC1);
            tmp.copyTo(corners.v1.row(i));
            tmp.copyTo(corners.v2.row(i));
        }
    }
}

void ChessBoardsDetector::scoreCorners()
{
    double width = img.cols;
    double height = img.rows;

    auto *ptr_p = corners.p.ptr<double>();
    double score[3] = { 0 };
	corners.score = cv::Mat::zeros(corners.p.rows, 1, CV_64FC1);
	auto *ptr_score = corners.score.ptr<double>();
    for(int i = 0; i != corners.p.rows; ++i)
    {
        int u = std::round(ptr_p[2 * i]);
        int v = std::round(ptr_p[2 * i + 1]);

        for(int j = 0; j != 3; ++j)
        {
            score[j] = 0;
            if(u > radius[j] && u < width - radius[j] && 
               v > radius[j] && v < height - radius[j])
            {
                cv::Mat img_sub = img.rowRange(v - radius[j], 
                    v + radius[j] + 1).colRange(u - radius[j], 
                    u + radius[j] + 1).clone();//img(v-radius(j):v+radius(j),u-radius(j):u+radius(j));
                cv::Mat img_angle_sub  = img_angle.rowRange(v - radius[j], 
                    v + radius[j] + 1).colRange(u - radius[j], 
                    u + radius[j] + 1).clone();
                cv::Mat img_weight_sub = img_weight.rowRange(v - radius[j], 
                    v + radius[j] + 1).colRange(u - radius[j], 
                    u + radius[j] + 1).clone();
                score[j] = cornerCorrelationScore(img_sub, img_weight_sub, corners.v1.row(i), corners.v2.row(i));
            }
        }
        
        double max_score = score[0] > score[1] ? (score[0] > score[2] ? score[0] : score[2]) : 
                                                (score[1] > score[2] ? score[1] : score[2]);
        ptr_score[i] = max_score;
    }
}

void ChessBoardsDetector::chessboardsFromCorners()
{
    // intialize chessboards
    std::vector<cv::Mat> chessboards;

    // for all seed corners do
    for(int i = 0; i != corners.p.rows; ++i)
    {
        // init 3x3 chessboard from seed i
        int size_before = boards_index.size();
        initChessBoard(i);

        // check if this is a useful initial guess
        if(boards_index.size() <= size_before ||
            chessboardEnergy(boards_index[size_before]) > 0)
        {
            continue;
        }

        // std::cout << i << std::endl;

        //if (i == 137 || i == 121)
        //{
        //    int debug_flag = 0;
        //}

        // try growing chessboard
        while(1)
        {
            // compute current energy
            double energy = chessboardEnergy(boards_index[size_before]);

            // compute proposals and energies
            std::vector<cv::Mat> proposal(4);
            std::vector<double> p_energy(4);
            for(int j = 0; j != 4; ++j)
            {
                proposal[j] = growChessboard(boards_index[size_before], j);
                p_energy[j] = chessboardEnergy(proposal[j]);
            }

            int min_idx;
            double min_val = std::numeric_limits<double>::max();
            for (int j = 0; j != 4; ++j)
            {
                if (p_energy[j] < min_val)
                {
                    min_val = p_energy[j];
                    min_idx = j;
                }
            }

            if (min_val < energy)
            {
                boards_index[size_before] = proposal[min_idx];
            }
            else // otherwise exit loop
            {
                break;
            }
            // auto iter_min = std::min_element(p_energy.begin(), p_energy.end());
        }

        if (chessboardEnergy(boards_index[size_before]) < -10)
        {
            cv::Mat overlap = cv::Mat::zeros(chessboards.size(), 2, CV_64FC1);
            for (int j = 0; j != chessboards.size(); ++j)
            {
                auto *ptr_overlap = overlap.ptr<double>(j);
                int element_size = chessboards[j].rows * chessboards[j].cols;
                int curr_element_size = boards_index[size_before].rows * boards_index[size_before].cols;
                auto *ptr_curr_board = boards_index[size_before].ptr<int>();
                auto *ptr_chessboard_j = chessboards[j].ptr<int>();
                for (int k = 0; k != element_size; ++k)
                {
                    auto iter = std::find(ptr_curr_board, ptr_curr_board + curr_element_size, ptr_chessboard_j[k]);
                    if (iter != ptr_curr_board + curr_element_size)
                    {
                        ptr_overlap[0] = 1;
                        ptr_overlap[1] = chessboardEnergy(chessboards[j]);
                        break;
                    }
                }
            }

            // add chessboard (and replace overlapping if neccessary)
            if (!cv::countNonZero(overlap.col(0)))
            {
                chessboards.push_back(boards_index[size_before].clone());
            }
            else
            {
                cv::Mat_<double> overlap_f = overlap;
                cv::Mat index = findValue(overlap_f, 1.0);
                int element_num = index.rows * index.cols;
                auto *ptr_index = index.ptr<int>();
                auto *ptr_overlap = overlap_f.ptr<double>();
                for (int j = 0; j != element_num; ++j)
                {
                    if (!(ptr_overlap[2 * ptr_index[j] + 1] <= chessboardEnergy(boards_index[size_before])))
                    {
                        chessboards.erase(chessboards.begin() + ptr_index[j]);
                        chessboards.push_back(boards_index[size_before].clone());
                        break;
                    }
                    
                }
            }
        }
    }

    boards_index = chessboards;
}

ChessBoards ChessBoardsDetector::detect(cv::Mat& _img)
{
    if (_img.channels() == 3)
    {
        cv::cvtColor(_img, img, CV_BGR2GRAY);
    }
    else
    {
        img = _img.clone();
    }
    img.convertTo(img, CV_64F, 1.0 / 255.0);

    // compute image derivatives (for principal axes estimation)
    cv::flip(du, du, -1);
    cv::flip(dv, dv, -1);
    cv::Point anchor(du.cols - du.cols / 2 - 1,
        du.rows - du.rows / 2 - 1);
    cv::filter2D(img, img_du, CV_64F, du, anchor, 0.0, cv::BORDER_CONSTANT);
    anchor.x = dv.cols - dv.cols / 2 - 1;
    anchor.y = dv.rows - dv.rows / 2 - 1;
    cv::filter2D(img, img_dv, CV_64F, dv, anchor, 0.0, cv::BORDER_CONSTANT);
    img_angle = MatAtan2(img_dv, img_du);
    cv::Mat tmp1, tmp2;
    cv::pow(img_du, 2, tmp1);
    cv::pow(img_dv, 2, tmp2);
    cv::sqrt(tmp1 + tmp2, img_weight);

    // scale input image
    double img_min, img_max;
    cv::minMaxIdx(img, &img_min, &img_max);
    img = (img - img_min) / (img_max - img_min);

    filter();
    refine(10);

    // remove corners without edges
    auto *ptr_v1 = corners.v1.ptr<double>();
    // auto *ptr_v2 = corners.v2.ptr<double>();
    std::vector<int> valid_idx;
    for(int i = 0; i != corners.v1.rows; ++i)
    {
        if(ptr_v1[2 * i] != 0.0 || ptr_v1[2 * i + 1] != 0.0)
        {
            valid_idx.push_back(i);
        }
    }

    cv::Mat ptmp(valid_idx.size(), 2, CV_64FC1);
    cv::Mat v1tmp(valid_idx.size(), 2, CV_64FC1);
    cv::Mat v2tmp(valid_idx.size(), 2, CV_64FC1);

    for(int i = 0; i != valid_idx.size(); ++i)
    {
        corners.p.row(valid_idx[i]).copyTo(ptmp.row(i));
        corners.v1.row(valid_idx[i]).copyTo(v1tmp.row(i));
        corners.v2.row(valid_idx[i]).copyTo(v2tmp.row(i));
    }

    corners.p = ptmp;
    corners.v1 = v1tmp;
    corners.v2 = v2tmp;
    
    // score corners
    scoreCorners();
    
    // remove low scoring corners
    valid_idx.clear();
    auto *ptr_score = corners.score.ptr<double>();
    for(int i = 0; i != corners.p.rows; ++i)
    {
        if(ptr_score[i] >= tau)
        {
            valid_idx.push_back(i);
        }
    }
    
    cv::Mat tmp_p = cv::Mat(valid_idx.size(), 2, CV_64FC1);
    cv::Mat tmp_v1 = cv::Mat(valid_idx.size(), 2, CV_64FC1);
    cv::Mat tmp_v2 = cv::Mat(valid_idx.size(), 2, CV_64FC1);
    cv::Mat tmp_score = cv::Mat(valid_idx.size(), 1, CV_64FC1);

    for(size_t i = 0; i != valid_idx.size(); ++i)
    {
        corners.p.row(valid_idx[i]).copyTo(tmp_p.row(i));
        corners.v1.row(valid_idx[i]).copyTo(tmp_v1.row(i));
        corners.v2.row(valid_idx[i]).copyTo(tmp_v2.row(i));
        corners.score.row(valid_idx[i]).copyTo(tmp_score.row(i));
    }
    
    corners.p = tmp_p;
    corners.v1 = tmp_v1;
    corners.v2 = tmp_v2;
    corners.score = tmp_score;
    
    // make v1(:,1)+v1(:,2) positive (=> comparable to c++ code)
    for(int i = 0; i != corners.v1.rows; ++i)
    {
        auto *ptr_v1 = corners.v1.ptr<double>(i);
        double sum_v1_row = ptr_v1[0] + ptr_v1[1];
        if(sum_v1_row < 0)
        {
            ptr_v1[0] = -ptr_v1[0];
            ptr_v1[1] = -ptr_v1[1];
        }
    }
    
    // make all coordinate systems right-handed 
    // (reduces matching ambiguities from 8 to 4)
    cv::Mat corners_n1 = corners.v1.clone();
    for(int i = 0; i != corners_n1.rows; ++i)
    {
        auto *ptr_n1 = corners_n1.ptr<double>(i);
        double tmp = ptr_n1[0];
        ptr_n1[0] = ptr_n1[1];
        ptr_n1[1] = -tmp;
    }

    cv::Mat sign_value = corners_n1.col(0).mul(corners.v2.col(0)) + 
        corners_n1.col(1).mul(corners.v2.col(1));

    cv::Mat flip = cv::Mat::zeros(corners_n1.rows, 1, CV_64FC1);
    auto *ptr_flip = flip.ptr<double>();
    auto *ptr_sign = sign_value.ptr<double>();
    for(int i = 0; i != corners_n1.rows; ++i)
    {
        ptr_flip[i] = ptr_sign[i] < 0.0 ? 1.0 : -1.0;
    }

    cv::Mat tmp_mat = flip * cv::Mat::ones(1, 2, CV_64FC1);
    corners.v2 = corners.v2.mul(tmp_mat);

    chessboardsFromCorners();

    ChessBoards CB;

    auto* ptr_points = corners.p.ptr<double>();
    for (int i = 0; i != boards_index.size(); ++i)
    {
        std::vector<cv::Point2f> corners_i;
        int corners_num = boards_index[i].rows * boards_index[i].cols;
        auto* ptr_board_index = boards_index[i].ptr<int>();
        for (int j = 0; j != corners_num; ++j)
        {
            int index = ptr_board_index[j];
            corners_i.push_back(cv::Point2f(ptr_points[index * 2], ptr_points[index * 2 + 1]));
        }

        CB.corners.push_back(corners_i);
        CB.boards_size.push_back(boards_index[i].size());
    }

    return CB;
}

double ChessBoardsDetector::normpdf(double x, double mu, double sigma)
{
    return std::exp(-0.5 * std::pow(((x - mu)/sigma), 2)) / (std::sqrt(2*CV_PI) * sigma);
}

int ChessBoardsDetector::rank(cv::Mat& m)
{
    /* https://stackoverflow.com/questions/37898019/how-to-calculate-matrix-rank-in-opencv */
    cv::Mat w, u, vt;
    cv::SVD::compute(m, w, u, vt);
    cv::Mat non_zeros = w > 0.0001;

    return cv::countNonZero(non_zeros);
}

void ChessBoardsDetector::findModesMeanShift(cv::Mat& hist, double sigma, cv::Mat& modes, 
        cv::Mat& hist_smoothed)
{
    // compute smoothed histogram
    hist_smoothed = cv::Mat::zeros(hist.size(), CV_64FC1);
    auto *ptr_sm = hist_smoothed.ptr<double>();
    auto *ptr_hi = hist.ptr<double>();
    for(int i = 0; i <= hist.cols; ++i)
    {
        // std::vector<int> idx;
        for(int j = -std::round(2 * sigma); j <= std::round(2 * sigma); ++j)
        {
            int idx = (i + j) % hist.cols;
            if(idx < 0) idx += hist.cols;
            ptr_sm[i] += ptr_hi[idx] * normpdf(j, 0, sigma);
        }
    }

    // check if at least one entry is non-zero
    // (otherwise mode finding may run infinitly)
    int flag = 0;
    for(int i = 0; i != hist_smoothed.cols; ++i)
    {
        flag += int(std::abs(ptr_sm[i] - ptr_sm[0]) > 1e-5);
    }
    if(flag == 0) return;

    // mode finding
    std::map<double, double> mode_pair;
    for(int i = 0; i != hist_smoothed.cols; ++i)
    {
        int j = i;
        while(true)
        {
            double h0 = ptr_sm[j];
            int j1 = (j + 1) % hist.cols;
            int j2 = (j - 1) % hist.cols;
            if(j1 < 0) j1 += hist.cols;
            if(j2 < 0) j2 += hist.cols;
            double h1 = ptr_sm[j1];
            double h2 = ptr_sm[j2];
            if(h1 >= h0 && h1 >= h2)
            {
                j = j1;
            }
            else if(h2 > h0 && h2 > h1)
            {
                j = j2;
            }
            else
            {
                break;
            }
        }

        if (mode_pair.find(double(j)) == mode_pair.end())
        {
            mode_pair[j] = ptr_sm[j];
        }
    }

    std::vector<std::pair<double, double>> mode_vector;
    std::copy(mode_pair.begin(), mode_pair.end(),
        std::back_inserter<std::vector<std::pair<double, double>>>(mode_vector));

    std::sort(mode_vector.begin(), mode_vector.end(), 
        [](const std::pair<double, double>& l, 
        const std::pair<double, double>& r) { return l.second > r.second; });

    modes = cv::Mat(mode_pair.size(), 2, CV_64FC1);
    auto *ptr_md = modes.ptr<double>();
    for(int i = 0; i != mode_vector.size(); ++i)
    {
        ptr_md[i * 2] = mode_vector[i].first;
        ptr_md[i * 2 + 1] = mode_vector[i].second;
    }
}

cv::Mat ChessBoardsDetector::nonMaximumSuppression(cv::Mat& _img, int n, double _tau, int margin)
{
    // extract parameters
    int width  = _img.cols;
    int height = _img.rows;

    std::vector<std::pair<double, double>> maxima_pair;
    auto *ptr_img = _img.ptr<double>();
    for(int i = n + margin; i < width - n - margin - 1; i = i + n + 1)
    {
        for(int j = n + margin; j < height - n - margin - 1; j = j + n + 1)
        {
            int maxi = i;
            int maxj = j;
            
            int idx_ij = j * width + i;
            double max_val = ptr_img[idx_ij];

            for(int ii = i; ii <= i + n; ++ii)
            {
                for(int jj = j; jj <= j + n; ++jj)
                {
                    int idx_ij2 = jj * width + ii;
                    double curr_val = ptr_img[idx_ij2];
                    if(curr_val > max_val)
                    {
                        maxi = ii;
                        maxj = jj;
                        max_val = curr_val;
                    }
                }
            }

            bool failed = false;
            int margin_ii = std::min(maxi + n, width - margin - 1);
            int margin_jj = std::min(maxj + n, height - margin - 1);
            for(int ii = maxi - n; ii <= margin_ii; ++ii)
            {
                for(int jj = maxj - n; jj <= margin_jj; ++jj)
                {
                    int idx_ij2 = jj * width + ii;
                    double currval = ptr_img[idx_ij2];
                    if(currval > max_val && (ii < i || ii > i + n || jj < j || jj > j + n))
                    {
                        failed = true;
                        break;
                    }
                }

                if(failed) break;
            }

            if(max_val >= _tau && !failed)
            {
                maxima_pair.push_back(std::make_pair(maxi, maxj));
            }
        }
    }

    cv::Mat maxima(maxima_pair.size(), 2, CV_64FC1);
    auto *ptr_maxima = maxima.ptr<double>();
    for(int i = 0; i != maxima_pair.size(); ++i)
    {
        ptr_maxima[i * 2] = maxima_pair[i].first;
        ptr_maxima[i * 2 + 1] = maxima_pair[i].second;
    }

    return maxima;
}

CorrelationPatch ChessBoardsDetector::createCorrelationPatch(cv::Mat& template_class)
{
    auto *ptr = template_class.ptr<double>();
    int radius = ptr[2];
    double angle1 = ptr[0], angle2 = ptr[1];

    int width  = radius*2 + 1;
    int height = radius*2 + 1;

    // initialize template
    CorrelationPatch patch;
    patch.A = cv::Mat::zeros(height, width, CV_64FC1);
    patch.B = cv::Mat::zeros(height, width, CV_64FC1);
    patch.C = cv::Mat::zeros(height, width, CV_64FC1);
    patch.D = cv::Mat::zeros(height, width, CV_64FC1);
    auto *ptr_A = patch.A.ptr<double>();
    auto *ptr_B = patch.B.ptr<double>();
    auto *ptr_C = patch.C.ptr<double>();
    auto *ptr_D = patch.D.ptr<double>();

    //  midpoint
    int mu = radius + 1;
    int mv = radius + 1;

    // compute normals from angles
    cv::Mat n1 = (cv::Mat_<double>(1, 2) << -std::sin(angle1), std::cos(angle1));
    cv::Mat n2 = (cv::Mat_<double>(1, 2) << -std::sin(angle2), std::cos(angle2));

    // for all points in template do
    double sum_A = 0, sum_B = 0, sum_C = 0, sum_D = 0;
    for(int u = 0; u != width; ++u)
    {
        for(int v = 0; v != height; ++v)
        {
            cv::Mat vec = (cv::Mat_<double>(1, 2) << u + 1 - mu, v + 1 - mv);
            double dist = cv::norm(vec);

            // check on which side of the norms we are
            cv::Mat s1_mat = vec * n1.t();
            cv::Mat s2_mat = vec * n2.t();
            double s1 = s1_mat.at<double>(0, 0);
            double s2 = s2_mat.at<double>(0, 0);

            int idx = v * width + u;
            if(s1 <= -0.1 && s2 <= -0.1)
            {
                ptr_A[idx] = normpdf(dist, 0, radius / 2.0f);
                sum_A += ptr_A[idx];
            }
            else if(s1 >= 0.1 && s2 >= 0.1)
            {
                ptr_B[idx] = normpdf(dist, 0, radius / 2.0f);
                sum_B += ptr_B[idx];
            }
            else if(s1 <= -0.1 && s2 >= 0.1)
            {
                ptr_C[idx] = normpdf(dist, 0, radius / 2.0f);
                sum_C += ptr_C[idx];
            }
            else if(s1 >= 0.1 && s2 <= -0.1)
            {
                ptr_D[idx] = normpdf(dist, 0, radius / 2.0f);
                sum_D += ptr_D[idx];
            }
        }
    }

    // normalize
    patch.A /= sum_A;
    patch.B /= sum_B;
    patch.C /= sum_C;
    patch.D /= sum_D;

    return patch;
}

void ChessBoardsDetector::edgeOrientations(cv::Mat& img_angle_sub, cv::Mat& img_weight_sub, 
    cv::Mat& v1, cv::Mat& v2)
{
    // init v1 and v2
    v1 = cv::Mat::zeros(1, 2, CV_64FC1);
    v2 = cv::Mat::zeros(1, 2, CV_64FC1);

    // number of bins
    int bin_num = 32;

    // convert angles from normals to directions
    img_angle_sub += CV_PI / 2;
    convertAngleRange(img_angle_sub);
    cv::Mat vec_angle, vec_weight;
    cv::transpose(img_angle_sub, vec_angle);
    cv::transpose(img_weight_sub, vec_weight);

    // create histogram
    cv::Mat angle_hist = cv::Mat::zeros(1, bin_num, CV_64FC1);
    auto *ptr_hist = angle_hist.ptr<double>();
    int pnum = vec_angle.rows * vec_angle.cols;
    auto *ptr_angle = vec_angle.ptr<double>();
    auto *ptr_weight = vec_weight.ptr<double>();
    double rad_per_bin = CV_PI / bin_num;
    for(int i = 0; i != pnum; ++i)
    {
        int bin = std::max(std::min(int(std::floor(ptr_angle[i] / rad_per_bin)), bin_num - 1), 0);
        ptr_hist[bin] = ptr_hist[bin] + ptr_weight[i];
    }

    // find modes of smoothed histogram
    cv::Mat modes, angle_hist_smoothed;
    findModesMeanShift(angle_hist, 1, modes, angle_hist_smoothed);

    // if only one or no mode => return invalid corner
    if(modes.rows <= 1)
    {
        return;
    }

    // compute orientation at modes
    cv::Mat modes_ext(modes.rows, 3, CV_64FC1);
    modes.copyTo(modes_ext.colRange(0, 2));
    cv::Mat tmp = modes.col(0) * CV_PI / bin_num;
    tmp.copyTo(modes_ext.col(2));

    // extract 2 strongest modes and sort by angle
    modes_ext = modes_ext.rowRange(0, 2);
    modes = modes_ext.clone();
    if(modes_ext.at<double>(0, 2) > modes_ext.at<double>(1, 2))
    {
        modes_ext.row(0).copyTo(modes.row(1));
        modes_ext.row(1).copyTo(modes.row(0));
    }

    // compute angle between modes
    double delta_angle = std::min<double>(modes.at<double>(1, 2) - modes.at<double>(0, 2), 
        modes.at<double>(0, 2) + CV_PI - modes.at<double>(1, 2));

    // if angle too small => return invalid corner
    if(delta_angle <= 0.3)
    {
        return;
    }

    // set statistics: orientations
    auto *ptr_v1 = v1.ptr<double>();
    auto *ptr_v2 = v2.ptr<double>();
    ptr_v1[0] = std::cos(modes.at<double>(0, 2));
    ptr_v1[1] = std::sin(modes.at<double>(0, 2));
    ptr_v2[0] = std::cos(modes.at<double>(1, 2));
    ptr_v2[1] = std::sin(modes.at<double>(1, 2));
}

double ChessBoardsDetector::cornerCorrelationScore(cv::Mat& img_sub, cv::Mat& img_weight_sub,
    cv::Mat v1, cv::Mat v2)
{
    //  center
    cv::Mat c = cv::Mat::ones(1, 2, CV_64FC1) * (img_weight_sub.rows + 1) * 0.5;

    // compute gradient filter kernel (bandwith = 3 px)
    cv::Mat img_filter = cv::Mat(img_weight_sub.size(), CV_64FC1, cv::Scalar(-1.0));
    auto *ptr_filter = img_filter.ptr<double>();
    for(int x = 0; x != img_weight_sub.cols; ++x)
    {
        for(int y = 0; y != img_weight_sub.rows; ++y)
        {
            cv::Mat p1 = (cv::Mat_<double>(1, 2) << x + 1, y + 1) - c;
            cv::Mat p2 = p1 * v1.t() * v1;
            cv::Mat p3 = p1 * v2.t() * v2;

            if(cv::norm(p1 - p2) <= 1.5 || 
                cv::norm(p1 - p3) <= 1.5)
            {
                ptr_filter[y * img_filter.cols + x] = 1.0;
            }
        }
    }

    // norm
    cv::Scalar std_dev, mean_value;
    cv::meanStdDev(img_weight_sub, mean_value, std_dev);
    img_weight_sub = (img_weight_sub - mean_value[0]) / std_dev[0];
    cv::meanStdDev(img_filter, mean_value, std_dev);
    img_filter = (img_filter - mean_value[0]) / std_dev[0];

    // compute gradient score
    cv::Mat tmp = img_weight_sub.mul(img_filter);
    double tmp_sum = cv::sum(tmp)[0] / (tmp.rows * tmp.cols);
    double score_gradient = tmp_sum > 0.0 ? tmp_sum : 0.0;

    // create intensity filter kernel
    auto *ptr_v1 = v1.ptr<double>();
    auto *ptr_v2 = v2.ptr<double>();
    auto *ptr_c = c.ptr<double>();
    cv::Mat template_class = (cv::Mat_<double>(1, 3) << 
        std::atan2(ptr_v1[1], ptr_v1[0]), 
        std::atan2(ptr_v2[1], ptr_v2[0]),
        ptr_c[0] - 1.0);
    auto template_patch = createCorrelationPatch(template_class);

    // checkerboard responses
    tmp = template_patch.A.mul(img_sub);
    double a1 = cv::sum(tmp)[0];
    tmp = template_patch.B.mul(img_sub);
    double a2 = cv::sum(tmp)[0];
    tmp = template_patch.C.mul(img_sub);
    double b1 = cv::sum(tmp)[0];
    tmp = template_patch.D.mul(img_sub);
    double b2 = cv::sum(tmp)[0];

    // mean
    double mu = (a1 + a2 + b1 + b2)/4;

    // case 1: a=white, b=black
    double score_a = std::min<double>(a1 - mu, a2 - mu);
    double score_b = std::min<double>(mu - b1, mu - b2);
    double score_1 = std::min<double>(score_a, score_b);

    // case 2: b=white, a=black
    score_a = std::min<double>(mu - a1, mu - a2);
    score_b = std::min<double>(b1 - mu, b2 - mu);
    double score_2 = std::min<double>(score_a, score_b);

    // intensity score: max. of the 2 cases
    double score_intensity = std::max<double>(std::max<double>(score_1, score_2), 0);
    
    // final score: product of gradient and intensity score
    double score = score_gradient * score_intensity;
    return score;
}

void ChessBoardsDetector::convertAngleRange(cv::Mat& angle)
{
    for(int i = 0; i != angle.rows; ++i)
    {
        auto *ptr = angle.ptr<double>(i);
        for(int j = 0; j != angle.cols; ++j)
        {
            if(ptr[j] > CV_PI) ptr[j] -= CV_PI;
        }
    }
}

void ChessBoardsDetector::initChessBoard(int idx)
{
    //  return if not enough corners
    if(corners.p.rows < 9)
    {
        return;
    }

    // init chessboard hypothesis
    boards_index.push_back(cv::Mat::ones(3, 3, CV_32SC1) * -1);
    int last_element = boards_index.size() - 1;
    auto *ptr_board_index = boards_index[last_element].ptr<int>();

    cv::Mat v1 = corners.v1.row(idx).clone();
    cv::Mat v2 = corners.v2.row(idx).clone();
    ptr_board_index[4] = idx;

    // find left/right/top/bottom neighbor
    cv::Mat dist1 = cv::Mat(1, 2, CV_64FC1);
    cv::Mat dist2 = cv::Mat(1, 6, CV_64FC1);
    auto *ptr_dist1 = dist1.ptr<double>();
    auto *ptr_dist2 = dist2.ptr<double>();
    directionalNeighbor(idx, v1, boards_index[last_element], ptr_board_index + 5, ptr_dist1);
    directionalNeighbor(idx, -v1, boards_index[last_element], ptr_board_index + 3, ptr_dist1 + 1);
    directionalNeighbor(idx, v2, boards_index[last_element], ptr_board_index + 7, ptr_dist2);
    directionalNeighbor(idx, -v2, boards_index[last_element], ptr_board_index + 1, ptr_dist2 + 1);

    // find top-left/top-right/bottom-left/bottom-right neighbors
    directionalNeighbor(ptr_board_index[3], -v2, boards_index[last_element], ptr_board_index, ptr_dist2 + 2);
    directionalNeighbor(ptr_board_index[3], v2, boards_index[last_element], ptr_board_index + 6, ptr_dist2 + 3);
    directionalNeighbor(ptr_board_index[5], -v2, boards_index[last_element], ptr_board_index + 2, ptr_dist2 + 4);
    directionalNeighbor(ptr_board_index[5], v2, boards_index[last_element], ptr_board_index + 8, ptr_dist2 + 5);

    // initialization must be homogenously distributed
    for(int i = 0; i != 2; ++i)
    {
        if(ptr_dist1[i] == std::numeric_limits<double>::max())
        {
            boards_index.erase(boards_index.begin() + last_element);
            return;
        }
    }

    for (int i = 0; i != 6; ++i)
    {
        if (ptr_dist2[i] == std::numeric_limits<double>::max())
        {
            boards_index.erase(boards_index.begin() + last_element);
            return;
        }
    }

    cv::Scalar mean_dist, std_dist;
    cv::meanStdDev(dist1, mean_dist, std_dist);
    if(std_dist[0] / mean_dist[0] > 0.3)
    {
        boards_index.erase(boards_index.begin() + last_element);
        return;
    }

    cv::meanStdDev(dist2, mean_dist, std_dist);
    if(std_dist[0] / mean_dist[0] > 0.3)
    {
        boards_index.erase(boards_index.begin() + last_element);
        return;
    }

//    boards.push_back(cv::Mat(boards_index[0].size(), CV_64FC2));
//    auto *ptr_corner_p = corners.p.ptr<double>();
//    for(int i = 0; i != boards[0].rows; ++i)
//    {
//        auto *ptr_board = boards[0].ptr<double>(i);
//        for(int j = 0; j != boards[0].cols; ++j)
//        {
//            int idx = ptr_board_index[i * boards_index[0].cols + j];
//            ptr_board[j * 2] = ptr_corner_p[idx * 2];
//            ptr_board[j * 2 + 1] = ptr_corner_p[idx * 2 + 1];
//        }
//    }
}

double ChessBoardsDetector::chessboardEnergy(cv::Mat& board_index)
{
    // energy: number of corners
    double energy_corners = -board_index.rows * board_index.cols;

    // energy: structure
    double E_structure = 0.0;

    // walk though rows
    for(int i = 0; i != board_index.rows; ++i)
    {
        for(int j = 0; j != board_index.cols - 2; ++j)
        {
            cv::Mat x(3, 2, CV_64FC1);
            auto *ptr_board_index = board_index.ptr<int>(i);
            for(int k = j, row_x = 0; k != j + 3; ++k, ++row_x)
            {
                //cv::Mat x = cv::Mat(3, 2, CV_64FC1);
                auto *ptr_x = x.ptr<double>(row_x);
                int pt_idx = ptr_board_index[k];
                auto *ptr_pt = corners.p.ptr<double>(pt_idx);
                ptr_x[0] = ptr_pt[0];
                ptr_x[1] = ptr_pt[1];
                //cv::Mat x = corners.p.row(j).colRange(k, k + 2).clone();
            }

            E_structure = std::max<double>(E_structure,
                    cv::norm(x.row(0) + x.row(2) - 2 * x.row(1)) /
                    cv::norm(x.row(0) - x.row(2)));
        }
    }

    // walk through columns
    for(int i = 0; i != board_index.cols; ++i)
    {
        for(int j = 0; j != board_index.rows - 2; ++j)
        {
            cv::Mat x(3, 2, CV_64FC1);
            for(int k = j, row_x = 0; k != j + 3; ++k, ++row_x)
            {
                auto *ptr_x = x.ptr<double>(row_x);
                auto *ptr_board_index = board_index.ptr<int>(k);
                int pt_idx = ptr_board_index[i];
                auto *ptr_pt = corners.p.ptr<double>(pt_idx);
                ptr_x[0] = ptr_pt[0];
                ptr_x[1] = ptr_pt[1];
            }

            E_structure = std::max<double>(E_structure,
                                          cv::norm(x.row(0) + x.row(2) - 2 * x.row(1)) /
                                          cv::norm(x.row(0) - x.row(2)));
        }
    }

    // final energy
    double E = energy_corners + 1.0 * board_index.rows * board_index.cols * E_structure;

    return E;
}

cv::Mat ChessBoardsDetector::extractPoints(cv::Mat& point_list, cv::Mat& index)
{
    int num = index.rows > index.cols ? index.rows : index.cols;

    cv::Mat points(num, 2, CV_64FC1);
    CV_Assert(index.rows == 1 || index.cols == 1);

    if(!index.isContinuous())
    {
        index = index.clone();
    }

    auto *ptr_index = index.ptr<int>();
    for(int i = 0; i != num; ++i)
    {
        int id = ptr_index[i];
        auto *ptr_points = points.ptr<double>(i);
        auto *ptr_point_list = point_list.ptr<double>(id);
        ptr_points[0] = ptr_point_list[0];
        ptr_points[1] = ptr_point_list[1];
    }

    return points;
}

cv::Mat ChessBoardsDetector::extractIndex(std::vector<int>& unused, cv::Mat& index)
{
    cv::Mat index_results = index.clone();
    auto *ptr_index_result = index_results.ptr<int>();
    int ele_num = index.cols * index.rows;
    for (int i = 0; i != ele_num; ++i)
    {
        auto iter = unused.begin();
        std::advance(iter, ptr_index_result[i]);
        ptr_index_result[i] = *iter;
    }

    return index_results;
}

cv::Mat ChessBoardsDetector::predictCorners(cv::Mat& p1, cv::Mat& p2, cv::Mat& p3)
{
    // compute vectors
    cv::Mat v1 = p2 - p1;
    cv::Mat v2 = p3 - p2;

    // predict angles
    cv::Mat a1 = cv::Mat(p1.rows, 1, CV_64FC1);
    cv::Mat a2 = cv::Mat(p1.rows, 1, CV_64FC1);
    cv::Mat a3 = cv::Mat(p1.rows, 1, CV_64FC1);
    auto *ptr_a1 = a1.ptr<double>();
    auto *ptr_a2 = a2.ptr<double>();
    auto *ptr_a3 = a3.ptr<double>();
    cv::Mat cos_sin_a3 = cv::Mat(p1.rows, 2, CV_64FC1);
    auto *ptr_cs_a3 = cos_sin_a3.ptr<double>();
    for(int i = 0; i != p1.rows; ++i)
    {
        auto *ptr_v1 = v1.ptr<double>(i);
        auto *ptr_v2 = v2.ptr<double>(i);
        ptr_a1[i] = std::atan2(ptr_v1[1], ptr_v1[0]);
        ptr_a2[i] = std::atan2(ptr_v2[1], ptr_v2[0]);
        ptr_a3[i] = 2 * ptr_a2[i] - ptr_a1[i];
        ptr_cs_a3[i * 2] = std::cos(ptr_a3[i]);
        ptr_cs_a3[i * 2 + 1] = std::sin(ptr_a3[i]);
    }

    // predict scales
    cv::Mat tmp1, tmp2, s1, s2, s3;

    cv::pow(v1.col(0), 2, tmp1);
    cv::pow(v1.col(1), 2, tmp2);
    cv::sqrt(tmp1 + tmp2, s1);

    cv::pow(v2.col(0), 2, tmp1);
    cv::pow(v2.col(1), 2, tmp2);
    cv::sqrt(tmp1 + tmp2, s2);

    s3 = 2 * s2 - s1;

    //  predict p3 (the factor 0.75 ensures that under extreme
    // distortions (omnicam) the closer prediction is selected)
    cv::Mat pred = 0.75 * s3 * cv::Mat::ones(1, 2, CV_64FC1);
    pred = pred.mul(cos_sin_a3) + p3;

    return pred;
}

cv::Mat ChessBoardsDetector::growChessboard(cv::Mat &board_index, int border_type)
{
    // return immediately, if there do not exist any chessboards
    cv::Mat board_index_grow;
    if(board_index.empty())
    {
        return board_index_grow;
    }

    // extract feature locations
    cv::Mat p = corners.p.clone();

    // list of unused feature elements
    std::unordered_set<int> unused;
    for(int i = 0; i != corners.p.rows; ++i)
    {
        unused.insert(i);
    }

    for(int i = 0; i != board_index.rows; ++i)
    {
        auto *ptr_board_index = board_index.ptr<int>(i);
        for(int j = 0; j != board_index.cols; ++j)
        {
            if(ptr_board_index[j] != -1)
            {
                unused.erase(ptr_board_index[j]);
            }
        }
    }

    std::vector<int> unused_vec;
    std::copy(unused.begin(), unused.end(),
        std::back_inserter<std::vector<int>>(unused_vec));
    std::sort(unused_vec.begin(), unused_vec.end());

    // candidates from unused corners
    cv::Mat cand = cv::Mat(unused_vec.size(), 2, CV_64FC1);
    int index = 0;
    for(auto iter = unused_vec.begin(); iter != unused_vec.end(); ++iter)
    {
        auto *ptr_p_iter = p.ptr<double>(*iter);
        auto *ptr_cand = cand.ptr<double>(index);
        ptr_cand[0] = ptr_p_iter[0];
        ptr_cand[1] = ptr_p_iter[1];
        ++index;
    }

    cv::Mat index_p1, index_p2, index_p3, p1, p2, p3, pred, idx, org_idx;
    switch(border_type)
    {
        case 0:
            index_p1 = board_index.col(board_index.cols - 3);
            index_p2 = board_index.col(board_index.cols - 2);
            index_p3 = board_index.col(board_index.cols - 1);
            p1 = extractPoints(p, index_p1);
            p2 = extractPoints(p, index_p2);
            p3 = extractPoints(p, index_p3);
            pred = predictCorners(p1, p2, p3);
            idx = assignClosestCorners(cand, pred);
            org_idx = extractIndex(unused_vec, idx);
            cv::hconcat(board_index, org_idx.t(), board_index_grow);
            break;
        case 1:
            index_p1 = board_index.row(board_index.rows - 3);
            index_p2 = board_index.row(board_index.rows - 2);
            index_p3 = board_index.row(board_index.rows - 1);
            p1 = extractPoints(p, index_p1);
            p2 = extractPoints(p, index_p2);
            p3 = extractPoints(p, index_p3);
            pred = predictCorners(p1, p2, p3);
            idx = assignClosestCorners(cand, pred);
            org_idx = extractIndex(unused_vec, idx);
            cv::vconcat(board_index, org_idx, board_index_grow);
            break;
        case 2:
            index_p1 = board_index.col(2);
            index_p2 = board_index.col(1);
            index_p3 = board_index.col(0);
            p1 = extractPoints(p, index_p1);
            p2 = extractPoints(p, index_p2);
            p3 = extractPoints(p, index_p3);
            pred = predictCorners(p1, p2, p3);
            idx = assignClosestCorners(cand, pred);
            org_idx = extractIndex(unused_vec, idx);
            cv::hconcat(org_idx.t(), board_index, board_index_grow);
            break;
        case 3:
            index_p1 = board_index.row(2);
            index_p2 = board_index.row(1);
            index_p3 = board_index.row(0);
            p1 = extractPoints(p, index_p1);
            p2 = extractPoints(p, index_p2);
            p3 = extractPoints(p, index_p3);
            pred = predictCorners(p1, p2, p3);
            idx = assignClosestCorners(cand, pred);
            org_idx = extractIndex(unused_vec, idx);
            cv::vconcat(org_idx, board_index, board_index_grow);
            break;
    }

    return board_index_grow;
}

cv::Mat ChessBoardsDetector::assignClosestCorners(cv::Mat& cand, cv::Mat& pred)
{
    // return error if not enough candidates are available
    cv::Mat result;
    if (cand.rows < pred.rows)
    {
        return result;
    }

    // build distance matrix
    cv::Mat D = cv::Mat::zeros(cand.rows, pred.rows, CV_64FC1);
    for (int i = 0; i != pred.rows; ++i)
    {
        cv::Mat delta = cand - 
            cv::Mat::ones(cand.rows, 1, CV_64FC1) * pred.row(i);
        cv::Mat tmp1, tmp2;
        cv::pow(delta.col(0), 2, tmp1);
        cv::pow(delta.col(1), 2, tmp2);
        cv::sqrt(tmp1 + tmp2, tmp1);
        tmp1.copyTo(D.col(i));
    }

    // search greedily for closest corners
    result = cv::Mat::ones(1, D.cols, CV_32SC1) * -1;
    auto *ptr_res = result.ptr<int>();
    for (int i = 0; i != pred.rows; ++i)
    {
        cv::Point min_loc;
        cv::minMaxLoc(D, nullptr, nullptr, &min_loc);
        auto *ptr_D_min_row = D.ptr<double>(min_loc.y);
        for (int ii = 0; ii != D.cols; ++ii)
        {
            ptr_D_min_row[ii] = std::numeric_limits<double>::max();
        }

        for (int ii = 0; ii != D.rows; ++ii)
        {
            auto *ptr_D_ii_row = D.ptr<double>(ii);
            ptr_D_ii_row[min_loc.x] = std::numeric_limits<double>::max();
        }

        ptr_res[min_loc.x] = min_loc.y;
    }

    return result;
}

cv::Mat ChessBoardsDetector::MatAtan2(cv::Mat& x, cv::Mat& y)
{
    CV_Assert(x.size() == y.size());

    cv::Mat angle = cv::Mat(x.size(), CV_64FC1);
    int element_num = x.rows * x.cols;
    auto *ptr_x = x.ptr<double>();
    auto *ptr_y = y.ptr<double>();
    auto *ptr_angle = angle.ptr<double>();
    for (int i = 0; i != element_num; ++i)
    {
        ptr_angle[i] = std::atan2(ptr_x[i], ptr_y[i]);

        // correct angle to lie in between[0, pi]
        if(ptr_angle[i] < 0) ptr_angle[i] += CV_PI;
        if(ptr_angle[i] > CV_PI) ptr_angle[i] -= CV_PI;
    }

    return angle;
}

void ChessBoardsDetector::directionalNeighbor(int idx, cv::Mat v, cv::Mat& board_index, int* neigh, double *dist)
{
    // list of neighboring elements, which are currently not in use
    std::unordered_set<int> unused;
    for(int i = 0; i != corners.p.rows; ++i)
    {
        unused.insert(i);
    }
    
    for(int i = 0; i != board_index.rows; ++i)
    {
        auto *ptr_board_index = board_index.ptr<int>(i);
        for(int j = 0; j != board_index.cols; ++j)
        {
            if(ptr_board_index[j] != -1)
            {
                unused.erase(ptr_board_index[j]);
            }
        }
    }

    std::vector<int> unused_vec;
    std::copy(unused.begin(), unused.end(), 
        std::back_inserter<std::vector<int>>(unused_vec));
    std::sort(unused_vec.begin(), unused_vec.end());

    // direction and distance to unused corners
    auto *ptr_v = v.ptr<double>();
    cv::Mat dir = cv::Mat::zeros(unused_vec.size(), 2, CV_64FC1);
    auto *ptr_pt_idx = corners.p.ptr<double>(idx);
    for(int i = 0; i != unused_vec.size(); ++i)
    {
        auto *ptr_dir = dir.ptr<double>(i);
        auto *ptr_pt_iter = corners.p.ptr<double>(unused_vec[i]);
        ptr_dir[0] = ptr_pt_iter[0] - ptr_pt_idx[0];
        ptr_dir[1] = ptr_pt_iter[1] - ptr_pt_idx[1];
    }
    cv::Mat distance = dir.col(0) * ptr_v[0] + dir.col(1) * ptr_v[1];

    // distances
    cv::Mat dist_edge = dir - distance * v;
    dist_edge = dist_edge.col(0).mul(dist_edge.col(0)) +
            dist_edge.col(1).mul(dist_edge.col(1));
    cv::sqrt(dist_edge, dist_edge);

    cv::Mat dist_point = distance.clone();
    auto *ptr_dist_point = dist_point.ptr<double>();
    for(int i = 0; i != dist_point.rows; ++i)
    {
        if(ptr_dist_point[i] < 0.0)
        {
            ptr_dist_point[i] = std::numeric_limits<double>::max();
        }
    }

    // find best neighbor
    double min_dist = std::numeric_limits<double>::max();
    auto *ptr_dist_edge = dist_edge.ptr<double>();
    int min_idx = 0;
    for(int i = 0; i != unused_vec.size(); ++i)
    {
        double tmp = ptr_dist_point[i] + 5.0f * ptr_dist_edge[i];
        if(tmp < min_dist)
        {
            min_dist = tmp;
            min_idx = i;
        }
    }

    *neigh = unused_vec[min_idx];
    *dist = min_dist;
}