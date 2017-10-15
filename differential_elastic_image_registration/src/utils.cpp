#include <utils.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

//Now I think this function is good
void diffxyt(cv::Mat& frame1, cv::Mat& frame2, std::vector<cv::Mat>& fxyt)
{
    cv::Mat px = (cv::Mat_<float>(1, 3) << 0.223755, 0.552490, 0.223755 ), 
        dx = (cv::Mat_<float>(1, 3) << 0.453014, 0.0, -0.453014),
        py = px, dy = dx;

    cv::Mat frame_pz = 0.5*frame1 + 0.5*frame2, fdx, fdy, fdt;
    cv::sepFilter2D(frame_pz, fdx, CV_32F, dx, py, cv::Point(-1, -1), 0, cv::BORDER_REFLECT_101);
    cv::sepFilter2D(frame_pz, fdy, CV_32F, px, -dy, cv::Point(-1, -1), 0, cv::BORDER_REFLECT_101);

    cv::Mat frame_dz = frame1 - frame2;
    cv::sepFilter2D(frame_dz, fdt, CV_32F, px, py, cv::Point(-1, -1), 0, cv::BORDER_REFLECT_101);

    fxyt.clear(); fxyt.push_back(fdx); fxyt.push_back(fdy); fxyt.push_back(fdt);
}

//This is also good
void reduce(cv::Mat& img, cv::Mat& output)
{
    //assume img is CV_32F
    int h = std::ceil(img.rows / 2), w = std::ceil(img.cols / 2);
    output = cv::Mat::zeros(h, w, CV_32F);
    cv::Mat kernel = (cv::Mat_<float>(1, 5) << 0.05, 0.25, 0.4, 0.25, 0.05);
    cv::Mat img_conv;
    cv::sepFilter2D(img, img_conv, CV_32F, kernel, kernel, cv::Point(-1, -1), 0, cv::BORDER_REFLECT_101);
    //downsample
    for(int i = 0; i != output.rows; ++i)
    {
        float* const ptr = output.ptr<float>(i);
        float* const ptr_img = img_conv.ptr<float>(2*i);
        for(int j = 0; j != output.cols; ++j)
        {
            ptr[j] = ptr_img[2*j];
        }
    }
}

void interp2(cv::Mat& X, cv::Mat& Y, cv::Mat& V, cv::Mat& XI, cv::Mat& YI, cv::Mat& Z)
{
    Z = cv::Mat::zeros(V.size(), CV_32F);
    cv::Mat Vext = cv::Mat::zeros(V.rows + 2, V.cols + 2, CV_32F);
    V.copyTo(Vext.rowRange(1, Vext.rows - 1).colRange(1, Vext.cols - 1));

    float xb = X.at<float>(0, 0), yb = Y.at<float>(0, 0), 
        dx = X.at<float>(0, 1) - X.at<float>(0, 0),
        dy = Y.at<float>(1, 0) - Y.at<float>(0, 0);


    for(int i = 0; i != V.rows; ++i)
    {
        for(int j = 0; j != V.cols; ++j)
        {
            float xaxis = XI.at<float>(i, j), yaxis = YI.at<float>(i, j);

            float float_nx = (xaxis - xb)/dx,
                float_ny = (yaxis - yb)/dy;

            if(float_nx < 0 || float_nx >= V.cols || float_ny < 0 || float_ny >= V.rows)
                continue;

            int pnx = std::floor(float_nx) + 1,
                nnx = std::floor(float_nx) + 2,
                pny = std::floor(float_ny) + 1,
                nny = std::floor(float_ny) + 2;
            float s = xaxis - X.at<float>(0, pnx - 1);
            float v1 = s/dx*(Vext.at<float>(pny, nnx) - Vext.at<float>(pny, pnx)) + Vext.at<float>(pny, pnx),
                v2 = s/dx*(Vext.at<float>(nny, nnx) - Vext.at<float>(nny, pnx)) + Vext.at<float>(nny, pnx);
            s = yaxis - Y.at<float>(pny - 1, 0);
            float v = s/dy*(v2 - v1) + v1;
            Z.at<float>(i, j) = v;
        }
    }
}

cv::Mat affine_warp(cv::Mat& img, cv::Mat& M)
{
    cv::Mat mx = cv::Mat::zeros(img.size(), CV_32F), my = mx.clone();
    int h = img.rows, w = img.cols;

    for(int i = 0; i != h; ++i)
    {
        float* ptr_mx = mx.ptr<float>(i);
        for(int j = 0; j != w; ++j)
        {
            ptr_mx[j] = j - w/2.0f + 0.5;
            
        }
    }
    
    for(int j = 0; j != w; ++j)
    {
        for(int i = 0; i != h; ++i)
        {
            float* ptr_my = my.ptr<float>(i);
            ptr_my[j] = -(i - h/2.0f + 0.5);
        }
    }

    cv::Mat mxt, myt, mxv, myv;
    cv::transpose(mx, mxt); cv::transpose(my, myt);
    mxv = mxt.reshape(0, mxt.rows*mxt.cols);
    myv = myt.reshape(0, myt.rows*myt.cols);

    mxv = mxv.t(); myv = myv.t();
    cv::Mat mm = cv::Mat::zeros(2, mxv.cols, CV_32F);
    mxv.copyTo(mm.row(0)); myv.copyTo(mm.row(1));

    float dx = M.at<float>(0, 2), dy = M.at<float>(1, 2);
    cv::Mat pnts = M.rowRange(0, 2).colRange(0, 2)*mm;
    cv::Mat mx2 = pnts.row(0) + dx, my2 = pnts.row(1) + dy;

    mx2 = mx2.reshape(0, img.cols).clone().t();
    my2 = my2.reshape(0, img.cols).clone().t();

    cv::Mat r;
    interp2(mx, my, img, mx2, my2, r);
    return r;
}

//This is good
void affine_find(affine_params& params, TempStatic& T)
{
    
    //static cv::Mat mx, my, H, mout_def, minus_one;
    
    if(T.mx.empty())
    {
        //init parameters
        int h = params.h, w = params.w;
        
        T.mx = cv::Mat::zeros(h, w, CV_32F); T.my = T.mx.clone();
        for(int i = 0; i != h; ++i)
        {
            float* ptr_mx = T.mx.ptr<float>(i);
            for(int j = 0; j != w; ++j)
            {
                ptr_mx[j] = j - w/2.0f + 0.5;
                
            }
        }
        
        for(int j = 0; j != w; ++j)
        {
            for(int i = 0; i != h; ++i)
            {
                float* ptr_my = T.my.ptr<float>(i);
                ptr_my[j] = -(i - h/2.0f + 0.5);
            }
        }
        
        T.minus_one = cv::Mat::ones(1, h*w, CV_32F)*(-1.0f);
        
        cv::Mat mxt, myt, Ht;
        cv::transpose(T.mx, mxt); cv::transpose(T.my, myt);
        T.mx = mxt.reshape(0, mxt.rows*mxt.cols);
        T.my = myt.reshape(0, myt.rows*myt.cols);
        T.mx = T.mx.t(); T.my = T.my.t();
        T.H = cv::Mat::ones(w, h, CV_32F);
        T.H.row(0) = 0; T.H.row(w - 1) = 0;
        T.H.col(0) = 0; T.H.col(h - 1) = 0;
        cv::transpose(T.H, Ht);
        T.H = Ht.reshape(0, Ht.rows*Ht.cols).clone().t();
        T.mout_def = (cv::Mat_<float>(1, 8) << 1, 0, 0, 1, 0, 0, 1, 0);
        //params.mout = (cv::Mat_<float>(1, 8) << 1, 0, 0, 1, 0, 0, 1, 0, 0);
    }
    
    params.fx = params.fx.mul(T.H);
    params.fy = params.fy.mul(T.H);
    params.ft = params.ft.mul(T.H);
    params.nf = params.nf.mul(T.H);
    T.minus_one = T.minus_one.mul(T.H);
    
    cv::Mat p1 = params.fx.mul(T.mx), p2 = params.fx.mul(T.my), 
        p3 = params.fy.mul(T.mx), p4 = params.fy.mul(T.my);
    
    params.pt = cv::Mat::zeros(8, p1.cols, CV_32F);
    p1.copyTo(params.pt.row(0)); p2.copyTo(params.pt.row(1));
    p3.copyTo(params.pt.row(2)); p4.copyTo(params.pt.row(3));
    params.fx.copyTo(params.pt.row(4));
    params.fy.copyTo(params.pt.row(5));
    params.nf.copyTo(params.pt.row(6));
    T.minus_one.copyTo(params.pt.row(7));
    
    params.kt = params.ft + params.nf + p1 + p4;
    
    cv::Mat mask; cv::repeat(params.mask, 8, 1, mask);
    
    cv::Mat P = params.pt.mul(mask) * params.pt.t(), 
        K = params.pt.mul(mask) * params.kt.t(), P_inv, m;

    float r;    
    double flag = cv::invert(P, P_inv);
    if(flag != 0.0)
    {
        m = P_inv * K;
        r = 1.0f;
    }
    else
    {
        m = T.mout_def.t();
        r = 0.0f;
    }

    params.mout = cv::Mat(1, 9, CV_32F); cv::transpose(m, m);
    m.copyTo(params.mout.row(0).colRange(0, 8));
    params.mout.at<float>(0, 8) = r;
}

//This is also good
RParams affine_find_api(cv::Mat& img1, cv::Mat& img2, cv::Mat& mask, TempStatic& T)
{
    std::vector<cv::Mat> fxyt;
    diffxyt(img1, img2, fxyt);
    
    cv::Mat lambda = cv::Mat::ones(8, 1, CV_32F)*10e11, 
        deviants = (cv::Mat_<float>(8, 1) << 0, 0, 0, 0, 0, 0, 1, 0);
        
    //std::cout << img1 << std::endl;
    //std::cout << img2 << std::endl;
    affine_params para;
        
    cv::Mat fxt, fyt, ftt, nimgt;
    cv::transpose(fxyt[0], fxt); cv::transpose(fxyt[1], fyt);
    cv::transpose(fxyt[2], ftt); cv::transpose(-img1, nimgt);
    para.fx = fxt.reshape(0, fxt.rows*fxt.cols).clone().t();
    para.fy = fyt.reshape(0, fyt.rows*fyt.cols).clone().t();
    para.ft = ftt.reshape(0, ftt.rows*ftt.cols).clone().t();
    para.nf = nimgt.reshape(0, nimgt.rows*nimgt.cols).clone().t();
    para.h = img1.rows; para.w = img1.cols;
    cv::Mat maskt; cv::transpose(mask, maskt);
    maskt = maskt.reshape(0, maskt.rows*maskt.cols);
    para.mask = maskt.clone().t();
    
    para.S = cv::Mat::diag(lambda);
    para.D = lambda.mul(deviants);
    
    affine_find(para, T);
    
    cv::Mat M = (cv::Mat_<float>(3, 3) << para.mout.at<float>(0, 0), para.mout.at<float>(0, 1), para.mout.at<float>(0, 4),
                                          para.mout.at<float>(0, 2), para.mout.at<float>(0, 3), para.mout.at<float>(0, 5),
                                          0,                    0,                    1);
    
    float c = para.mout.at<float>(0, 6), b = para.mout.at<float>(0, 7),
        r = para.mout.at<float>(0, 8);
        
    RParams R;
    R.M = M; R.pt = para.pt; R.kt = para.kt;
    R.c = c; R.b = b; R.r = r;

    return R;
}

cv::Mat mask_compute(cv::Mat& img1, cv::Mat& img2, float sigma)
{
    cv::Mat r1 = img1 - img2, r1_sq = r1.mul(r1);
    cv::Mat kernel = cv::Mat::ones(1, 3, CV_32F)/3.0f;
    cv::sepFilter2D(r1_sq, r1_sq, CV_32F, kernel, kernel, cv::Point(-1, -1), 0, cv::BORDER_REFLECT_101);

    float r2_sq = sigma / 10.0f;
    
    cv::Mat er1 = -r1_sq.mul(1.0f/sigma);
    for(int i = 0; i != er1.rows; ++i)
    {
        float* const ptr = er1.ptr<float>(i);
        for(int j = 0; j != er1.cols; ++j)
        {
            ptr[j] = std::exp(ptr[j]);
        }
    }
    
    float er2 = std::exp(-r2_sq/sigma);
    
    cv::Mat W = er1.mul(1.0f/(er2 + er1));
    
    double max_v;
    cv::minMaxIdx(W, nullptr, &max_v, nullptr, nullptr);
    W = W/max_v;
    
    return W;
}

void affine_iter(cv::Mat& img1, cv::Mat& img2, int iters, cv::Mat& M, float& b, float& c)
{
    int h = img1.rows, w = img1.cols;
    cv::Mat W = cv::Mat::ones(h, w, CV_32F);
    
    c = 1.0f, b = 0.0f;
    M = cv::Mat::eye(3, 3, CV_32F);
    
    cv::Mat imgN1 = img1.clone();
    
    TempStatic T;
    for(int i = 0; i != iters; ++i)
    {
        RParams R = affine_find_api(imgN1, img2, W, T);
        
        M = M*R.M;
        b = b + R.b;
        c = c*R.c;
        
        if(c < 0.1) c = 1;
        
        //update the img1
        //cv::warpAffine(img1, imgN1, M.rowRange(0, 2), img1.size(), cv::INTER_CUBIC);
        imgN1 = affine_warp(img1, M);
        imgN1 = imgN1*c + b;
        
        W = mask_compute(imgN1, img2, 0.01);
        
        //cv::Mat show = imgN1.clone(); double max_v, min_v;
        //cv::minMaxIdx(imgN1, &min_v, &max_v);
        //show = (show - min_v)/(max_v - min_v);
        //cv::imshow("imgN", show);
        //cv::imshow("img2", img2);
        //cv::waitKey();
    }
}

void get_affine_params(cv::Mat& img1, cv::Mat& img2, int iters, cv::Mat& M, float& bnew, float& cnew)
{
    cv::Mat img1c, img2c;
    img1.convertTo(img1c, CV_32F, 1.0f/255);
    img2.convertTo(img2c, CV_32F, 1.0f/255);
    
    int steps = std::log2(img1c.cols/32) + 1;
    
    std::vector<cv::Mat> pyr1, pyr2;
    pyr1.push_back(img1c);
    pyr2.push_back(img2c);
    
    int scale = 1;
    for(int i = 1; i != steps; ++i)
    {
        cv::Mat pyr1next, pyr2next;
        reduce(pyr1[i - 1], pyr1next); pyr1.push_back(pyr1next);
        reduce(pyr2[i - 1], pyr2next); pyr2.push_back(pyr2next);
        scale *= 2;
    }

    M = cv::Mat::eye(3, 3, CV_32F);
    bnew = 0, cnew = 1.0f;
    for(int i = steps; i != 0; --i)
    {
        cv::Mat imgS1 = pyr1[i - 1].clone(), imgS2 = pyr2[i - 1].clone();
        
        if(i != steps)
        {
            cv::Mat M1 = M.clone();
            M1.at<float>(0, 2) /= scale; M1.at<float>(1, 2) /= scale;
            imgS1 = affine_warp(imgS1, M1);
            //cv::warpAffine(imgS1, imgS1, M1.rowRange(0, 2), imgS1.size(), cv::INTER_CUBIC);
        }
        
        cv::Mat Mnew; float b, c;
        affine_iter(imgS1, imgS2, 10, Mnew, b, c);
        cnew = c * cnew; bnew = b + bnew;
        
        Mnew.at<float>(0, 2) *= scale; Mnew.at<float>(1, 2) *= scale;
        M = Mnew * M;
        
        scale /= 2;
    }
}

cv::Mat otsu(cv::Mat& img)
{

    //histgram
    float hist[256] = { 0.0f };
    int counter = 0;
    for(int i = 0; i != img.rows; ++i)
    {
        const uchar* const ptr = img.ptr<uchar>(i);
        //const uchar* const ptr_mask = mask.ptr<uchar>(i);
        for(int j = 0; j != img.cols; ++j)
        {
            if(ptr[j] == 0) continue;
            hist[ptr[j]] += 1;
            counter += 1;
        }
    }

    float u = 0.0f;
    for(int i = 0; i != 256; ++i)
    {
        hist[i] /= counter;
        u += hist[i]*i;
    }


    int thres = 0; float max_dev = 0.0f;
    for(int i = 0; i != 256; ++i)
    {
        float u_d = 0.0f, u_u = 0.0f, w_d = 0.0f, w_u = 0.0f;
        for(int j = 0; j != i; ++j)
        {
            u_d += hist[j]*j;
            w_d += hist[j];
        }

        for(int j = i; j != 256; ++j)
        {
            u_u += hist[j]*j;
            w_u += hist[j];
        }

        float dev = w_d*(u_d - u)*(u_d - u) + w_u*(u_u - u)*(u_u - u);
        
        if(dev > max_dev)
        {
            max_dev = dev;
            thres = i;
        }
    }

    thres = cv::saturate_cast<uchar>(1.3*thres);
    cv::Mat r = cv::Mat::zeros(img.size(), CV_8U);
    for(int i = 0; i != r.rows; ++i)
    {
        uchar* const ptr_r = r.ptr<uchar>(i);
        uchar* const ptr = img.ptr<uchar>(i);
        //const uchar* const ptr_mask = mask.ptr<uchar>(i);
        for(int j = 0; j != r.cols; ++j)
        {
            if(ptr[j] == 0) continue;

            if(ptr[j] > thres) ptr_r[j] = 255;
        }
    }

    return r;
}
