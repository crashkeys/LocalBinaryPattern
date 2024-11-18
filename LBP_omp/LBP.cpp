//
// Created by nenec on 21/10/2024.
//
#include <opencv2/opencv.hpp>
#include "LBP.h"

#include <omp.h>

using namespace cv;
using namespace std;

bitset<8> createCode(Mat& frame, int i, int j) {
    int neighbors[8];
    neighbors[0] = frame.at<uchar>(i - 1, j - 1);
    neighbors[1] = frame.at<uchar>(i - 1, j);
    neighbors[2] = frame.at<uchar>(i - 1, j + 1);
    neighbors[3] = frame.at<uchar>(i, j + 1);
    neighbors[4] = frame.at<uchar>(i + 1, j + 1);
    neighbors[5] = frame.at<uchar>(i + 1, j);
    neighbors[6] = frame.at<uchar>(i + 1, j - 1);
    neighbors[7] = frame.at<uchar>(i, j - 1);

    int threshold = frame.at<uchar>(i, j); //central pixel
    std::bitset<8> code;
    for (int ii = 0; ii < 8; ii++) {
        if (neighbors[ii] > threshold)
            code[ii] = true;
        else
            code[ii] = false;
    }
    return code;
}

Mat drawHist(Mat &frame, int numThreads) {
    int hist[256] = {}; //init

    //prepare black image out of parallel section
    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound( static_cast<double>(hist_w)/256);
    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0,0,0));

    omp_set_num_threads(numThreads);

// OMP 4.5
#pragma omp parallel reduction(+:hist) default(none) shared(frame)
    {
#pragma omp for
        for (int i = 1; i < frame.rows-1; i++) {
            for (int j = 1; j < frame.cols-1; j++) {
                bitset<8> code = createCode(frame, i, j);
                hist[code.to_ulong()]++;
            }
        }
    }


/*
// OMP 2.0
#pragma omp parallel shared(hist, frame) default(none)
    {
        int priv_hist[256] = {};

#pragma omp for
        for (int i = 1; i < frame.rows-1; i++) {
            for (int j = 1; j < frame.cols-1; j++) {
                bitset<8> code = createCode(frame, i, j);
                priv_hist[code.to_ulong()]++;
            }
        }

#pragma omp critical
        for (int i = 0; i < 256; i++) {
            hist[i] += priv_hist[i];
        }
    }
*/

    for (int i = 0; i < 256; i++) {
        cv::rectangle(histImage,
                        Point(i*bin_w, hist_h),
                        Point((i+1)*bin_w, hist_h - hist[i]),
                        Scalar(199,224,157), //bgr
                        FILLED);
    }

    return histImage;


}


