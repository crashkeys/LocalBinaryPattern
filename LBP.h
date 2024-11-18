//
// Created by nenec on 21/10/2024.
//

#ifndef LBP_H
#define LBP_H


#include <opencv2/opencv.hpp>
#include <bitset>

using namespace cv;
using namespace std;

bitset<8> createCode(Mat& frame, int i, int j);

Mat drawHist(Mat &frame, int numThreads);

#endif //LBP_H
