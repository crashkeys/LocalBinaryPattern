#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/core/utils/logger.hpp>
#include <fstream>
#include <iomanip>
#include <chrono>
#include "LBP.h"

using namespace cv;
using namespace std;

/*SEQUENTIAL VERSION */

int main(int argc, char** argv) {
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);

    int REPETITIONS = 30;
    ofstream csvFile("C:/Users/Irene/Desktop/graphs/OMP_4.5/test.csv");
    csvFile << "ImgSize,Time(microsec)" << endl;

    String filename = "scacchi.jpg";
    Mat frame = imread("./images/" + filename, IMREAD_GRAYSCALE);

    printf("Channels: %d \n", frame.channels());
    printf("Size: %d x %d \n", frame.cols, frame.rows);

    //We need to create a padding for the pixels at the borders
    copyMakeBorder(frame, frame, 1, 1, 1, 1, BORDER_CONSTANT, 0); //black


    auto start = std::chrono::high_resolution_clock::now();
    Mat histImg = drawHist(frame);
    auto end = std::chrono::high_resolution_clock::now();
    auto ms_int = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf(" - Time seq.: %lld \n", ms_int.count());

    csvFile << std::fixed << std::setprecision(6) << frame.cols << "," << ms_int.count() << endl;


    for (int i = 1; i < REPETITIONS+1; i++) {
        int pdd = 30*i; //padding
        copyMakeBorder(frame, frame, pdd, pdd, pdd, pdd, BORDER_CONSTANT, std::rand()%(255));

        auto start = std::chrono::high_resolution_clock::now();
        Mat histImg = drawHist(frame);
        auto end = std::chrono::high_resolution_clock::now();
        auto ms_int = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        printf("Size: %d x %d at repetition no. %d", frame.cols, frame.rows, i);
        printf(" - Time seq.: %lld \n", ms_int.count());

        csvFile << std::fixed << std::setprecision(6) << frame.cols << "," << ms_int.count() << endl;
    }

    csvFile.close();


    //show histogram
    copyMakeBorder(histImg, histImg, 25, 0, 10, 10, BORDER_CONSTANT,0); //for better visualisation
    namedWindow("_hist", WINDOW_AUTOSIZE);
    imshow("_hist", histImg);


    namedWindow(filename, WINDOW_NORMAL);
    imshow(filename, frame);
    waitKey();

    return 0;
}


