#ifndef COMMON_HPP__
#define COMMON_HPP__

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace std;
using namespace cv;

void printMatrixInfo(const Mat &source);

float getValueOfMatrix(const Mat &source, int y, int x);
void setValueOfMatrix(Mat &source, int y, int x, float value);
Mat addMatrix(const Mat& a, const Mat& b);
string getImageNameFromImageDir(string imageDir);
int getMatrixArea(const Mat &source);
#endif //COMMON_HPP__