#ifndef IMAGE_OPERATOR_HPP__
#define IMAGE_OPERATOR_HPP__

#include "common.hpp"
#include "image.hpp"
#include "kernel_generator.hpp"
#include "image_operator_opencv.hpp"

class ImageOperator {
	// PUBLIC FUNCTIONS
public:

	// Sobel edge detection
    static Mat EdgeDetectSobel(const Mat& sourceImage, int gaussSize=5, float gaussStd=1.0, int edge_thres=100, bool isShow=true);

	// Prewitt edge detection 
    static Mat EdgeDetectPrewitt(const Mat& sourceImage, int gaussSize=5, float gaussStd=1.0, int edge_thres=100, bool isShow=true);

	// Laplacian edge detection
	static Mat EdgeDetectLaplacian(const Mat& sourceImage, int gaussSize=5, float gaussStd=1.0, float max_thres=0.2, bool isShow=true);

	// Canny edge detection
	static Mat EdgeDetectCanny(const Mat& sourceIamge, int gaussSize=5, float gaussStd=1.0, int low_thres=10, int high_thres=50, bool isShow=true);

	// CONVOLUTION 2D Version 2
	static Mat conv2d(const Mat& source, const Mat& kernel, bool acceptNegative = false, bool acceptExceed = false);

	// measure Difference between 2 Mat
	static int measureDifference(const Mat &result, const Mat &ground_truth);

	// calculate false positive edge points
	static int calculateFalsePositivePoints(const Mat& result, const Mat& groundTruth);
	
	// calculate false negative edge points
	static int calculateFalseNegativePoints(const Mat& result, const Mat& groundTruth);
	
	// calculate true positive edge points
	static int calculateTruePositivePoints(const Mat& result, const Mat& groundTruth);
	
	// calculate true negative edge points
	static int calculateTrueNegativePoints(const Mat& result, const Mat& groundTruth);

	// PRIVATE FUNCTION:
private:
	// ---------------------------------------------------------------------------------------------------
	// Conv2D helper functions
	static float applyConvolutionAtPosition(const Mat& source, int x, int y, const Mat& kernel);

	// ---------------------------------------------------------------------------------------------------
	// Laplacian helper functions
	//static int getLaplacianThreshold();
	static int getMaxValue(const Mat& source);

	static Mat findZeroCrossingPoints(const Mat& source, float slopeThres);
	static void checkNonZeroBetween(const Mat& source, Mat& result, int y, int x, float slopeThres);
	static void checkZeroBetween(const Mat& source, Mat& result, int y, int x, float slopeThres);
	static bool checkEdgePointCondition(float point1, float point2, float slopeThres);

	// ---------------------------------------------------------------------------------------------------
	// Canny helper functions
	static Mat computeMagnitude(const Mat& a, const Mat& b);
	static Mat computeDirection(const Mat& gx, const Mat &gy);
	static void NonMaxSuppression(const Mat &direction, Mat &gradient);
	static void dfs(Mat &canny_mask, const Mat &gradient, int y, int x, float low_threshold, vector<vector<bool>> &visited);
	static Mat HysteresisThresholding(const Mat &gradient, float high_threshold, float low_threshold);


    // ---------------------------------------------------------------------------------------------------
	// Refinement helper functions
    static void maximizeEdgePixels(Mat& source, int thres);
};

#endif //IMAGE_OPERATOR_HPP__