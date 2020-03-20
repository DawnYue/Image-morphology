#include<opencv2/opencv.hpp>
#include<iostream>
#include<math.h>

#include <list>
#include <vector>
#include <map>
#include <stack>
using namespace cv;
using namespace std;
int main()
{   
	Mat  dstImage_Erode, dstImage_Dilate, dstImage_Open, dstImage_Close;
	cv::Mat binaryMat;
	cv::Mat srcMat = imread("E:\\coin.png", 0);

	//二值化
	cv::threshold(srcMat, binaryMat, 0, 255, THRESH_OTSU);
	Mat kernel= getStructuringElement(MORPH_RECT, Size(3, 3));
	erode(binaryMat, dstImage_Erode, kernel);
	dilate(binaryMat, dstImage_Dilate, kernel);
	morphologyEx(binaryMat, dstImage_Open,MORPH_OPEN, kernel);
	morphologyEx(binaryMat, dstImage_Close, MORPH_CLOSE, kernel);
	imshow("binaryMat", binaryMat);
	imshow("Image_Erode", dstImage_Erode);
	imshow("Image_Dilate", dstImage_Dilate);
	imshow("Image_Open", dstImage_Open);
	imshow("Image_Close", dstImage_Close);
	waitKey(0);
	return 0;
}
