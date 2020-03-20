﻿#include<opencv2/opencv.hpp>
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
	cv::Mat srcMat = imread("E:\\IMG.jpg", 0);
	cv::Mat labelMat;
	cv::Mat statsMat;
	cv::Mat centrMat;
	cv::Mat resultMat;
	//二值化
	bitwise_not(srcMat, srcMat);//注意前景为白色
	cv::threshold(srcMat, binaryMat, 0, 255, THRESH_OTSU);
	Mat kernel= getStructuringElement(MORPH_RECT, Size(11, 11));
	morphologyEx(binaryMat, dstImage_Open, 2, kernel,Point(-1,-1),1);//开运算

	int nComp = cv::connectedComponentsWithStats(dstImage_Open,
		labelMat,
		statsMat,
		centrMat,
		8,
		CV_32S);

	//输出连通域信息
	for (int i = 0; i < nComp; i++)
	{
		//各个连通域的统计信息保存在stasMat中
		cout << "connected Components NO. " << i << endl;
		cout << "pixels = " << statsMat.at<int>(i, 4) << endl;
		cout << "width = " << statsMat.at<int>(i, 2) << endl;
		cout << "height = " << statsMat.at<int>(i, 3) << endl;
		cout << endl;
	}
	
	cout << "the total of connected Components = " << nComp - 1 << endl;//-1,nComp包括背景
	resultMat = cv::Mat::zeros(srcMat.size(), CV_8UC3);	//显示用图像
	std::vector<cv::Vec3b> colors(nComp);
	colors[0] = cv::Vec3b(0, 0, 0);//背景黑色
	//随机产生颜色
	for (int n = 1; n < nComp; n++)
	{
		colors[n] = cv::Vec3b(rand() / 255, rand() / 255, rand() / 255);
	}


	//绘制bounding box
	for (int i = 1; i < nComp; i++)
	{
		Rect bndbox;
		//左上角坐标
		bndbox.x = statsMat.at<int>(i, 0);
		bndbox.y = statsMat.at<int>(i, 1);
		//宽和长 
		bndbox.width = statsMat.at<int>(i, 2);
		bndbox.height = statsMat.at<int>(i, 3);
		//绘制
		rectangle(resultMat, bndbox, CV_RGB(255, 255, 255), 1, 8, 0);
		}


	imshow("binaryMat", binaryMat);
	imshow("results", resultMat);
	imshow("frame", srcMat);
	moveWindow("frame", 0, 20);
	moveWindow("binaryMat", srcMat.cols, 20);
	moveWindow("results", srcMat.cols * 2, 20);
	waitKey(0);
	
	return 0;
}
