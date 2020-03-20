//课前准备
#include<opencv2/opencv.hpp>
#include<iostream>
#include<math.h>
#include <string>
#include <list>
#include <vector>
#include <map>
#include <stack>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace cv;
using namespace std;

IplImage *src = 0;
IplImage *dst = 0;
IplConvKernel *element = 0;//声明一个结构元素  
int element_shape = CV_SHAPE_RECT;//长方形形状的元素  
int max_iters = 10;
int open_close_pos = 0;
int erode_dilate_pos = 0;

void Seed_Filling(const cv::Mat& binImg, cv::Mat& lableImg)   //种子填充法https://blog.csdn.net/cooelf/article/details/26581539
{
	// 4邻接方法


	if (binImg.empty() ||
		binImg.type() != CV_8UC1)
	{
		return;
	}

	lableImg.release();
	binImg.convertTo(lableImg, CV_32SC1);

	int label = 1;

	int rows = binImg.rows - 1;
	int cols = binImg.cols - 1;
	for (int i = 1; i < rows - 1; i++)
	{
		int* data = lableImg.ptr<int>(i);
		for (int j = 1; j < cols - 1; j++)
		{
			if (data[j] == 1)
			{
				std::stack<std::pair<int, int>> neighborPixels;
				neighborPixels.push(std::pair<int, int>(i, j));     // 像素位置: <i,j>
				++label;  // 没有重复的团，开始新的标签
				while (!neighborPixels.empty())
				{
					std::pair<int, int> curPixel = neighborPixels.top(); //如果与上一行中一个团有重合区域，则将上一行的那个团的标号赋给它
					int curX = curPixel.first;
					int curY = curPixel.second;
					lableImg.at<int>(curX, curY) = label;

					neighborPixels.pop();

					if (lableImg.at<int>(curX, curY - 1) == 1)
					{//左边
						neighborPixels.push(std::pair<int, int>(curX, curY - 1));
					}
					if (lableImg.at<int>(curX, curY + 1) == 1)
					{// 右边
						neighborPixels.push(std::pair<int, int>(curX, curY + 1));
					}
					if (lableImg.at<int>(curX - 1, curY) == 1)
					{// 上边
						neighborPixels.push(std::pair<int, int>(curX - 1, curY));
					}
					if (lableImg.at<int>(curX + 1, curY) == 1)
					{// 下边
						neighborPixels.push(std::pair<int, int>(curX + 1, curY));
					}
				}
			}
		}
	}

}

void Two_Pass(const cv::Mat& binImg, cv::Mat& lableImg)    //两遍扫描法
{
	if (binImg.empty() ||
		binImg.type() != CV_8UC1)
	{
		return;
	}

	// 第一个通路

	lableImg.release();
	binImg.convertTo(lableImg, CV_32SC1);

	int label = 1;
	std::vector<int> labelSet;
	labelSet.push_back(0);
	labelSet.push_back(1);

	int rows = binImg.rows - 1;
	int cols = binImg.cols - 1;
	for (int i = 1; i < rows; i++)
	{
		int* data_preRow = lableImg.ptr<int>(i - 1);
		int* data_curRow = lableImg.ptr<int>(i);
		for (int j = 1; j < cols; j++)
		{
			if (data_curRow[j] == 1)
			{
				std::vector<int> neighborLabels;
				neighborLabels.reserve(2);
				int leftPixel = data_curRow[j - 1];
				int upPixel = data_preRow[j];
				if (leftPixel > 1)
				{
					neighborLabels.push_back(leftPixel);
				}
				if (upPixel > 1)
				{
					neighborLabels.push_back(upPixel);
				}

				if (neighborLabels.empty())
				{
					labelSet.push_back(++label);  // 不连通，标签+1
					data_curRow[j] = label;
					labelSet[label] = label;
				}
				else
				{
					std::sort(neighborLabels.begin(), neighborLabels.end());
					int smallestLabel = neighborLabels[0];
					data_curRow[j] = smallestLabel;

					// 保存最小等价表
					for (size_t k = 1; k < neighborLabels.size(); k++)
					{
						int tempLabel = neighborLabels[k];
						int& oldSmallestLabel = labelSet[tempLabel];
						if (oldSmallestLabel > smallestLabel)
						{
							labelSet[oldSmallestLabel] = smallestLabel;
							oldSmallestLabel = smallestLabel;
						}
						else if (oldSmallestLabel < smallestLabel)
						{
							labelSet[smallestLabel] = oldSmallestLabel;
						}
					}
				}
			}
		}
	}

	// 更新等价对列表
	// 将最小标号给重复区域
	for (size_t i = 2; i < labelSet.size(); i++)
	{
		int curLabel = labelSet[i];
		int preLabel = labelSet[curLabel];
		while (preLabel != curLabel)
		{
			curLabel = preLabel;
			preLabel = labelSet[preLabel];
		}
		labelSet[i] = curLabel;
	};

	for (int i = 0; i < rows; i++)
	{
		int* data = lableImg.ptr<int>(i);
		for (int j = 0; j < cols; j++)
		{
			int& pixelLabel = data[j];
			pixelLabel = labelSet[pixelLabel];
		}
	}
}
//彩色显示
cv::Scalar GetRandomColor()
{
	uchar r = 255 * (rand() / (1.0 + RAND_MAX));
	uchar g = 255 * (rand() / (1.0 + RAND_MAX));
	uchar b = 255 * (rand() / (1.0 + RAND_MAX));
	return cv::Scalar(b, g, r);
}


void LabelColor(const cv::Mat& labelImg, cv::Mat& colorLabelImg)
{
	if (labelImg.empty() ||
		labelImg.type() != CV_32SC1)
	{
		return;
	}

	std::map<int, cv::Scalar> colors;

	int rows = labelImg.rows;
	int cols = labelImg.cols;

	colorLabelImg.release();
	colorLabelImg.create(rows, cols, CV_8UC3);
	colorLabelImg = cv::Scalar::all(0);

	for (int i = 0; i < rows; i++)
	{
		const int* data_src = (int*)labelImg.ptr<int>(i);
		uchar* data_dst = colorLabelImg.ptr<uchar>(i);
		for (int j = 0; j < cols; j++)
		{
			int pixelValue = data_src[j];
			if (pixelValue > 1)
			{
				if (colors.count(pixelValue) <= 0)
				{
					colors[pixelValue] = GetRandomColor();
				}

				cv::Scalar color = colors[pixelValue];
				*data_dst++ = color[0];
				*data_dst++ = color[1];
				*data_dst++ = color[2];
			}
			else
			{
				data_dst++;
				data_dst++;
				data_dst++;
			}
		}
	}
}


void OpenClose(int pos)
{
	int n = open_close_pos - max_iters;
	int an = n > 0 ? n : -n;
	element = cvCreateStructuringElementEx(an * 2 + 1, an * 2 + 1, an, an, element_shape, 0);//创建结构元素  
	if (n < 0){
		cvErode(src, dst, element, 1);//腐蚀图像  
		cvDilate(dst, dst, element, 1);//膨胀图像  
	}
	else
	{
		cvDilate(dst,dst,element,1);//膨胀图像  
		cvErode(src,dst,element,1);//腐蚀图像  
	}
	cvReleaseStructuringElement(&element);
	cvShowImage("Open/Close", dst);
}

void ErodeDilate(int pos)
{
	int n = erode_dilate_pos - max_iters;
	int an = n > 0 ? n : -n;
	element = cvCreateStructuringElementEx(an * 2 + 1, an * 2 + 1, an, an, element_shape, 0);
	if (n < 0)
	{
		cvErode(src, dst, element, 1);
	}
	else
	{
		cvDilate(src,dst,element,1);
	}
	cvReleaseStructuringElement(&element);
	cvShowImage("Erode/Dilate", dst);
}

void morphology(int argc, char **argv)

{

	char *filename = argc == 2 ? argv[1] : (char *)"lena.jpg";
	if ((src = cvLoadImage(filename, 1)) == 0)
		return ;
 
	dst = cvCloneImage(src);

	cvNamedWindow("Open/Close", 1);
	cvNamedWindow("Erode/Dilate", 1);

	open_close_pos = erode_dilate_pos = max_iters;

	cvCreateTrackbar("iterations", "Open/Close", &open_close_pos, max_iters * 2 + 1, OpenClose);
	cvCreateTrackbar("iterations", "Erode/Dilate", &erode_dilate_pos, max_iters * 2 + 1, ErodeDilate);

	for (;;)
	{
		int c; 
		OpenClose(open_close_pos);
		ErodeDilate(erode_dilate_pos);
		c = cvWaitKey(0);
		if (c == 27)
		{
			break;
		}
		switch (c) {
		case 'e':
			element_shape = CV_SHAPE_ELLIPSE;
			break;
		case 'r':
			element_shape = CV_SHAPE_RECT;
			break;
		case '/r':
			element_shape = (element_shape + 1) % 3;
			break;
		default:
			break;
		}
	}
	cvReleaseImage(&src);
	cvReleaseImage(&dst);
	cvDestroyWindow("Open/Close");
	cvDestroyWindow("Erode/Dilate");
	return ;}
void cvErode(IplImage* src, IplImage* dst, IplConvKernel* B = NULL, int iterations = 1);
int main()
{   
	Mat  g_grayImage, g_dstImage;
	cv::Mat src_colar = imread("E:\\4.png",0);
	cv::Mat g_srcImage = imread("E:\\4.png");
	cv::threshold(src_colar, src_colar, 50, 1, CV_THRESH_BINARY_INV);
	cv::Mat labelImg;
	Two_Pass(src_colar, labelImg);
	//Seed_Filling(binImage, labelImg);
	//彩色显示
	cv::Mat colorLabelImg;
	LabelColor(labelImg, colorLabelImg);
	cv::imshow("colorImg", colorLabelImg);
	/*	//灰度显示
		cv::Mat grayImg;
		labelImg *= 10;
		labelImg.convertTo(grayImg, CV_8UC1);
		cv::imshow("labelImg", grayImg);
	*/

	cvtColor(g_srcImage, g_grayImage, COLOR_RGB2GRAY);//原图的灰度图
    std::cout << "Hello World!\n";
	imshow("Hello World", g_grayImage);
	waitKey(0);//等待用户按键


	cv::Mat binaryMat;
	cv::Mat labelMat;
	cv::Mat statsMat;
	cv::Mat centrMat;
	cv::Mat resultMat;
	cv::Mat srcMat = imread("E:\\coin.png", 0);
	cv::Mat srcMat2 = imread("E:\\IMG.jpg", 0);
	cv::Mat srcMat3 = imread("E:\\clip.png", 0);
	element = cvCreateStructuringElementEx(1 * 2 + 1, 1 * 2 + 1, 1, 1, element_shape, 0);
	int font_face = cv::FONT_HERSHEY_COMPLEX;
	double font_scale = 1.5;
	int thickness = 1;
	int baseline;

	int elementSize = 5;

	cout << "------program starts------ " << endl;

	if (srcMat.empty())
	{
		cout << "load image error!" << endl;
	}

	//二值化
	cv::threshold(srcMat, binaryMat, 0, 255, THRESH_OTSU);
/*	IplImage *image = cvLoadImage(binaryMat);
	CvMat *imageR = cvCreateMat(512, 512, CV_8UC3);
	cvResize(image, imageR);
	cvShowImage("原图", imageR);*/
	//CvMat *imageD1 = cvCreateMat(512, 512, CV_8UC3);
	//cvDilate(binaryMat, imageD1, element, 1);//膨胀图像
	//获得连通域
	//open
	Mat elementSizel = getStructuringElement(MORPH_ELLIPSE,Size(9,11));
	morphologyEx(srcMat, binaryMat, MORPH_OPEN, elementSizel);

	int nComp = cv::connectedComponentsWithStats(binaryMat,
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

	//-1,nComp包括背景
	cout << "the total of connected Components = " << nComp - 1 << endl;
	//通过颜色表现连通域
	//显示用图像
	resultMat = cv::Mat::zeros(srcMat.size(), CV_8UC3);
	std::vector<cv::Vec3b> colors(nComp);
	//背景使用黑色
	colors[0] = cv::Vec3b(0, 0, 0);
	//使用随机数产生函数randu，随机产生颜色
	for (int n = 1; n < nComp; n++)
	{
		colors[n] = cv::Vec3b(rand() / 255, rand() / 255, rand() / 255);
	}

	//对所有像素按照连通域编号进行着色
	for (int y = 0; y < srcMat.rows; y++)
	{
		for (int x = 0; x < srcMat.cols; x++)
		{
			int label = labelMat.at<int>(y, x);
			CV_Assert(0 <= label && label <= nComp);
			resultMat.at<cv::Vec3b>(y, x) = colors[label];
		}
	}

	//绘制bounding box
	for (int i = 1; i < nComp; i++)
	{
		char num[10];
		sprintf_s(num, "%d", i);

		Rect bndbox;
		//bounding box左上角坐标
		bndbox.x = statsMat.at<int>(i, 0);
		bndbox.y = statsMat.at<int>(i, 1);
		//bouding box的宽和长 
		bndbox.width = statsMat.at<int>(i, 2);
		bndbox.height = statsMat.at<int>(i, 3);
		//绘制
		rectangle(resultMat, bndbox, CV_RGB(255, 255, 255), 1, 8, 0);
		//连通域编号
		cv::putText(resultMat, num, Point(bndbox.x, bndbox.y), font_face, 1, cv::Scalar(0, 255, 255), thickness, 8, 0);
	}
	//将连通域数量绘制在图片上
	//设置绘制文本的相关参数
	char text[20];
	int length = sprintf_s(text, "%d", nComp - 1);

	//获取文本框的长宽
	cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, &baseline);
	//将文本框居中绘制
	cv::Point origin;
	origin.x = 0;
	origin.y = text_size.height;
	cv::putText(resultMat, text, origin, font_face, font_scale, cv::Scalar(0, 255, 255), thickness, 8, 0);

	imshow("binaryMat", binaryMat);
	imshow("results", resultMat);
	imshow("frame", srcMat);
	moveWindow("frame", 0, 20);
	moveWindow("binaryMat", srcMat.cols, 20);
	moveWindow("results", srcMat.cols * 2, 20);
	waitKey(0);

	return 0;
}
