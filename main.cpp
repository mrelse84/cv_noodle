#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"
#include <iostream>

using namespace std;
using namespace cv;

int main()
{
	// OpenCV Version
	cout << "OpenCV Version : " << CV_VERSION << endl;

	// OpenCL을 사용할 수 있는지 테스트 
	if (!ocl::haveOpenCL()) {
		cout << "에러 : OpenCL을 사용할 수 없는 시스템입니다." << endl;
		return  -1;
	}

	// 컨텍스트 생성
	ocl::Context context;
	if (!context.create(ocl::Device::TYPE_GPU)) {
		cout << " 에러 : 컨텍스트를 생성할 수 없습니다." << endl;
		return  -1;
	}

	// GPU 장치 정보
	cout << context.ndevices() << " GPU device (s) detected " << endl;
	for (size_t i = 0; i < context.ndevices(); i++) {
		ocl::Device device = context.device(i);
		cout << " - Device " << i << " --- " << endl;
		cout << " Name : " << device.name() << endl;
		cout << " Availability : " << device.available() << endl;
		cout << "Image Support : " << device.imageSupport() << endl;
		cout << " OpenCL C version : " << device.OpenCL_C_Version() << endl;
	}

	// 장치 0 번 사용 
	ocl::Device(context.device(0));

	// Enable OpenCL
	ocl::setUseOpenCL(true);

	// 실행 시간 측정 
	static int64 start, end;
	static float time;

	// Load Noodle Image
	Mat img = imread("d:/images/CJ_Noodle/200422/NG/9864,9865.jpg", IMREAD_GRAYSCALE);
	//Mat img = imread("d:/images/CJ_NW_Noodle/200422/NG/9852-9853.jpg");
	//Mat img = imread("d:/images/CJ_NW_Noodle/200422/NG/die_03.png");

	if (img.empty())
	{
		cerr << "File open failed!" << endl;
		return -1;
	}

	namedWindow("image", WINDOW_NORMAL);
	resizeWindow("image", img.cols / 10, img.rows / 10);
	imshow("image", img);
	waitKey();

	// Threshold
	//=======================================================================================
	//
	Mat th_img;

	start = getTickCount();
	threshold(img, th_img, 200, 255, THRESH_BINARY);
	end = getTickCount();
	time = (end - start) / getTickFrequency() * 1000;
	cout << "threshold - Processing Time : " << time << " msec. " << endl;

	//threshold(img, th_img, 200, 255, THRESH_OTSU);
	//adaptiveThreshold(img, th_img, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 5, 10);
	imshow("image", th_img);
	waitKey();

	return 0;
}