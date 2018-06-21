#include <iostream>
using namespace std;

#include <opencv2/opencv.hpp>
//using namespace cv;

#include <timer.hpp>
#include "fastlib/fast.h"
#include "OpenBMP/OpenBMP.hpp"
#include "LATCH/LATCH.h"

struct KeyPt {
	float x, y, scale;

	// RADIANS
	float angle;

	// angle in RADIANS
	KeyPt(const float _x, const float _y, const float _scale, const float _angle) : x(_x), y(_y), scale(_scale), angle(_angle) {}
};
//====================================================================================
void cvInitializeOpenCL() {
	using namespace cv;
	vector<Point2f> corners;
	Mat cvImg(1, 1, CV_8U);
	UMat ugray = cvImg.getUMat(ACCESS_RW);
	Mat herrisMask(Mat::zeros(Size(cvImg.cols, cvImg.rows),CV_8U));
	goodFeaturesToTrack(ugray, corners, 2000, 0.01, 3, herrisMask);
}

struct basic_FAST {
	xy* posi;
	int num;
};
class FAST9: public basic_FAST{
public:
	FAST9(const basic_ImgData& grayImg) {
		const unsigned char* data = grayImg.raw_img.data();
		const int xsize = grayImg.width;
		const int ysize = grayImg.height;
		const int stride = xsize;
		const int threshold = 16;

		posi = fast9_detect_nonmax(data, xsize, ysize, xsize, threshold, &num);
	}
public:
	void info_print() {
		cout << "corner_num = " << num << endl;
	}
	int size() const {
		return num;
	}
	operator const xy*() const {
		return posi;
	}
};

void keyPt_HarrisCroner(vector<LATCH::KeyPoint>& key, 
	const basic_ImgData& grayImg, const basic_FAST& corner)
{
	using namespace cv;

	Timer t1;
	t1.priSta=0;

	// 初始化mask --- >0ms
	Mat cvImg(grayImg.height, grayImg.width, CV_8U, (void*)grayImg.raw_img.data());
	Mat herrisMask(Mat::zeros(Size(grayImg.width, grayImg.height),CV_8U));
	
	// 建構遮罩
	int maskEdg=3+20;
	for(int i = 0; i < corner.num; i++) {
		int x=corner.posi[i].x;
		int y=corner.posi[i].y;
		// 過濾邊緣位置
		if( x>=(maskEdg) && x<=(int)grayImg.width-(maskEdg) &&
			y>=(maskEdg) && y<=(int)grayImg.height-(maskEdg)) {
			herrisMask.at<uchar>(Point(x, y)) = 255;
		}
	}

	// Herris 角點
	vector<cv::Point2f> corners;
	UMat ucvImg = cvImg.getUMat(cv::ACCESS_RW);
	//goodFeaturesToTrack(cvImg, corners, 2000, 0.01, 3, herrisMask, 3, true);
	goodFeaturesToTrack(ucvImg, corners, 2000, 0.01, 3, herrisMask, 3, true);
	//goodFeaturesToTrack(ucvImg, corners, 445, 0.01, 3, noArray(), 3, true);

	// 輸出到 keyPt
	key.clear();
	for (auto&& kp : corners)
		key.emplace_back(kp.x, kp.y, 0.f, -1.f);
}

void FATS_drawPoint(ImgData out, const FAST9& corner) {
	for (size_t i = 0; i < corner.size(); i++) {
		int x = corner[i].x;
		int y = corner[i].y;
		unsigned char* p = out.at2d(y, x);
		p[0] = 255;
		p[1] = 0;
		p[2] = 0;
	}
	out.bmp("_FSATCorner.bmp");
}
void keyPt_drawPoint(ImgData out, const vector<LATCH::KeyPoint>& corner) {
	using uch = unsigned char;
	auto drawRGB = [](uch* p, uch r, uch g, uch b) {
		p[0] = r, p[1] = g, p[2] = b;
	};
	for (size_t i = 0; i < corner.size(); i++) {
		const int x = (int)corner[i].x;
		const int y = (int)corner[i].y;
		drawRGB(out.at2d(y+0, x+0), 255, 0, 0);
		drawRGB(out.at2d(y+1, x+0), 255, 0, 0);
		drawRGB(out.at2d(y+0, x+1), 255, 0, 0);
		drawRGB(out.at2d(y-1, x+0), 255, 0, 0);
		drawRGB(out.at2d(y+0, x-1), 255, 0, 0);
	}
	out.bmp("_drawKeyPt.bmp");
}

void keyPt_grayCentroidAngle(vector<LATCH::KeyPoint>& key, const basic_ImgData& grayImg) {

}
//====================================================================================
int main(int argc, char const *argv[]) {
	cvInitializeOpenCL();

	Timer t1;
	const ImgData img("ball_01_blend.bmp");
	//const ImgData img("kanna.bmp");
	//const ImgData img("lena.bmp");

	const ImgData grayImg = img.toConvertGray();
	

	// FAST
	t1.start();
	FAST9 corner(grayImg);
	t1.print(" FAST12");
	FATS_drawPoint(img, corner);
	// Harris
	vector<LATCH::KeyPoint> key;
	t1.start();
	keyPt_HarrisCroner(key, grayImg, corner);
	t1.print(" dstCorner");
	cout << "Herris Corner num = " << key.size() << endl;
	keyPt_drawPoint(img, key);
	// angle
	keyPt_grayCentroidAngle(key, grayImg);


	return 0;
}
//====================================================================================
