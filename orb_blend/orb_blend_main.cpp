#include <iostream>
#include <bitset>
#include <vector>
#include <string>
using namespace std;

#include <opencv2/opencv.hpp>
//using namespace cv;

#include <timer.hpp>
#include "fastlib/fast.h"
#include "OpenBMP/OpenBMP.hpp"
#include "LATCH/LATCH.h"

#define ORB_DSET_R 15
float fastAtan2f_rad(float dy, float dx){
	static const float M_PI     = 3.14159265358979323846f;
	static const float atan2_p1 =  0.9997878412794807f;
	static const float atan2_p3 = -0.3258083974640975f;
	static const float atan2_p5 =  0.1555786518463281f;
	static const float atan2_p7 = -0.04432655554792128f;
	static const float atan2_DBL_EPSILON = 2.2204460492503131e-016;

	float ax = std::abs(dx), ay = std::abs(dy);
	float a, c, c2;
	if (ax >= ay) {
		c = ay/(ax + static_cast<float>(atan2_DBL_EPSILON));
		c2 = c*c;
		a = (((atan2_p7*c2 + atan2_p5)*c2 + atan2_p3)*c2 + atan2_p1)*c;
	} else {
		c = ax/(ay + static_cast<float>(atan2_DBL_EPSILON));
		c2 = c*c;
		a = M_PI/0.5 - (((atan2_p7*c2 + atan2_p5)*c2 + atan2_p3)*c2 + atan2_p1)*c;
	}
	if (dx < 0)
		a = M_PI - a;
	if (dy < 0)
		a = M_PI*2.0 - a;
	return a;
}
float fastAtan2f(float dy, float dx){
	static const float M_PI     = 3.14159265358979323846f;
	static const float atan2_p1 = 0.9997878412794807f*(float)(180/M_PI);
	static const float atan2_p3 = -0.3258083974640975f*(float)(180/M_PI);
	static const float atan2_p5 = 0.1555786518463281f*(float)(180/M_PI);
	static const float atan2_p7 = -0.04432655554792128f*(float)(180/M_PI);
	static const float atan2_DBL_EPSILON = 2.2204460492503131e-016;

	float ax = std::abs(dx), ay = std::abs(dy);
	float a, c, c2;
	if (ax >= ay) {
		c = ay/(ax + static_cast<float>(atan2_DBL_EPSILON));
		c2 = c*c;
		a = (((atan2_p7*c2 + atan2_p5)*c2 + atan2_p3)*c2 + atan2_p1)*c;
	} else {
		c = ax/(ay + static_cast<float>(atan2_DBL_EPSILON));
		c2 = c*c;
		a = 90.f - (((atan2_p7*c2 + atan2_p5)*c2 + atan2_p3)*c2 + atan2_p1)*c;
	}
	if (dx < 0)
		a = 180.f - a;
	if (dy < 0)
		a = 360.f - a;
	return a;
}
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
	int maskEdg = ORB_DSET_R;
	for(int i = 0; i < corner.num; i++) {
		int x=corner.posi[i].x;
		int y=corner.posi[i].y;
		// 過濾邊緣位置
		if( x>=(maskEdg) && x<=(int)grayImg.width-(maskEdg) &&
			y>=(maskEdg) && y<=(int)grayImg.height-(maskEdg))
		{
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
		key.emplace_back(kp.x, kp.y, 31.f, -1.f);
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
	constexpr int r = ORB_DSET_R;
	for(int idx = 0; idx < key.size();) {
		//double m00 = 0;
		double m01 = 0;
		double m10 = 0;
		for(int j = -r; j <= r; j++) {
			for(int i = -r; i <= r; i++) {
				int x = key[idx].x +i;
				int y = key[idx].y +j;
				int idx = y*grayImg.width +x;

				//m00 +=     grayImg.raw_img[idx];
				m10 += i * grayImg.raw_img[idx];
				m01 += j * grayImg.raw_img[idx];

				// debug
				if((x < 0.0 or x >= grayImg.width) or
					(y < 0.0 or y >= grayImg.height)) {
					cout << "thisx=" << x << endl;
					cout << "thisy=" << y << endl;
					throw out_of_range("出現負號");
				}
			}
		}
		key[idx++].angle = fastAtan2f_rad(m01, m10);
	}
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
	t1.start();
	keyPt_grayCentroidAngle(key, grayImg);
	t1.print(" keyPt_grayCentroidAngle");

	// desc
	vector<uint64_t> desc(8 * key.size());
	t1.start();
	LATCH::LATCH<1>(grayImg.raw_img.data(), grayImg.width, grayImg.height, 
		static_cast<int>(grayImg.width), key, desc.data());
	t1.print(" LATCH");

	cout << "descNum=" << desc.size()/8 << endl;

	// 查看描述值
	vector<uint64_t>& desc2=desc;
	vector<LATCH::KeyPoint>& key2 = key;
	t1.start();
	for (int keyIdx = 0; keyIdx < 2; keyIdx++) {
		int dist=0;
		for (int i = 0; i < 8; ++i) {
			//cout << bitset<64>(desc[keyIdx+i]);

			dist += (bitset<64>(desc[(keyIdx)*8+i])^bitset<64>(desc2[(keyIdx+1)*8+i])).count();

		} //cout << endl;

		cout << "dist=" << dist << endl;
	}
	t1.print(" distence");

	return 0;
}
//====================================================================================
