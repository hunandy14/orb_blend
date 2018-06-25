#include <iostream>
#include <bitset>
#include <vector>
#include <string>
using namespace std;

#include <opencv2/opencv.hpp>
//using namespace cv;

#include <timer.hpp>
#include "fast.h"
#include "OpenBMP.hpp"
#include "LATCH.h"
#include "orb.hpp"

#define ORB_DSET_R 15

// =====================================================================================
// 輸出測試圖函式


// =====================================================================================
struct basic_FAST {
	xy* posi = nullptr;
	int num = 0;
};
class FAST9 : public basic_FAST {
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
	const basic_ImgData& grayImg, const basic_FAST& corner = basic_FAST())
{
	using cv::Mat;
	using cv::UMat;

	int minEdg = std::min(grayImg.width, grayImg.height);
	int HarrisNum = 256;
	int Harrisdis = minEdg / 20;

	Harrisdis = std::max(Harrisdis, 3);
	Harrisdis = std::min(Harrisdis, 50);


	// 初始化mask --- >0ms
	Mat cvImg(grayImg.height, grayImg.width, CV_8U, (void*)grayImg.raw_img.data());
	Mat herrisMask(Mat::zeros(cv::Size(grayImg.width, grayImg.height), CV_8U));

	vector<uint8_t> harrisMask;

	// 建構遮罩
	int maskEdg = ORB_DSET_R;
	for (int i = 0; i < corner.num; i++) {
		int x = corner.posi[i].x;
		int y = corner.posi[i].y;
		// 過濾邊緣位置
		if (x >= (maskEdg) && x <= (int)grayImg.width - (maskEdg) &&
			y >= (maskEdg) && y <= (int)grayImg.height - (maskEdg))
		{
			herrisMask.at<uchar>(cv::Point(x, y)) = 255;
		}
	}

	// Herris 角點
	vector<cv::Point2f> corners;
	UMat ucvImg = cvImg.getUMat(cv::ACCESS_RW);

	if (corner.posi) {
		//goodFeaturesToTrack(ucvImg, corners, 445, 0.01, 30, cv::noArray(), 3, true);
		//goodFeaturesToTrack(ucvImg, corners, 445, 0.01, 3, herrisMask, 3, true);
	}
	else {
		goodFeaturesToTrack(ucvImg, corners, HarrisNum, 0.01, Harrisdis, cv::noArray(), 3, true);
	}

	// 輸出到 keyPt
	key.clear();
	for (auto&& kp : corners)
		key.emplace_back(kp.x, kp.y, 31.f, -1.f);
}


void FATS_drawPoint(ImgData out, const FAST9& corner, string name = "_FSATCorner") {
	using uch = unsigned char;
	auto drawRGB = [](uch* p, uch r, uch g, uch b) {
		p[0] = r, p[1] = g, p[2] = b;
	};
	for (size_t i = 0; i < corner.size(); i++) {
		int x = corner[i].x;
		int y = corner[i].y;
		unsigned char* p = out.at2d(y, x);
		int r = 1;
		for (int rj = -r; rj <= r; rj++) {
			for (int ri = -r; ri <= r; ri++) {
				drawRGB(out.at2d(y + rj, x + ri), 242, 66, 54);
			}
		}
	}
	static int num = 0;
	out.bmp(name + to_string(num++) + ".bmp");
}
void keyPt_drawPoint(ImgData out, const vector<LATCH::KeyPoint>& corner, string name = "_drawKeyPt")
{
	using uch = unsigned char;
	auto drawRGB = [](uch* p, uch r, uch g, uch b) {
		p[0] = r, p[1] = g, p[2] = b;
	};
	for (size_t i = 0; i < corner.size(); i++) {
		const int x = (int)corner[i].x;
		const int y = (int)corner[i].y;

		int r = 2;
		for (int rj = -r; rj <= r; rj++) {
			for (int ri = -r; ri <= r; ri++) {
				drawRGB(out.at2d(y + rj, x + ri), 242, 66, 54);
			}
		}
	}
	static int num = 0;
	out.bmp(name + to_string(num++) + ".bmp");
}
void keyPt_drawAngle(ImgData out, const vector<LATCH::KeyPoint>& key, string name = "_drawKeyPtAngle")
{
	using uch = unsigned char;
	for (size_t i = 0; i < key.size(); i++) {
		const int x = (int)key[i].x;
		const int y = (int)key[i].y;
		Draw::draw_arrowRGB(out, key[i].y, key[i].x, 20, key[i].angle * 180 / 3.14);
	}
	static int num = 0;
	out.bmp(name + to_string(num++) + ".bmp");
}

void keyPt_grayCentroidAngle(vector<LATCH::KeyPoint>& key, const basic_ImgData& grayImg) {
	constexpr int r = ORB_DSET_R;
	for (int idx = 0; idx < key.size();) {
		//double m00 = 0;
		double m01 = 0;
		double m10 = 0;
		for (int j = -r; j <= r; j++) {
			for (int i = -r; i <= r; i++) {
				int x = key[idx].x + i;
				int y = key[idx].y + j;
				int idx = y * grayImg.width + x;

				// debug
				if ((x < 0.0 or x >= grayImg.width) or
					(y < 0.0 or y >= grayImg.height)) {
					/*cout << "thisx=" << x << endl;
					cout << "thisy=" << y << endl;*/
					continue;
					throw out_of_range("出現負號");
				}

				//m00 +=     grayImg.raw_img[idx];
				m10 += i * grayImg.raw_img[idx];
				m01 += j * grayImg.raw_img[idx];
			}
		}
		//cout << "m01=" << m01 << ", " << "m10::" << m10;
		float a = fastAtan2f(m01, m10);
		//cout << "angle=" << a << endl;
		key[idx++].angle = a * 3.14 / 180.0;
	}
}
int desc_distance(const vector<uint64_t>& desc1, const vector<uint64_t>& desc2, int idx1, int idx2) {
	int dist = 0;
	dist += (bitset<64>(desc1[idx1 * 8 + 0]) ^ bitset<64>(desc2[idx2 * 8 + 0])).count();
	dist += (bitset<64>(desc1[idx1 * 8 + 1]) ^ bitset<64>(desc2[idx2 * 8 + 1])).count();
	dist += (bitset<64>(desc1[idx1 * 8 + 2]) ^ bitset<64>(desc2[idx2 * 8 + 2])).count();
	dist += (bitset<64>(desc1[idx1 * 8 + 3]) ^ bitset<64>(desc2[idx2 * 8 + 3])).count();
	dist += (bitset<64>(desc1[idx1 * 8 + 4]) ^ bitset<64>(desc2[idx2 * 8 + 4])).count();
	dist += (bitset<64>(desc1[idx1 * 8 + 5]) ^ bitset<64>(desc2[idx2 * 8 + 5])).count();
	dist += (bitset<64>(desc1[idx1 * 8 + 6]) ^ bitset<64>(desc2[idx2 * 8 + 6])).count();
	dist += (bitset<64>(desc1[idx1 * 8 + 7]) ^ bitset<64>(desc2[idx2 * 8 + 7])).count();
	return dist;
}
void ORB_dsec(const ImgData& grayImg, vector<LATCH::KeyPoint>& key, vector<uint64_t>& desc) {
	Timer t1;
	t1.priSta = 1;

	// FAST
	//t1.start();
	//FAST9 corner(grayImg);
	//t1.print(" FAST12 Corner");
	//FATS_drawPoint(img, corner); // 驗證::畫點

	// Harris
	t1.start();
	//keyPt_HarrisCroner(key, grayImg, corner);
	keyPt_HarrisCroner(key, grayImg);
	t1.print("    Harris Corner");
	//cout << "Herris Corner num = " << key.size() << endl;
	//keyPt_drawPoint(img, key); // 驗證::畫點

	// angle
	t1.start();
	keyPt_grayCentroidAngle(key, grayImg);
	t1.print("    keyPt_grayCentroidAngle");
	//keyPt_drawAngle(img, key); // 驗證::畫箭頭


	// desc
	t1.start();
	desc.resize(8 * key.size());
	LATCH::LATCH<0>(grayImg.raw_img.data(), grayImg.width, grayImg.height, static_cast<int>(grayImg.width), key, desc.data());
	t1.print("    LATCH");
}

// =====================================================================================
class DMatch
{
public:
	DMatch() : queryIdx(-1), trainIdx(-1), imgIdx(-1), distance(FLT_MAX) {}
	DMatch(int _queryIdx, int _trainIdx, float _distance) :
		queryIdx(_queryIdx), trainIdx(_trainIdx), imgIdx(-1), distance(_distance) {}

	DMatch(int _queryIdx, int _trainIdx, int _imgIdx, float _distance) :
		queryIdx(_queryIdx), trainIdx(_trainIdx), imgIdx(_imgIdx), distance(_distance) {}


	int queryIdx; // query descriptor index
	int trainIdx; // train descriptor index
	int imgIdx;   // train image index

	float distance;

	// less is better
	bool operator < (const DMatch &m) const {
		return distance < m.distance;
	}
};
void keyPt_drawMatchLine(
	const ImgData& img1, vector<LATCH::KeyPoint>& key1, 
	const ImgData& img2, vector<LATCH::KeyPoint>& key2,
	vector<DMatch>& dmatch, string name = "__matchImg.bmp")
{
	using uch = unsigned char;
	// 合併圖像
	ImgData matchImg(img1.width * 2, img1.height, img1.bits);
	for (int j = 0; j < matchImg.height; j++) {
		// img1
		for (int i = 0; i < img1.width; i++) {
			uch* mp = matchImg.at2d(j, i);
			const uch* p1 = img1.at2d(j, i);
			mp[0] = p1[0];
			mp[1] = p1[1];
			mp[2] = p1[2];
		}
		// img2
		for (int i = 0; i < img2.width; i++) {
			uch* mp = matchImg.at2d(j, i + img1.width);
			const uch* p2 = img2.at2d(j, i);
			mp[0] = p2[0];
			mp[1] = p2[1];
			mp[2] = p2[2];
		}
	}
	// 連線
	int drawNum = 0;
	for (size_t i = 0; i < dmatch.size(); i++) {
		if (dmatch[i].distance != FLT_MAX) {
			int x1 = key1[dmatch[i].queryIdx].x;
			int y1 = key1[dmatch[i].queryIdx].y;
			int x2 = key2[dmatch[i].trainIdx].x + img2.width;
			int y2 = key2[dmatch[i].trainIdx].y;
			Draw::drawLineRGB_p(matchImg, y1, x1, y2, x2);
			drawNum++;
		}
	}
	//cout << "draw Num = " << drawNum << "/" << dmatch.size() << endl;
	matchImg.bmp(name);
}
void ORB_match(vector<LATCH::KeyPoint>& key1, vector<uint64_t>& desc1,
	vector<LATCH::KeyPoint>& key2, vector<uint64_t>& desc2, vector<DMatch>& dmatch)
{
	dmatch.resize(key1.size());				// 由key1去找key2
	const float matchDistance = 128;			// 少於多少距離才選定
	const float noMatchDistance = 192;		// 大於多少距離就不連

	int matchNum = 0;
	for (int j = 1; j < key1.size(); j++) {
		float& distMin = dmatch[j].distance;
		int& matchIdx = dmatch[j].trainIdx;
		// key1[0]去尋找 key2[all] 中最短距離者
		for (int i = 1; i < key2.size(); i++) {
			int dist = desc_distance(desc2, desc1, j, i);
			if (dist > noMatchDistance) continue;
			if (dist < distMin) {
				distMin = dist;
				matchIdx = i;
			}
		}
		// 紀錄最短距離與索引
		if (distMin < matchDistance) {
			matchNum++;
			// 紀錄當前最短距離者
			dmatch[j].queryIdx = matchIdx;
			dmatch[j].trainIdx = j;
			dmatch[j].distance = distMin;
		}
		else {
			dmatch[j].queryIdx = 0;
			dmatch[j].trainIdx = 0;
			dmatch[j].distance = FLT_MAX;
		}
	}
	cout << "Match Num = " << matchNum << "/" << dmatch.size() << endl;
}





// =====================================================================================
void ORB_test(const ImgData& img1, const ImgData& img2) {
	Timer t1, t0;
	// 轉灰階
	const ImgData gray1 = img1.toConvertGray();
	const ImgData gray2 = img2.toConvertGray();

	t0.start();
	// 取得關鍵點與描述值
	vector<LATCH::KeyPoint> key1, key2;
	vector<uint64_t> desc1, desc2;
	t1.start();
	ORB_dsec(gray1, key1, desc1);
	ORB_dsec(gray2, key2, desc2);
	t1.print("  ORB_desc");

	// 匹配描述值
	vector<DMatch> dmatch;
	t1.start();
	ORB_match(key1, desc1, key2, desc2, dmatch);
	t1.print("  ORB_match");
	//keyPt_drawMatchLine(img1, key1, img2, key2, dmatch);

	using namespace cv;
	// 隨機抽樣
	vector<cv::Point2f> featPoint1;
	vector<cv::Point2f> featPoint2;
	for (size_t i = 0; i < dmatch.size(); i++) {
		if (dmatch[i].distance != FLT_MAX && dmatch[i].queryIdx != 0 && dmatch[i].trainIdx != 0) {
			int x = key1[dmatch[i].queryIdx].x;
			int y = key1[dmatch[i].queryIdx].y;
			int x2 = key2[dmatch[i].trainIdx].x;
			int y2 = key2[dmatch[i].trainIdx].y;

			cv::Point pt(x, y);
			cv::Point pt2(x2, y2);
			featPoint1.push_back(pt);
			featPoint2.push_back(pt2);
		}
	}

	// get Homography and RANSAC mask
	vector<uint8_t> RANSAC_mask;
	t1.start();
	cv::Mat Hog = cv::findHomography(featPoint2, featPoint1, cv::RANSAC, 3, RANSAC_mask, 2000, 0.995);
	t1.print("  findHomography");
	//keyPt_drawMatchLine(img1, key1, img2, key2, dmatch, RANSAC_mask);
	t0.print("## ALL");

	// 合併圖像
	using uch = unsigned char;
	ImgData matchImg(img1.width * 2, img1.height, img1.bits);
	for (int j = 0; j < matchImg.height; j++) {
		// img1
		for (int i = 0; i < img1.width; i++) {
			uch* mp = matchImg.at2d(j, i);
			const uch* p1 = img1.at2d(j, i);
			mp[0] = p1[0];
			mp[1] = p1[1];
			mp[2] = p1[2];
		}
		// img2
		for (int i = 0; i < img2.width; i++) {
			uch* mp = matchImg.at2d(j, i + img1.width);
			const uch* p2 = img2.at2d(j, i);
			mp[0] = p2[0];
			mp[1] = p2[1];
			mp[2] = p2[2];
		}
	}


	// 更新到 Dmatch
	int RANNum = 0;
	vector<cv::DMatch> RANmatch;
	for (size_t i = 0; i < RANSAC_mask.size(); i++) {
		if (RANSAC_mask[i] != 0) { // 正確的
			int x1 = featPoint1[i].x;
			int y1 = featPoint1[i].y;
			int x2 = featPoint2[i].x + img2.width;
			int y2 = featPoint2[i].y;
			Draw::drawLineRGB_p(matchImg, y1, x1, y2, x2);
			RANNum++;
		}
	}
	matchImg.bmp("__mth.bmp");

	cout << "RANNum = " << RANNum << endl;
	cout << "Hog = \n" << Hog << endl;
}



// =====================================================================================
void cvInitializeOpenCL() {
	cout << "cvInitializeOpenCL...\n";
	const ImgData img1("kanna.bmp");
	vector<LATCH::KeyPoint> key;
	keyPt_HarrisCroner(key, img1.toConvertGray());
	/*using namespace cv;
	vector<Point2f> corners;
	Mat cvImg(1, 1, CV_8U);
	UMat ugray = cvImg.getUMat(ACCESS_RW);
	Mat herrisMask(Mat::zeros(Size(cvImg.cols, cvImg.rows),CV_8U));
	goodFeaturesToTrack(ugray, corners, 2000, 0.01, 3, herrisMask);*/
	cout << "cvInitializeOpenCL...done\n\n";
}