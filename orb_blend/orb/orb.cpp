#include <iostream>
#include <bitset>
#include <vector>
#include <string>
#include <cmath>
using namespace std;

#include <opencv2/opencv.hpp>
//using namespace cv;

#include "Timer.hpp"
#include "OpenBMP.hpp"
#include "LATCH.h"
#include "orb.hpp"

#define ORB_DSET_R 15
#define HARRIS_total 20000
#define HARRIS_DIST 3
#define HARRIS_de 3
#define ORB_DIST 96
#define ORB_DISTMAX 128



// =====================================================================================
/*
#include "FAST9.hpp"
void FATS_drawPoint(ImgData out, const FAST9& pt, string name = "_FSATCorner") {
	using uch = unsigned char;
	out.convertRGB();
	auto drawRGB = [](uch* p, uch r, uch g, uch b) {
		p[0] = r, p[1] = g, p[2] = b;
	};
	for (size_t i = 0; i < pt.size(); i++) {
		int x = pt[i].x;
		int y = pt[i].y;
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
void oFASTCroner(vector<LATCH::KeyPoint>& key, const basic_ImgData& grayImg)
{
	using cv::Mat;
	using cv::UMat;

	int edgeMaskDist = ORB_DSET_R;
	int HarrisNum = 256;
	int minEdg = std::min(grayImg.width, grayImg.height);
	int HarrisDist = minEdg / 20;
	HarrisDist = std::max(HarrisDist, 3);
	HarrisDist = std::min(HarrisDist, 50);

	// Herris 角點啟用 GPU 運算
	vector<cv::Point2f> corners;
	Mat cvImg(grayImg.height, grayImg.width, CV_8U, (void*)grayImg.raw_img.data());
	UMat ucvImg = cvImg.getUMat(cv::ACCESS_READ);

	// FAST 遮罩
	FAST9 pt(grayImg);

	if (pt.pt!=nullptr) {
		Mat FASTMask(Mat::zeros(cv::Size(grayImg.width, grayImg.height), CV_8U));
		for (int i = 0; i < pt.num; i++) {
			int x = pt.pt[i].x;
			int y = pt.pt[i].y;
			// 過濾邊緣位置
			if (x >= (edgeMaskDist) && x <= (int)grayImg.width - (edgeMaskDist) &&
				y >= (edgeMaskDist) && y <= (int)grayImg.height - (edgeMaskDist))
			{
				int idx = y * grayImg.width + x;
				FASTMask.at<uchar>(idx) = 255;
			}
		}
		goodFeaturesToTrack(ucvImg, corners, 445, 0.01, 3, FASTMask, 3, true);
	}

	// 輸出到 keyPt
	key.clear();
	for (auto&& kp : corners) {
		key.emplace_back(kp.x, kp.y, 31.f, -1.f);
	}
}
*/


// =====================================================================================
/// 輸出測試圖函式

void keyPt_drawPoint(ImgData out, const vector<LATCH::KeyPoint>& corner, string name = "_drawKeyPt")
{
	using uch = unsigned char;
	out.convertRGB();
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
	float radial = 1.f;
	radial = 180.f/3.141592654f; // 來源是逕度

	out.convertRGB();
	for (size_t i = 0; i < key.size(); i++) {
		const int x = (int)key[i].x;
		const int y = (int)key[i].y;
		Draw::draw_arrowRGB(out, key[i].y, key[i].x, 20, key[i].angle * radial);
	}
	static int num = 0;
	out.bmp(name + to_string(num++) + ".bmp");
}
void keyPt_drawMatchLine(
	const ImgData& img1, vector<LATCH::KeyPoint>& key1, 
	const ImgData& img2, vector<LATCH::KeyPoint>& key2,
	vector<ORB::DMatch>& dmatch, string name = "__matchImg.bmp")
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
void keyPt_drawRANSACLine(const ImgData& img1, vector<LATCH::KeyPoint>& key1, 
	const ImgData& img2, vector<LATCH::KeyPoint>& key2,
	vector<ORB::DMatch>& dmatch, string name = "__matchImg.bmp")
{
	vector<cv::Point> fp1, fp2;

	// 獲取有效座標
	for (int i=0, c=0; i < dmatch.size(); i++) {
		if (dmatch[i].distance != FLT_MAX) {
			int x1 = key1[dmatch[i].queryIdx].x;
			int y1 = key1[dmatch[i].queryIdx].y;
			int x2 = key2[dmatch[i].trainIdx].x;
			int y2 = key2[dmatch[i].trainIdx].y;
			fp1.emplace_back(cv::Point(x1, y1));
			fp2.emplace_back(cv::Point(x2, y2));
		}
	}
	// 隨機抽樣
	vector<uint8_t> RANSAC_mask;
	cv::Mat Hog = cv::findHomography(fp2, fp1, cv::RANSAC, 3, RANSAC_mask, 2000, 0.995);

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
	// 畫線
	int RANNum = 0;
	for (size_t i = 0; i < RANSAC_mask.size(); i++) {
		if (RANSAC_mask[i] != 0) { // 正確的
			int x1 = fp1[i].x;
			int y1 = fp1[i].y;
			int x2 = fp2[i].x + img2.width;
			int y2 = fp2[i].y;
			Draw::drawLineRGB_p(matchImg, y1, x1, y2, x2);
			RANNum++;
		}
	}

	// 輸出
	matchImg.bmp("__mth.bmp");
	cout << "Hog = \n" << Hog << endl;
	cout << "RANNum = " << RANNum << "/" << RANSAC_mask.size() << endl;
}
void keyPt_drawMatchLine(
	const ImgData& img1, vector<LATCH::KeyPoint>& key1, 
	const ImgData& img2, vector<LATCH::KeyPoint>& key2,
	string name = "__matchImg.bmp")
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
	for (size_t i = 0; i < key1.size(); i++) {
		int x1 = key1[i].x;
		int y1 = key1[i].y;
		int x2 = key2[i].x + img2.width;
		int y2 = key2[i].y;
		Draw::drawLineRGB_p(matchImg, y1, x1, y2, x2);
		drawNum++;
	}
	cout << "draw Num = " << drawNum << endl;
	matchImg.bmp(name);
}


// =====================================================================================
/// ORB 流程

// Harris 角點檢測
void HarrisCroner(vector<LATCH::KeyPoint>& key, const basic_ImgData& grayImg)
{
	using cv::Mat;
	using cv::UMat;

	int edgeMaskDist = ORB_DSET_R;
	int HarrisNum = HARRIS_total;
	int HarrisDist = HARRIS_DIST;
	/*HarrisDist = std::min(grayImg.width, grayImg.height) / 20;
	HarrisDist = std::max(HarrisDist, 3);
	HarrisDist = std::min(HarrisDist, 3);*/

	// Herris 角點啟用 GPU 運算
	vector<cv::Point2f> corners;
	Mat cvImg(grayImg.height, grayImg.width, CV_8U, (void*)grayImg.raw_img.data());
	UMat ucvImg = cvImg.getUMat(cv::ACCESS_READ);

	// 過濾邊緣的MASK
	Mat edgeMask(Mat::zeros(cv::Size(grayImg.width, grayImg.height), CV_8U));
	for (int j = edgeMaskDist; j < grayImg.height - edgeMaskDist; j++) {
		for (int i = edgeMaskDist; i < grayImg.width - edgeMaskDist; i++) {
			int idx = j * grayImg.width + i;
			edgeMask.at<uchar>(idx) = 255;
		}
	}
	int de = HARRIS_de;
	goodFeaturesToTrack(ucvImg, corners, HarrisNum, 0.01, HarrisDist, edgeMask, 3, true);
	goodFeaturesToTrack(ucvImg, corners, corners.size()>>de, 0.01, (HarrisDist<<de)+.5, edgeMask, 3, true);

	// 輸出到 keyPt
	key.clear();
	for (auto&& kp : corners) {
		key.emplace_back(kp.x, kp.y, 31.f, -1.f);
	}
}
// 灰度質心
void grayCentroidAngle(vector<LATCH::KeyPoint>& key, const basic_ImgData& grayImg) {
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
				/*if ((x < 0.0 or x >= grayImg.width) or
					(y < 0.0 or y >= grayImg.height)) {
					//cout << "thisx=" << x << endl;
					//cout << "thisy=" << y << endl;
					throw out_of_range("出現負號");
					continue;
				}*/

				//m00 +=     grayImg.raw_img[idx];
				m10 += i * grayImg.raw_img[idx];
				m01 += j * grayImg.raw_img[idx];
			}
		}
		key[idx++].angle = fastAtan2f_rad(m01, m10);
	}
}
// 描述
int descDistance(const vector<uint64_t>& desc1, const vector<uint64_t>& desc2, int idx1, int idx2) {
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



// =====================================================================================
/// 縫合流程

// ORB 描述
void ORB_dsec(const ImgData& grayImg, vector<LATCH::KeyPoint>& key, vector<uint64_t>& desc) {
	Timer t1;
	t1.priSta = 0;
	
	// KeyPoint
	t1.start();
	HarrisCroner(key, grayImg);
	//oFASTCroner(key, grayImg);
	t1.print("    Harris Corner");

	// angle
	t1.start();
	grayCentroidAngle(key, grayImg);
	t1.print("    keyPt_grayCentroidAngle");

	// desc
	t1.start();
	desc.resize(8 * key.size());
	LATCH::LATCH<1>(grayImg.raw_img.data(), grayImg.width, grayImg.height, static_cast<int>(grayImg.width), key, desc.data());
	t1.print("    LATCH");
}
// ORB 匹配
void ORB_match(vector<LATCH::KeyPoint>& key1, vector<uint64_t>& desc1,
	vector<LATCH::KeyPoint>& key2, vector<uint64_t>& desc2, vector<ORB::DMatch>& dmatch)
{
	dmatch.resize(key1.size());				  // 由key1去找key2
	const float matchDistance = ORB_DIST;	  // 少於多少距離才選定
	const float noMatchDistance = ORB_DISTMAX;// 大於多少距離就不連

	int matchNum = 0;
	for (int j = 0; j < key1.size(); j++) {
		float& distMin = dmatch[j].distance;
		int& matchIdx = dmatch[j].trainIdx;
		// key1[0]去尋找 key2[all] 中最短距離者
		distMin = FLT_MAX;
		for (int i = 0; i < key2.size(); i++) {
			int dist = descDistance(desc2, desc1, j, i);
			if (dist > noMatchDistance)
				continue;
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
// 獲取投影矩陣
vector<double> findHomography(
	vector<LATCH::KeyPoint>& key1, vector<LATCH::KeyPoint>& key2, vector<ORB::DMatch>& dmatch)
{
	vector<double> HomogMat;

	// 獲取有效座標
	vector<cv::Point> fp1, fp2;
	for (int i=0, c=0; i < dmatch.size(); i++) {
		if (dmatch[i].distance != FLT_MAX) {
			int x1 = key1[dmatch[i].queryIdx].x;
			int y1 = key1[dmatch[i].queryIdx].y;
			int x2 = key2[dmatch[i].trainIdx].x;
			int y2 = key2[dmatch[i].trainIdx].y;
			fp1.emplace_back(cv::Point(x1, y1));
			fp2.emplace_back(cv::Point(x2, y2));
		}
	}
	// 隨機抽樣
	vector<uint8_t> RANSAC_mask;
	cv::Mat Hog = cv::findHomography(fp2, fp1, cv::RANSAC, 3, RANSAC_mask, 2000, 0.995);

	// 重置關鍵座標
	long long int x=0, y=0;
	int count=0;
	vector<LATCH::KeyPoint> rankey1, rankey2;
	for (int i = 0; i < RANSAC_mask.size(); i++) {
		if (RANSAC_mask[i] != 0) { // 正確的
			int x1 = fp1[i].x;
			int y1 = fp1[i].y;
			int x2 = fp2[i].x;
			int y2 = fp2[i].y;
			rankey1.emplace_back(LATCH::KeyPoint(x1, y1, 1, -1));
			rankey2.emplace_back(LATCH::KeyPoint(x2, y2, 1, -1));
		}
	}
	key1 = std::move(rankey1);
	key2 = std::move(rankey2);

	// 輸出到 hog
	HomogMat.resize(Hog.cols*Hog.rows);
	for (int i = 0; i < HomogMat.size(); i++)
		HomogMat[i] = Hog.at<double>(i);
	cout << "Hog = " << Hog << endl;

	return HomogMat;
}



// =====================================================================================
/// 焦距與偏差估算

// 輸入 仿射矩陣 獲得焦距
static void focalsFromHomography(const vector<double> &HomogMat, double &f0, double &f1, bool &f0_ok, bool &f1_ok)
{
	const auto& h = HomogMat;

	double d1, d2; // Denominators
	double v1, v2; // Focal squares value candidates

	f1_ok = true;
	d1 = h[6] * h[7];
	d2 = (h[7] - h[6]) * (h[7] + h[6]);
	v1 = -(h[0] * h[1] + h[3] * h[4]) / d1;
	v2 = (h[0] * h[0] + h[3] * h[3] - h[1] * h[1] - h[4] * h[4]) / d2;
	if (v1 < v2) std::swap(v1, v2);
	if (v1 > 0 && v2 > 0) f1 = std::sqrt(std::abs(d1) > std::abs(d2) ? v1 : v2);
	else if (v1 > 0) f1 = std::sqrt(v1);
	else f1_ok = false;

	f0_ok = true;
	d1 = h[0] * h[3] + h[1] * h[4];
	d2 = h[0] * h[0] + h[1] * h[1] - h[3] * h[3] - h[4] * h[4];
	v1 = -h[2] * h[5] / d1;
	v2 = (h[5] * h[5] - h[2] * h[2]) / d2;
	if (v1 < v2) std::swap(v1, v2);
	if (v1 > 0 && v2 > 0) f0 = std::sqrt(std::abs(d1) > std::abs(d2) ? v1 : v2);
	else if (v1 > 0) f0 = std::sqrt(v1);
	else f0_ok = false;
}
// 從矩陣獲得焦距
double estimateFocal(const vector<double> &HomogMat, size_t img1Size, size_t img2Size) {
	const int num_images = 2;
	double median;

	vector<double> all_focals;
	if(!HomogMat.empty()) {
		double f0 ,f1;
		bool f0ok, f1ok;
		focalsFromHomography(HomogMat, f0, f1, f0ok, f1ok);
		if(f0ok && f1ok) {
			double temp = sqrtf(f0 * f1);
			cout << "fff=" << temp << endl;
			all_focals.push_back(sqrtf(f0 * f1));
		}
	}

	if(all_focals.size() >= num_images - 1) {
		std::sort(all_focals.begin(), all_focals.end());
		if(all_focals.size() % 2 == 1) {
			median = all_focals[all_focals.size() / 2];
		} else {
			median = (all_focals[all_focals.size() / 2 - 1] + all_focals[all_focals.size() / 2]) * 0.5f;
		}
	} 

	else {
		//throw out_of_range("123");
		double focals_sum = 0;
		focals_sum += img1Size + img2Size;
		median = focals_sum / num_images;
	}
	cout << "ft = " << median << endl;
	return median;
}
// 估算焦距
double estimateFocal(const vector<double> &HomogMat) {
	double focals = 0;
	if (!HomogMat.empty()) {
		double f0, f1;
		bool f0ok, f1ok;
		focalsFromHomography(HomogMat, f0, f1, f0ok, f1ok);
		if (f0ok && f1ok) focals = std::sqrt(f0 * f1);
	}
	return focals;
}
// 對齊取得第二張圖偏移量
auto getWarpOffset(
	const basic_ImgData &img1, vector<LATCH::KeyPoint> key1,
	const basic_ImgData &img2, vector<LATCH::KeyPoint> key2, 
	float FL)
{
	const int ptSize = key1.size();
	// 中間值.
	const float mid_x1 = (float)img1.width / 2.f;
	const float mid_x2 = (float)img2.width / 2.f;
	const float mid_y1 = (float)img1.height / 2.f;
	const float mid_y2 = (float)img2.height / 2.f;
	// 先算平方.
	const float fL1 = FL;
	const float fL2 = FL;
	const float fL1_pow = powf(fL1, 2.f);
	const float fL2_pow = powf(fL2, 2.f);

	float cal_dx=0, cal_dy=0;

//#pragma omp parallel for
	for (int i = 0; i < ptSize-1; i++) {
		const LATCH::KeyPoint& pt1 = key1[i];
		const LATCH::KeyPoint& pt2 = key2[i];
		const float& imgX1 = pt1.x;
		const float& imgY1 = pt1.y;
		const float& imgX2 = pt2.x;
		const float& imgY2 = pt2.y;

		// 圖1
		float theta1 = fastAtanf_rad((imgX1 - mid_x1) / fL1);
		float h1 = imgY1 - mid_y1;
		h1 /= sqrtf(powf((imgX1 - mid_x1), 2) + fL1_pow);
		float x1 = (fL1*theta1 + mid_x1+.5);
		float y1 = (fL1*h1 + mid_y1+.5);
		// 圖2
		float theta2 = fastAtanf_rad((imgX2 - mid_x2) / fL2);
		float h2 = imgY2 - mid_y2;
		h2 /= sqrtf(powf((imgX2 - mid_x2), 2) + fL2_pow);
		float x2 = (fL2*theta2 + mid_x2 + img1.width +.5);
		float y2 = (fL2*h2 + mid_y2 +.5);
		// 累加座標.
		float distX = x2 - x1;
		float distY = (float)img1.height - y1 + y2;
		cal_dx += distX;
		cal_dy += distY;
	}

	// 平均座標.
	int avg_dx = round(cal_dx / (ptSize-1));
	int avg_dy = round(cal_dy / (ptSize-1));

	// 假如 y 的偏移量大於圖片高
	int xM, yM;
	if(avg_dy > img1.height) {
		int dyy = -((int)img1.height - abs((int)img1.height - avg_dy));
		int xMove = avg_dx;
		int yMove = dyy;
		xM = (int)img1.width - xMove;
		yM = -((int)img1.height) - yMove;
	} else { // 通常情況
		int xMove = avg_dx;
		int yMove = avg_dy;

		xM = (int)img1.width - xMove;
		yM = (int)img1.height - yMove;
	}

	// 輸出座標
	ORB::Point offsetPt = {xM, yM};
	return offsetPt;
}



// =====================================================================================
/// 公開函式

// ORB 獲得投影矩陣
ORB::warpData ORB_Homography(const ImgData& img1, const ImgData& img2) {
	Timer t1, t0;
	t1.priSta = 1;
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
	vector<ORB::DMatch> dmatch;
	t1.start();
	ORB_match(key1, desc1, key2, desc2, dmatch);
	t1.print("  ORB_match");
	cout << endl;


	/********************* 驗證 *************************/
	//keyPt_drawPoint(img1, key1); // 驗證::畫點
	//keyPt_drawPoint(img2, key2); // 驗證::畫點
	//keyPt_drawAngle(img1, key1); // 驗證::畫箭頭
	//keyPt_drawAngle(img2, key2); // 驗證::畫箭頭
	//keyPt_drawMatchLine(img1, key1, img2, key2, dmatch); 
	//keyPt_drawRANSACLine(img1, key1, img2, key2, dmatch);
	/***************************************************/

	
	// 投影矩陣
	vector<double> HomogMat;
	if (dmatch.size() > 3) {
		HomogMat = findHomography(key1, key2, dmatch);// 這裡會改變 key1, key2
	}


	// 估算焦距
	double focals;
	t1.start();
	focals = estimateFocal(HomogMat);
	t1.print("  estimateFocal");
	cout << "focals = " << focals << endl;
	// 估算偏移
	t1.start();
	ORB::Point offsetPt = getWarpOffset(img1, key1, img2, key2, focals);
	t1.print("  getWarpOffset");
	cout << "ofset = " << offsetPt.x << ", " << offsetPt.y << endl;


	ORB::warpData warpdata{HomogMat, offsetPt, focals};
	
	t0.print("ORB::all run time");
	// RANSAC 連線圖
	//keyPt_drawMatchLine(img1, key1, img2, key2, "__RANSACmatchImg.bmp");
	return warpdata;
}

// 測試函式
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
	cout << endl;

	// 匹配描述值
	vector<ORB::DMatch> dmatch;
	t1.start();
	ORB_match(key1, desc1, key2, desc2, dmatch);
	t1.print("  ORB_match");
	cout << endl;
	keyPt_drawMatchLine(img1, key1, img2, key2, dmatch);

	// 投影矩陣
	if (dmatch.size() > 3) {
		vector<double> HomogMat = findHomography(key1, key2, dmatch);
		keyPt_drawRANSACLine(img1, key1, img2, key2, dmatch);
	}
}



// =====================================================================================
void cvInitializeOpenCL() {
	cout << "cvInitializeOpenCL...\n";
	/*const ImgData img1("kanna.bmp");
	vector<LATCH::KeyPoint> key;
	HarrisCroner(key, img1.toConvertGray());*/
	
	using namespace cv;

	Mat cvImg(3, 3, CV_8U);
	UMat ugray = cvImg.getUMat(ACCESS_RW);

	vector<Point2f> corners;
	goodFeaturesToTrack(ugray, corners, 2000, 0.01, 3);
	cout << "cvInitializeOpenCL...done\n\n";
}