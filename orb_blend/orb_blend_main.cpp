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
	using cv::Mat;
	using cv::UMat;


	Timer t1;
	t1.priSta=0;

	// ��l��mask --- >0ms
	Mat cvImg(grayImg.height, grayImg.width, CV_8U, (void*)grayImg.raw_img.data());
	Mat herrisMask(Mat::zeros(cv::Size(grayImg.width, grayImg.height), CV_8U));
	
	// �غc�B�n
	int maskEdg = ORB_DSET_R;
	for(int i = 0; i < corner.num; i++) {
		int x=corner.posi[i].x;
		int y=corner.posi[i].y;
		// �L�o��t��m
		if( x>=(maskEdg) && x<=(int)grayImg.width-(maskEdg) &&
			y>=(maskEdg) && y<=(int)grayImg.height-(maskEdg))
		{
			herrisMask.at<uchar>(cv::Point(x, y)) = 255;
		}
	}

	// Herris ���I
	vector<cv::Point2f> corners;
	UMat ucvImg = cvImg.getUMat(cv::ACCESS_RW);
	//goodFeaturesToTrack(cvImg, corners, 2000, 0.01, 3, herrisMask, 3, true);
	//goodFeaturesToTrack(ucvImg, corners, 445, 0.01, 3, herrisMask, 3, true);
	goodFeaturesToTrack(ucvImg, corners, 445, 0.01, 30, cv::noArray(), 3, true);

	// ��X�� keyPt
	key.clear();
	for (auto&& kp : corners)
		key.emplace_back(kp.x, kp.y, 31.f, -1.f);
}



void FATS_drawPoint(ImgData out, const FAST9& corner, string name="_FSATCorner") {
	using uch = unsigned char;
	auto drawRGB = [](uch* p, uch r, uch g, uch b) {
		p[0] = r, p[1] = g, p[2] = b;
	};
	for (size_t i = 0; i < corner.size(); i++) {
		int x = corner[i].x;
		int y = corner[i].y;
		unsigned char* p = out.at2d(y, x);
		int r=1;
		for (int rj = -r; rj<=r ; rj++) {
			for (int ri = -r; ri<=r; ri++) {
				drawRGB(out.at2d(y+rj, x+ri), 242, 66, 54);
			}
		}
	}
	static int num = 0;
	out.bmp(name+to_string(num++)+".bmp");
}
void keyPt_drawPoint(ImgData out, const vector<LATCH::KeyPoint>& corner,
	string name="_drawKeyPt")
{
	using uch = unsigned char;
	auto drawRGB = [](uch* p, uch r, uch g, uch b) {
		p[0] = r, p[1] = g, p[2] = b;
	};
	for (size_t i = 0; i < corner.size(); i++) {
		const int x = (int)corner[i].x;
		const int y = (int)corner[i].y;

		int r=2;
		for (int rj = -r; rj<=r ; rj++) {
			for (int ri = -r; ri<=r; ri++) {
				drawRGB(out.at2d(y+rj, x+ri), 242, 66, 54);
			}
		}
	}
	static int num = 0;
	out.bmp(name+to_string(num++)+".bmp");
}
void keyPt_drawAngle(ImgData out, const vector<LATCH::KeyPoint>& key, 
	string name="_drawKeyPtAngle") 
{
	using uch = unsigned char;
	for (size_t i = 0; i < key.size(); i++) {
		const int x = (int)key[i].x;
		const int y = (int)key[i].y;
		Draw::draw_arrowRGB(out, key[i].y, key[i].x, 20, key[i].angle*180/3.14);
	}
	static int num = 0;
	out.bmp(name+to_string(num++)+".bmp");
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

				// debug
				if((x < 0.0 or x >= grayImg.width) or
					(y < 0.0 or y >= grayImg.height)) {
					/*cout << "thisx=" << x << endl;
					cout << "thisy=" << y << endl;*/
					continue;
					throw out_of_range("�X�{�t��");
				}

				//m00 +=     grayImg.raw_img[idx];
				m10 += i * grayImg.raw_img[idx];
				m01 += j * grayImg.raw_img[idx];
			}
		}
		//cout << "m01=" << m01 << ", " << "m10::" << m10;
		float a=fastAtan2f(m01, m10);
		//cout << "angle=" << a << endl;
		key[idx++].angle = a * 3.14/180.0;
	}
}
int desc_distance(const vector<uint64_t>& desc1, const vector<uint64_t>& desc2, int idx1, int idx2) {
	int dist=0;
	dist += (bitset<64>(desc1[idx1*8 +0]) ^ bitset<64>(desc2[idx2*8 +0])).count();
	dist += (bitset<64>(desc1[idx1*8 +1]) ^ bitset<64>(desc2[idx2*8 +1])).count();
	dist += (bitset<64>(desc1[idx1*8 +2]) ^ bitset<64>(desc2[idx2*8 +2])).count();
	dist += (bitset<64>(desc1[idx1*8 +3]) ^ bitset<64>(desc2[idx2*8 +3])).count();
	dist += (bitset<64>(desc1[idx1*8 +4]) ^ bitset<64>(desc2[idx2*8 +4])).count();
	dist += (bitset<64>(desc1[idx1*8 +5]) ^ bitset<64>(desc2[idx2*8 +5])).count();
	dist += (bitset<64>(desc1[idx1*8 +6]) ^ bitset<64>(desc2[idx2*8 +6])).count();
	dist += (bitset<64>(desc1[idx1*8 +7]) ^ bitset<64>(desc2[idx2*8 +7])).count();
	return dist;
}

void ORB_dsec(const ImgData& img, vector<LATCH::KeyPoint>& key, vector<uint64_t>& desc) {
	Timer t1;
	t1.priSta = 0;
	const ImgData grayImg = img.toConvertGray();

	// FAST
	t1.start();
	FAST9 corner(grayImg);
	t1.print(" FAST12 Corner");
	//FATS_drawPoint(img, corner); // ����::�e�I

	// Harris
	t1.start();
	keyPt_HarrisCroner(key, grayImg, corner);
	t1.print(" Harris Corner");
	cout << "Herris Corner num = " << key.size() << endl;
	//keyPt_drawPoint(img, key); // ����::�e�I

	// angle
	t1.start();
	keyPt_grayCentroidAngle(key, grayImg);
	t1.print(" keyPt_grayCentroidAngle");
	keyPt_drawAngle(img, key); // ����::�e�b�Y


	// desc
	desc.resize(8 * key.size());
	t1.start();
	LATCH::LATCH<1>(grayImg.raw_img.data(), grayImg.width, grayImg.height, 
		static_cast<int>(grayImg.width), key, desc.data());
	t1.print(" LATCH");
}



class DMatch
{
public:
	DMatch(): queryIdx(-1), trainIdx(-1), imgIdx(-1), distance(FLT_MAX) {}
	DMatch(int _queryIdx, int _trainIdx, float _distance):
		queryIdx(_queryIdx), trainIdx(_trainIdx), imgIdx(-1), distance(_distance) {}

	DMatch(int _queryIdx, int _trainIdx, int _imgIdx, float _distance):
		queryIdx(_queryIdx), trainIdx(_trainIdx), imgIdx(_imgIdx), distance(_distance) {}


	int queryIdx; // query descriptor index
	int trainIdx; // train descriptor index
	int imgIdx;   // train image index

	float distance;

	// less is better
	bool operator < (const DMatch &m) const
	{
		return distance < m.distance;
	}
};
void keyPt_drawMatchLine(
	const ImgData& img1, vector<LATCH::KeyPoint>& key1,  
	const ImgData& img2, vector<LATCH::KeyPoint>& key2,
	vector<DMatch>& dmatch)
{
	using uch = unsigned char;
	// �X�ֹϹ�
	ImgData matchImg(img1.width*2, img1.height, img1.bits);
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
			uch* mp = matchImg.at2d(j, i+img1.width);
			const uch* p2 = img2.at2d(j, i);
			mp[0] = p2[0];
			mp[1] = p2[1];
			mp[2] = p2[2];
		}
	}
	// �s�u
	int count = 0;
	for (size_t i = 0; i < key1.size(); i++) {
		count++;
		if (dmatch[i].distance != FLT_MAX) {
			int x1 = key1[dmatch[i].queryIdx].x;
			int y1 = key1[dmatch[i].queryIdx].y;
			int x2 = key2[dmatch[i].trainIdx].x + img2.width;
			int y2 = key2[dmatch[i].trainIdx].y;
			Draw::drawLineRGB_p(matchImg, y1, x1, y2, x2);
		}
	}
	cout << "Match Num=" << count << endl;
	matchImg.bmp("__matchImg.bmp");
}

void ORB_match(vector<LATCH::KeyPoint>& key1, vector<uint64_t>& desc1,
	vector<LATCH::KeyPoint>& key2, vector<uint64_t>& desc2, vector<DMatch>& dmatch)
{
	dmatch.resize(key1.size());		// 
	const float matchDistance = 64;			// �֩�h�ֶZ���~��w
	const float noMatchDistance = 256;		// �j��h�ֶZ���N���s

	int linkNum = 0;
	for (int j = 1; j < key1.size(); j++) {
		float& distMin = dmatch[j].distance;
		int& matchIdx = dmatch[j].trainIdx;
		// key1[0]�h�M�� key2[all] ���̵u�Z����
		for (int i = 1; i < key2.size(); i++) {
			int dist = desc_distance(desc2, desc1, j, i);
			if (dist > noMatchDistance) continue;
			if (dist < distMin) {
				distMin = dist;
				matchIdx = i;
			}
		}
		// �����̵u�Z���P����
		if (distMin < matchDistance) {
			linkNum++;
			cout << "distMin=" << distMin << endl;
			cout << "desc1 = " << key1[matchIdx].x << ", " << key1[matchIdx].y;
			cout << ", desc2 = " << key2[j].x << ", " << key2[j].y << endl;
			// ������e�̵u�Z����
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
	cout << "link Num = " << linkNum << endl;
}
//====================================================================================
int main(int argc, char const *argv[]) {
	cvInitializeOpenCL();

	Timer t1;
	const ImgData img1("kanna.bmp");
	const ImgData img2("kanna90.bmp");
	//const ImgData img("kanna.bmp");
	//const ImgData img("lena.bmp");


	vector<LATCH::KeyPoint> key1, key2;
	vector<uint64_t> desc1, desc2;
	ORB_dsec(img1, key1, desc1);
	ORB_dsec(img2, key2, desc2);

	cout << "descNum=" << desc1.size()/8 << endl;
	cout << "descNum=" << desc2.size()/8 << endl;
	desc_distance(desc1, desc2, 0, 0);
	// �ǰt�y�z��
	vector<DMatch> dmatch;
	ORB_match(key1, desc1, key2, desc2, dmatch);
	keyPt_drawMatchLine(img1, key1, img2, key2, dmatch);

	return 0;
}
//====================================================================================
