#pragma once
#include "OpenBMP.hpp"

// =====================================================================================
/// ¸ê®Æµ²ºc

namespace ORB {

struct Point {
	float x;
	float y;
};

struct DMatch{
	int queryIdx;
	int trainIdx;
	float distance;
};

struct warpData {
	vector<double> HomogMat;
	Point pt;
	double ft;
};

}; // namespace ORB



 // =====================================================================================

ORB::warpData ORB_Homography(const ImgData & img1, const ImgData & img2);
void ORB_test(const ImgData & img1, const ImgData & img2);
void cvInitializeOpenCL();