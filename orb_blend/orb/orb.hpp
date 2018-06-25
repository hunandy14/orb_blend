#pragma once
#include "OpenBMP.hpp"

vector<double> ORB_Homography(const ImgData & img1, const ImgData & img2);
void ORB_test(const ImgData & img1, const ImgData & img2);
void cvInitializeOpenCL();