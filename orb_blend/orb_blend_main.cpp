#include <iostream>
#include <string>
using namespace std;

#include "orb/orb.hpp"
#include "LapBlend/LapBlend.hpp"
#include "Timer.hpp"

//====================================================================================
int main(int argc, char const *argv[]) {
	cvInitializeOpenCL();

	string name1, name2;
	/*======================����==========================*/
	//name1 = "kanna.bmp", name2 = "kanna90.bmp"; // 90�״���
	//name1 = "ball_01.bmp", name2 = "ball_02.bmp";
	name1 = "sc02.bmp", name2 = "sc03.bmp";
	ImgData img1(name1), img2(name2);
	/*===================================================*/
	Timer t;
	ImgData blend;
	ORB::warpData w;
	
	
	t.start();
	// �����v�x�} �P ��W��v��T
	w = ORB_Homography(img1, img2);
	// �V�X�Ϲ�
	LapBlender(blend, img1, img2, w.ft, w.pt.x, w.pt.y-2);
	t.print("__blend");


	blend.bmp("__blend.bmp");


	return 0;
}
