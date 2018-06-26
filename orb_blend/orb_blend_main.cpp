#include <iostream>
#include <string>
using namespace std;

#include "orb/orb.hpp"



//====================================================================================
int main(int argc, char const *argv[]) {
	cvInitializeOpenCL();

	string name1, name2;
	/*======================測資==========================*/
	//name1 = "kanna.bmp", name2 = "kanna90.bmp"; // 90度測試
	name1 = "ball_01.bmp", name2 = "ball_02.bmp";
	//name1 = "sc02.bmp", name2 = "sc03.bmp";
	const ImgData img1(name1), img2(name2);
	/*===================================================*/

	// 獲取投影矩陣 與 圓柱投影資訊
	ORB::warpData w = ORB_Homography(img1, img2);




	return 0;
}
