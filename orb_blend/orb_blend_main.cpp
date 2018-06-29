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
	/*======================測資==========================*/
	//name1 = "kanna.bmp", name2 = "kanna90.bmp"; // 90度測試
	//name1 = "ball_01.bmp", name2 = "ball_02.bmp";
	//name1 = "sc02.bmp", name2 = "sc03.bmp";
	//name1 = "data/DSC_2936.bmp", name2 = "data/DSC_2937.bmp";
	//name1 = "data/DSC_2938.bmp", name2 = "data/DSC_2939.bmp";
	//name1 = "data/DSC_2940.bmp", name2 = "data/DSC_2941.bmp";
	//name1 = "data/DSC_2942.bmp", name2 = "data/DSC_2943.bmp";
	//name1 = "data/DSC_2944.bmp", name2 = "data/DSC_2945.bmp";
	name1 = "data/DSC_2946.bmp", name2 = "data/DSC_2947.bmp";
	name1 = "data/DSC_2952.bmp", name2 = "data/DSC_2953.bmp";
	name1 = "data/DSC_2956.bmp", name2 = "data/DSC_2957.bmp";
	name1 = "data/DSC_2958.bmp", name2 = "data/DSC_2959.bmp";
	ImgData img1(name1), img2(name2);
	/*===================================================*/
	Timer t1, t0;
	ImgData blend;
	ORB::warpData w;
	
	
	t0.start();
	// 獲取投影矩陣 與 圓柱投影資訊
	w = ORB_Homography(img1, img2);
	// 混合圖像
	t1.start();
	LapBlender(blend, img1, img2, w.ft, w.pt.x, w.pt.y);
	t1.print("  LapBlender");
	t0.print("@ ALL TIME");

	blend.bmp("__blend.bmp");


	return 0;
}
