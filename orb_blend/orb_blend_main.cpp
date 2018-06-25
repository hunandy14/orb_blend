#include <iostream>
#include <string>
using namespace std;

#include "orb/orb.hpp"
#include "FAST9/FAST9.hpp"

//====================================================================================
int main(int argc, char const *argv[]) {
	cvInitializeOpenCL();

	string name1, name2;
	/*======================����==========================*/
	//name1 = "kanna.bmp", name2 = "kanna90.bmp"; // 90�״���
	//name1 = "ball_01.bmp", name2 = "ball_02.bmp";
	name1 = "sc02.bmp", name2 = "sc03.bmp";
	/*===================================================*/

	const ImgData img1(name1), img2(name2);
	ORB_test(img1, img2);

	ImgData gray = img1.toConvertGray();
	FAST9 fast(gray);

	return 0;
}
