#include <iostream>
using namespace std;

#include <timer.hpp>
#include "fastlib/fast.h"
#include "OpenBMP/OpenBMP.hpp"


struct basic_FAST12 {
	xy* posi;
	int num;
};
class FAST12: private basic_FAST12 {
public:
	FAST12(const basic_ImgData& grayImg) {
		const unsigned char* data = grayImg.raw_img.data();
		const int xsize = grayImg.width;
		const int ysize = grayImg.height;
		const int stride = xsize;
		const int threshold = 16;
		posi = fast12_detect_nonmax(data, xsize, ysize, xsize, threshold, &num);
	}
public:
	int size() const {
		return num;
	}
	operator const xy*() const {
		return posi;
	}
};

//====================================================================================
int main(int argc, char const *argv[]) {
	Timer t1;
	//const ImgData img("kanna.bmp");
	const ImgData img("lena.bmp");
	ImgData grayImg = img.toConvertGray();
	
	t1.start();
	FAST12 corner(grayImg);
	t1.print();

	ImgData out = img;

	cout << "numcor=" << corner.size() << endl;
	for (size_t i = 0; i < corner.size(); i++) {
		int x = corner[i].x;
		int y = corner[i].y;
		unsigned char* p = out.at2d(y, x);
		p[0] = 255;
		p[1] = 0;
		p[2] = 0;
	}

	out.bmp("out.bmp");
	return 0;
}
//====================================================================================
