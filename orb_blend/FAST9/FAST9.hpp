/*****************************************************************
Name :
Date : 2018/06/25
By   : CharlotteHonG
Final: 2018/06/25
*****************************************************************/
#pragma once
#include "OpenBMP.hpp"

extern "C" {
#include "fastlib/fast.h"
}

class FAST9 {
public:
	FAST9() = default;
	FAST9(const basic_ImgData& grayImg) {
		const unsigned char* data = grayImg.raw_img.data();
		const int xsize = grayImg.width;
		const int ysize = grayImg.height;
		const int stride = xsize;
		const int threshold = 16;
		pt = fast9_detect_nonmax(data, xsize, ysize, xsize, threshold, &num);
	}
	~FAST9() {
		if (pt) {
			free(pt);
			num = 0;
		}
	}
	FAST9(const FAST9& o) = delete;
	FAST9(FAST9&& o) noexcept = delete; 
	FAST9& operator=(const FAST9& other)=delete;
	FAST9& operator=(FAST9&& other) noexcept = delete;
public:
	void info_print() const {
		cout << "FAST num = " << num << endl;
	}
	int size() const {
		return num;
	}
	operator const xy*() const {
		return pt;
	}
	operator xy*() {
		return pt;
	}
public:
	xy* pt = nullptr;
	int num = 0;
};