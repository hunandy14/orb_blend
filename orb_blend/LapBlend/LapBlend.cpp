﻿/*****************************************************************
Name :
Date : 2018/04/12
By   : CharlotteHonG
Final: 2018/04/12
*****************************************************************/
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <timer.hpp>
using namespace std;

#include "LapBlend.hpp"


//==================================================================================
// 圖片放大縮小
//==================================================================================
// 快速線性插值
inline static void fast_Bilinear_rgb(unsigned char* p, 
	const basic_ImgData& src, double y, double x)
{
	// 起點
	int _x = (int)x;
	int _y = (int)y;
	// 左邊比值
	double l_x = x - (double)_x;
	double r_x = 1.f - l_x;
	double t_y = y - (double)_y;
	double b_y = 1.f - t_y;
	int srcW = src.width;
	int srcH = src.height;

	// 計算RGB
	double R , G, B;
	int x2 = (_x+1) > src.width -1? src.width -1: _x+1;
	int y2 = (_y+1) > src.height-1? src.height-1: _y+1;
	R  = (double)src.raw_img[(_y*srcW + _x)*3 + 0]*(r_x * b_y);
	G  = (double)src.raw_img[(_y*srcW + _x)*3 + 1]*(r_x * b_y);
	B  = (double)src.raw_img[(_y*srcW + _x)*3 + 2]*(r_x * b_y);
	R += (double)src.raw_img[(_y*srcW + x2)*3 + 0]*(l_x * b_y);
	G += (double)src.raw_img[(_y*srcW + x2)*3 + 1]*(l_x * b_y);
	B += (double)src.raw_img[(_y*srcW + x2)*3 + 2]*(l_x * b_y);
	R += (double)src.raw_img[(y2*srcW + _x)*3 + 0]*(r_x * t_y);
	G += (double)src.raw_img[(y2*srcW + _x)*3 + 1]*(r_x * t_y);
	B += (double)src.raw_img[(y2*srcW + _x)*3 + 2]*(r_x * t_y);
	R += (double)src.raw_img[(y2*srcW + x2)*3 + 0]*(l_x * t_y);
	G += (double)src.raw_img[(y2*srcW + x2)*3 + 1]*(l_x * t_y);
	B += (double)src.raw_img[(y2*srcW + x2)*3 + 2]*(l_x * t_y);
	
	p[0] = (unsigned char) R;
	p[1] = (unsigned char) G;
	p[2] = (unsigned char) B;
}
// 快速補值
inline static void fast_NearestNeighbor_rgb(unsigned char* p,
	const basic_ImgData& src, double y, double x) 
{
	// 位置(四捨五入)
	int _x = (int)(x+0.5);
	int _y = (int)(y+0.5);
	int srcW = src.width;
	int srcH = src.height;

	// 計算RGB
	double R , G, B;
	int x2 = (_x+1) > src.width -1? src.width -1: _x+1;
	int y2 = (_y+1) > src.height-1? src.height-1: _y+1;
	R = (double)src.raw_img[(y2*srcW + x2) *3 + 0];
	G = (double)src.raw_img[(y2*srcW + x2) *3 + 1];
	B = (double)src.raw_img[(y2*srcW + x2) *3 + 2];

	p[0] = (unsigned char) R;
	p[1] = (unsigned char) G;
	p[2] = (unsigned char) B;
}
// 圖像縮放
void WarpScale(const basic_ImgData &src, basic_ImgData &dst, double ratio){
	int newH = (int)((src.height * ratio) +0.5);
	int newW = (int)((src.width  * ratio) +0.5);
	// 初始化 dst
	dst.raw_img.resize(newW * newH * src.bits/8.0);
	dst.width  = newW;
	dst.height = newH;
	dst.bits   = src.bits;

	// 縮小的倍率
	double r1W = ((double)src.width )/(dst.width );
	double r1H = ((double)src.height)/(dst.height);
	// 放大的倍率
	double r2W = (src.width -1.0)/(dst.width -1.0);
	double r2H = (src.height-1.0)/(dst.height-1.0);
	// 縮小時候的誤差
	double deviW = ((src.width-1.0)  - (dst.width -1.0)*(r1W)) /dst.width;
	double deviH = ((src.height-1.0) - (dst.height-1.0)*(r1H)) /dst.height;

	// 跑新圖座標
#pragma omp parallel for
	for (int j = 0; j < newH; ++j) {
		for (int i = 0; i < newW; ++i) {
			// 調整對齊
			double srcY, srcX;
			if (ratio < 1.0) {
				srcX = i*(r1W+deviW);
				srcY = j*(r1H+deviH);
			} else if (ratio >= 1.0) {
				srcX = i*r2W;
				srcY = j*r2H;
			}
			// 獲取插補值
			unsigned char* p = &dst.raw_img[(j*newW + i) *3];
			fast_Bilinear_rgb(p, src, srcY, srcX);
		}
	}
}


//==================================================================================
// 模糊圖片
//==================================================================================
// 高斯核心
static vector<double> getGaussianKernel(int n, double sigma)
{
	const int SMALL_GAUSSIAN_SIZE = 7;
	static const float small_gaussian_tab[][SMALL_GAUSSIAN_SIZE] =
	{
		{1.f},
		{0.25f, 0.5f, 0.25f},
		{0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f},
		{0.03125f, 0.109375f, 0.21875f, 0.28125f, 0.21875f, 0.109375f, 0.03125f}
	};

	const float* fixed_kernel = n % 2 == 1 && n <= SMALL_GAUSSIAN_SIZE && sigma <= 0 ?
		small_gaussian_tab[n>>1] : 0;

	vector<double> kernel(n);
	double* cd = kernel.data();

	double sigmaX = sigma > 0 ? sigma : ((n-1)*0.5 - 1)*0.3 + 0.8;
	double scale2X = -0.5/(sigmaX*sigmaX);
	double sum = 0;

	int i;
	for( i = 0; i < n; i++ )
	{
		double x = i - (n-1)*0.5;
		double t = fixed_kernel ? (double)fixed_kernel[i] : std::exp(scale2X*x*x);
		cd[i] = t;
		sum += cd[i];
	} 

	sum = 1./sum;

	for( i = 0; i < n; i++ )
	{
		cd[i] *= sum;
	}

	return kernel;
}
// 高斯模糊
void GaussianBlur(const basic_ImgData& src, basic_ImgData& dst, size_t mat_len, double p=0)
{
	Timer t1;
	size_t width  = src.width;
	size_t height = src.height;

	vector<double> gau_mat = getGaussianKernel(mat_len, p);
	// 初始化 dst
	dst.raw_img.resize(width*height * src.bits/8.0);
	dst.width  = width;
	dst.height = height;
	dst.bits   = src.bits;
	// 緩存
	vector<double> img_gauX(width*height*3);
	// 高斯模糊 X 軸
	const size_t r = gau_mat.size() / 2;
#pragma omp parallel for
	for (int j = 0; j < height; ++j) {
		for (int i = 0; i < width; ++i) {
			double sumR = 0;
			double sumG = 0;
			double sumB = 0;
			for (int k = 0; k < gau_mat.size(); ++k) {
				int idx = i-r + k;
				// idx超出邊緣處理
				if (idx < 0) {
					idx = 0;
				} else if (idx >(int)(width-1)) {
					idx = (width-1);
				}
				sumR += (double)src.raw_img[(j*width + idx)*3 + 0] * gau_mat[k];
				sumG += (double)src.raw_img[(j*width + idx)*3 + 1] * gau_mat[k];
				sumB += (double)src.raw_img[(j*width + idx)*3 + 2] * gau_mat[k];
			}
			img_gauX[(j*width + i)*3 + 0] = sumR;
			img_gauX[(j*width + i)*3 + 1] = sumG;
			img_gauX[(j*width + i)*3 + 2] = sumB;
		}
	}
	// 高斯模糊 Y 軸
#pragma omp parallel for
	for (int j = 0; j < height; ++j) {
		for (int i = 0; i < width; ++i) {
			double sumR = 0;
			double sumG = 0;
			double sumB = 0;
			for (int k = 0; k < gau_mat.size(); ++k) {
				int idx = j-r + k;
				// idx超出邊緣處理
				if (idx < 0) {
					idx = 0;
				} else if (idx > (int)(height-1)) {
					idx = (height-1);
				}
				sumR += img_gauX[(idx*width + i)*3 + 0] * gau_mat[k];
				sumG += img_gauX[(idx*width + i)*3 + 1] * gau_mat[k];
				sumB += img_gauX[(idx*width + i)*3 + 2] * gau_mat[k];

			}
			dst.raw_img[(j*width + i)*3 + 0] = sumR;
			dst.raw_img[(j*width + i)*3 + 1] = sumG;
			dst.raw_img[(j*width + i)*3 + 2] = sumB;
		}
	}
}
// 積分模糊
void Lowpass(const basic_ImgData& src, basic_ImgData& dst) {
	int d=5;
	// 初始化 dst
	dst.raw_img.resize(src.width * src.height* src.bits/8.0);
	dst.width  = src.width;
	dst.height = src.height;
	dst.bits   = src.bits;
	int radius = (d-1) *0.5;
	// 開始跑圖(邊緣扣除半徑)
#pragma omp parallel for
	for (int j = 0; j < src.height; j++) {
		for (int i = 0; i < src.width; i++) {
			int r_t=0, g_t=0, b_t=0;
			// 半徑內平均
			for (int dy = -radius; dy <= radius; dy++) {
				for (int dx = -radius; dx <= radius; dx++) {
					int yy = (j+dy)<0? 0:(j+dy);
					int xx = (i+dx)<0? 0:(i+dx);
					if (yy > dst.height-1) yy = dst.height-1;
					if (xx > dst.width-1) xx = dst.width-1;
					int posi = (yy*src.width + xx)*3;
					r_t += src.raw_img[posi +0];
					g_t += src.raw_img[posi +1];
					b_t += src.raw_img[posi +2];
				}
			}
			dst.raw_img[(j*src.width+i)*3 +0] = r_t /(d*d);
			dst.raw_img[(j*src.width+i)*3 +1] = g_t /(d*d);
			dst.raw_img[(j*src.width+i)*3 +2] = b_t /(d*d);
		}
	}
}
// 高斯矩陣
static vector<double> getGauKer(int x){
	vector<double> kernel(x);
	double half = (x-1) / 2.f;

	constexpr double rlog5_2 = -0.721348; // 1 / (2.f*log(0.5f))
	double sigma = sqrt( -powf(x-1-half, 2.f) * rlog5_2 );
	double rSigma22 = 1.0/(2 * sigma * sigma);

	//#pragma omp parallel for
	for(int i = 0; i < x; i++){
		float g;
		if(i <= (x - half)){
			g = exp( -(i*i*rSigma22) );
		} else{
			g = 1.0 - exp(-powf(x-i-1, 2.f) * rSigma22);
		}
		kernel[i] = g;
	}
	return kernel;
}



//==================================================================================
// 金字塔處理
//==================================================================================
void pyraUp(const basic_ImgData &src, basic_ImgData &dst) {
	int newH = (int)(src.height * 2.0);
	int newW = (int)(src.width  * 2.0);

	// 初始化 dst
	dst.raw_img.resize(newW * newH * src.bits/8.0);
	dst.width  = newW;
	dst.height = newH;
	dst.bits   = src.bits;

	basic_ImgData temp;
	WarpScale(src, temp, 2.0);
	GaussianBlur(temp, dst, 3);
}
void pyraDown(const basic_ImgData &src, basic_ImgData &dst) {
	//Timer t1;
	int newH = (int)(src.height * 0.5);
	int newW = (int)(src.width  * 0.5);

	// 初始化 dst
	dst.raw_img.clear();
	dst.raw_img.resize(newW * newH * src.bits/8.0);
	dst.width  = newW;
	dst.height = newH;
	dst.bits   = src.bits;

	basic_ImgData temp;
	WarpScale(src, temp, 0.5);
	GaussianBlur(temp, dst, 3);
}
void imgSub(basic_ImgData &src, const basic_ImgData &dst) {
	int i, j;
#pragma omp parallel for private(i, j)
	for (j = 0; j < src.height; j++) {
		for (i = 0; i < src.width; i++) {
			int srcIdx = (j*src.width + i) * 3;
			int dstIdx = (j*dst.width + i) * 3;

			int pixR = (int)src.raw_img[srcIdx+0] - (int)dst.raw_img[dstIdx+0] +128;
			int pixG = (int)src.raw_img[srcIdx+1] - (int)dst.raw_img[dstIdx+1] +128;
			int pixB = (int)src.raw_img[srcIdx+2] - (int)dst.raw_img[dstIdx+2] +128;

			pixR = pixR <0? 0: pixR;
			pixG = pixG <0? 0: pixG;
			pixB = pixB <0? 0: pixB;
			pixR = pixR >255? 255: pixR;
			pixG = pixG >255? 255: pixG;
			pixB = pixB >255? 255: pixB;

			src.raw_img[srcIdx+0] = pixR;
			src.raw_img[srcIdx+1] = pixG;
			src.raw_img[srcIdx+2] = pixB;
		}
	}
}
void imgAdd(basic_ImgData &src, const basic_ImgData &dst) {
	int i, j;
#pragma omp parallel for private(i, j)
	for (j = 0; j < src.height; j++) {
		for (i = 0; i < src.width; i++) {
			int srcIdx = (j*src.width + i) * 3;
			int dstIdx = (j*dst.width + i) * 3;

			int pixR = (int)src.raw_img[srcIdx+0] + (int)dst.raw_img[dstIdx+0] -128;
			int pixG = (int)src.raw_img[srcIdx+1] + (int)dst.raw_img[dstIdx+1] -128;
			int pixB = (int)src.raw_img[srcIdx+2] + (int)dst.raw_img[dstIdx+2] -128;

			pixR = pixR <0? 0: pixR;
			pixG = pixG <0? 0: pixG;
			pixB = pixB <0? 0: pixB;
			pixR = pixR >255? 255: pixR;
			pixG = pixG >255? 255: pixG;
			pixB = pixB >255? 255: pixB;

			src.raw_img[srcIdx+0] = pixR;
			src.raw_img[srcIdx+1] = pixG;
			src.raw_img[srcIdx+2] = pixB;
		}
	}
}

// 金字塔
using LapPyr = vector<basic_ImgData>;
void buildPyramids(const basic_ImgData &src, vector<basic_ImgData> &pyr, int octvs=5) {
	pyr.clear();
	pyr.resize(octvs);
	pyr[0]=src;
	for(int i = 1; i < octvs; i++) {
		pyraDown(pyr[i-1], pyr[i]);
	}
}
void buildLaplacianPyramids(const basic_ImgData &src, LapPyr &pyr, int octvs=5) {
	Timer t1;
	t1.priSta=0;
	pyr.clear();
	pyr.resize(octvs);
	pyr[0]=src;

	for(int i = 1; i < octvs; i++) {
		basic_ImgData expend;
		t1.start();
		pyraDown(pyr[i-1], pyr[i]); // 0.6
		t1.print("    pyraDown");
		t1.start();
		WarpScale(pyr[i], expend, 2.0); // 0.5
		t1.print("    WarpScale");
		imgSub(pyr[i-1], expend);
	}
}
void reLaplacianPyramids(LapPyr &pyr, basic_ImgData &dst, int octvs=5) {
	Timer t1;
	int newH = (int)(pyr[0].height);
	int newW = (int)(pyr[0].width);

	// 初始化 dst
	dst.raw_img.clear();
	dst.raw_img.resize(newW * newH * pyr[0].bits/8.0);
	dst.width  = newW;
	dst.height = newH;
	dst.bits   = pyr[0].bits;

	for(int i = octvs-1; i >= 1; i--) {
		basic_ImgData expend;
		WarpScale(pyr[i], expend, 2.0);
		imgAdd(pyr[i-1], expend);
	}
	dst = pyr[0];
}
// 混合拉普拉斯金字塔
void imgBlendHalf(const basic_ImgData& imgA, const basic_ImgData& imgB, basic_ImgData& dst) {
#pragma omp parallel for
	for(int j = 0; j < dst.height; j++) {
		for(int i = 0; i < dst.width; i++) {
			for(int rgb = 0; rgb < 3; rgb++) {
				int dstIdx = (j* dst.width+i)*3;
				int LAIdx  = (j*imgA.width+i)*3;
				int LBIdx  = (j*imgB.width+i)*3;
				int center = dst.width >>1;
				// 拉普拉斯差值區 (左邊就放左邊差值，右邊放右邊差值，正中間放平均)
				if(i == center) {// 正中間
					dst.raw_img[dstIdx +rgb] = (imgA.raw_img[LAIdx +rgb] + imgB.raw_img[LBIdx +rgb]) >>1;
				} else if(i > center) {// 右半部
					dst.raw_img[dstIdx +rgb] = imgB.raw_img[LBIdx +rgb];
				} else { // 左半部
					dst.raw_img[dstIdx +rgb] = imgA.raw_img[LAIdx +rgb];
				}
			}
		}
	}
}
void imgBlendAlpha(const basic_ImgData& imgA, const basic_ImgData& imgB, basic_ImgData& dst) {
	double rat = 1.0 / dst.width;
#pragma omp parallel for
	for(int j = 0; j < dst.height; j++) {
		for(int i = 0; i < dst.width; i++) {
			for(int rgb = 0; rgb < 3; rgb++) {
				int dstIdx = (j* dst.width +i)*3;
				int LAIdx  = (j*imgA.width +i)*3;
				int LBIdx  = (j*imgB.width +i)*3;
				double r1 = rat*i;
				double r2 = 1.0-r1;
				dst.raw_img[dstIdx +rgb] = imgA.raw_img[LAIdx +rgb]*r2 + imgB.raw_img[LBIdx +rgb]*r1;
			}
		}
	}
}
void blendLaplacianPyramids(LapPyr& LS, const LapPyr& LA, const LapPyr& LB) {
	LS.resize(LA.size());
	// 混合圖片
	for(int idx = 0; idx < LS.size(); idx++) {
		// 初始化
		basic_ImgData& dst = LS[idx];
		ImgData_resize(dst, LA[idx].width, LA[idx].height, LA[idx].bits);
		// 開始混合各層
		if(idx == LS.size()-1) {
			imgBlendAlpha(LA[idx], LB[idx], dst);
		} else {
			imgBlendHalf(LA[idx], LB[idx], dst);
		}
	}
}
void blendLaplacianPyramids2(LapPyr& LS, const LapPyr& LA, const LapPyr& LB) {
	LS.resize(LA.size());
	// 高斯矩陣
	auto gausKernal = getGauKer(LA.back().width);
	// 混合圖片
	for(int idx = 0; idx < LS.size(); idx++) {
		// 初始化
		ImgData_resize(LS[idx], LA[idx].width, LA[idx].height, LA[idx].bits);
		// 捷徑
		basic_ImgData& dst = LS[idx];
		const basic_ImgData& imgA = LA[idx];
		const basic_ImgData& imgB = LB[idx];

		// 開始混合各層
#pragma omp parallel for
		for(int j = 0; j < dst.height; j++) {
			for(int i = 0; i < dst.width; i++) {
				for(int rgb = 0; rgb < 3; rgb++) {
					int dstIdx = (j*dst.width + i) * 3;
					int LAIdx = (j*LA[idx].width+i)*3;
					int LBIdx = (j*LB[idx].width+i)*3;
					int center = dst.width >>1;

					if(idx == LS.size()-1) {
						// 拉普拉斯彩色區 (L*高斯) + (R*(1-高斯))
						dst.raw_img[dstIdx +rgb] = 
							LA[idx].raw_img[LAIdx +rgb] * gausKernal[i] +
							LB[idx].raw_img[LBIdx +rgb] * (1.f - gausKernal[i]);
					} else {
						// 拉普拉斯差值區 (左邊就放左邊差值，右邊放右邊差值，正中間放平均)
						if(i == center) {// 正中間
							dst.raw_img[dstIdx +rgb] = (imgA.raw_img[LAIdx +rgb] + imgB.raw_img[LBIdx +rgb]) >>1;
						} else if(i > center) {// 右半部
							dst.raw_img[dstIdx +rgb] = imgB.raw_img[LBIdx +rgb];
						} else { // 左半部
							dst.raw_img[dstIdx +rgb] = imgA.raw_img[LAIdx +rgb];
						}
					}
				}

			}
		}
	}
}
// 混合圖片
void blendLaplacianImg(basic_ImgData& dst, const basic_ImgData& src1, const basic_ImgData& src2) {
	Timer t1;
	t1.priSta=0;
	// 拉普拉斯金字塔 AB
	vector<basic_ImgData> LA, LB;
	t1.start();
	buildLaplacianPyramids(src1, LA);
	t1.print("  buildLapA");
	t1.start();
	buildLaplacianPyramids(src2, LB);
	t1.print("  buildLapB");
	// 混合金字塔
	LapPyr LS;
	t1.start();
	blendLaplacianPyramids(LS, LA, LB);
	t1.print("  blendImg");
	// 還原拉普拉斯金字塔
	t1.start();
	reLaplacianPyramids(LS, dst);
	t1.print("  rebuildLaplacianPyramids");
}



//==================================================================================
// 圓柱投影
//==================================================================================
// 圓柱投影座標反轉換
inline static  void WarpCylindrical_CoorTranfer_Inve(double R,
	size_t width, size_t height, 
	double& x, double& y)
{
	double r2 = (x - width*.5);
	double k = sqrt(R*R + r2*r2) / R;
	x = (x - width *.5)*k + width *.5;
	y = (y - height*.5)*k + height*.5;
}
// 圓柱投影 basic_ImgData
void WarpCylindrical(basic_ImgData &dst, const basic_ImgData &src, 
	double R ,int mx, int my, double edge)
{
	int w = src.width;
	int h = src.height;
	int moveH = (h*edge) + my;
	unsigned int moveW = mx;

	dst.raw_img.clear();
	dst.raw_img.resize((w+moveW)*3 *h *(1+edge*2));
	dst.width = w+moveW;
	dst.height = h * (1+edge*2);

	// 圓柱投影
#pragma omp parallel for
	for (int j = 0; j < h; j++){
		for (int i = 0; i < w; i++){
			double x = i, y = j;
			WarpCylindrical_CoorTranfer_Inve(R, w, h, x, y);
			if (x >= 0 && y >= 0 && x < w - 1 && y < h - 1) {
				unsigned char* p = &dst.raw_img[((j+moveH)*(w+moveW) + (i+moveW)) *3];
				fast_Bilinear_rgb(p, src, y, x);
			}
		}
	}
}
// 找到圓柱投影角點
void WarpCyliCorner(const basic_ImgData &src, vector<int>& corner) {
	corner.resize(4);
	// 左上角角點
	for (int i = 0; i < src.width; i++) {
		int pix = (int)src.raw_img[(src.height/2*src.width +i)*3 +0];
		if (i<src.width/2 and pix != 0) {
			corner[0]=i;
			//cout << "corner=" << corner[0] << endl;
			i=src.width/2;
		} else if (i>src.width/2 and pix == 0) {
			corner[2] = i-1;
			//cout << "corner=" << corner[2] << endl;
			break;
		}
	}
	// 右上角角點
	for (int i = 0; i < src.height; i++) {
		int pix = (int)src.raw_img[(i*src.width +corner[0])*3 +0];
		if (i<src.height/2 and pix != 0) {
			corner[1] = i;
			//cout << "corner=" << corner[2] << endl;
			i=src.height/2;
		} else if (i>src.height/2 and pix == 0) {
			corner[3] = i-1;
			//cout << "corner=" << corner[3] << endl;
			break;
		}
	}
}
// 刪除左右黑邊
void delPillarboxing(const basic_ImgData &src, basic_ImgData &dst,
	vector<int>& corner)
{
	// 新圖大小
	int newH=src.height;
	int newW=corner[2]-corner[0];
	ImgData_resize(dst, newW, newH, 24);
#pragma omp parallel for
	for (int j = 0; j < newH; j++) {
		for (int i = 0; i < newW; i++) {
			for (int  rgb = 0; rgb < 3; rgb++) {
				dst.raw_img[(j*dst.width+i)*3 +rgb] =
					src.raw_img[(j*src.width+(i+corner[0]))*3 +rgb];
			}
		}
	}
	ImgData_write(dst, "delPillarboxing.bmp");
}
// 取出重疊區
void getOverlap(const basic_ImgData &src1, const basic_ImgData &src2,
	basic_ImgData& cut1, basic_ImgData& cut2, vector<int> corner)
{
	// 偏移量
	int mx=corner[4];
	int my=corner[5];
	// 新圖大小
	int newH=corner[3]-corner[1]-abs(my);
	int newW=corner[2]-corner[0]+mx;
	// 重疊區大小
	int lapH=newH;
	int lapW=corner[2]-corner[0]-mx;
	// 兩張圖的高度偏差值
	int myA = my<0? 0:my;
	int myB = my>0? 0:-my;
	// 重疊區
	ImgData_resize(cut1, lapW, lapH, 24);
	ImgData_resize(cut2, lapW, lapH, 24);
#pragma omp parallel for
	for (int j = 0; j < newH; j++) {
		for (int i = 0; i < newW-mx; i++) {
			// 圖1
			if (i < corner[2]-corner[0]-mx) {
				for (int  rgb = 0; rgb < 3; rgb++) {
					cut1.raw_img[(j*cut1.width +i) *3+rgb] = 
						src1.raw_img[(((j+myA)+corner[1])*src1.width +(i+corner[0]+mx)) *3+rgb];
				}
			}
			// 圖2
			if (i >= mx) {
				for (int  rgb = 0; rgb < 3; rgb++) {
					cut2.raw_img[(j*cut2.width +(i-mx)) *3+rgb] = 
						src2.raw_img[(((j+myB)+corner[1])*src1.width +((i-mx)+corner[0])) *3+rgb];
				}
			}
		}
	}
	//ImgData_write(cut1, "__cut1.bmp");
	//ImgData_write(cut2, "__cut2.bmp");
}

void getOverlap_noncut(const basic_ImgData &src1, const basic_ImgData &src2,
	basic_ImgData& cut1, basic_ImgData& cut2, vector<int> corner)
{
	// 偏移量
	int mx=corner[4];
	int my=corner[5];
	// 新圖大小
	int newH=src1.height+abs(my);
	int newW=corner[2]-corner[0]+mx;
	// 重疊區大小
	int lapH=newH;
	int lapW=corner[2]-corner[0]-mx;
	// 兩張圖的高度偏差值
	int myA = my>0? 0:-my;
	int myB = my<0? 0:my;
	// 重疊區
	ImgData_resize(cut1, lapW, lapH, 24);
	ImgData_resize(cut2, lapW, lapH, 24);
#pragma omp parallel for
	for (int j = 0; j < newH; j++) {
		for (int i = 0; i < newW-mx; i++) {
			// 圖1
			if (i < corner[2]-corner[0]-mx and j<src1.height-1) {
				for (int  rgb = 0; rgb < 3; rgb++) {
					cut1.raw_img[((j+myA)*cut1.width +i) *3+rgb]
						= src1.raw_img[((j)*src1.width +(i+corner[0]+mx)) *3+rgb];
				}
			}
			// 圖2
			if (i >= mx and j<src2.height-1) {
				for (int  rgb = 0; rgb < 3; rgb++) {
					cut2.raw_img[((j+myB)*cut2.width +(i-mx)) *3+rgb] = 
						src2.raw_img[((j)*src1.width +((i-mx)+corner[0])) *3+rgb];
				}
			}
		}
	}
	//ImgData_write(cut1, "__cut1.bmp");
	//ImgData_write(cut2, "__cut2.bmp");
}
// 重疊區與兩張原圖合併
void mergeOverlap(const basic_ImgData &src1, const basic_ImgData &src2,
	const basic_ImgData &blend, basic_ImgData &dst, vector<int> corner)
{
	// 偏移量
	int mx=corner[4];
	int my=corner[5];
	// 新圖大小
	int newH=corner[3]-corner[1]-abs(my);
	int newW=corner[2]-corner[0]+mx;
	ImgData_resize(dst, newW, newH, 24);
	// 兩張圖的高度偏差值
	int myA = my<0? 0:my;
	int myB = my>0? 0:-my;

	// 合併圖片
#pragma omp parallel for
	for (int j = 0; j < newH; j++) {
		for (int i = 0; i < newW; i++) {
			// 圖1
			if (i < mx) {
				for (int  rgb = 0; rgb < 3; rgb++) {
					dst.raw_img[(j*dst.width +i) *3+rgb] = src1.raw_img[(((j+myA)+corner[1])*src1.width +(i+corner[0])) *3+rgb];
				}
			}
			// 重疊區
			else if (i >= mx and i < corner[2]-corner[0]) {
				for (int  rgb = 0; rgb < 3; rgb++) {
					dst.raw_img[(j*dst.width +i) *3+rgb] = blend.raw_img[(j*blend.width+(i-mx)) *3+rgb];
				}
			}
			// 圖2
			else if (i >= corner[2]-corner[0]) {
				for (int  rgb = 0; rgb < 3; rgb++) {
					dst.raw_img[(j*dst.width +i) *3+rgb] = src2.raw_img[(((j+myB)+corner[1])*src1.width +((i-mx)+corner[0])) *3+rgb];
				}
			}
		}
	}
}

void mergeOverlap_noncut(const basic_ImgData &src1, const basic_ImgData &src2,
	const basic_ImgData &blend, basic_ImgData &dst, vector<int> corner)
{
	// 偏移量
	int mx=corner[4];
	int my=corner[5];
	// 新圖大小
	int newH=src1.height+abs(my);
	int newW=corner[2]-corner[0]+mx;
	ImgData_resize(dst, newW, newH, 24);
	// 兩張圖的高度偏差值
	int myA = my>0? 0:-my;
	int myB = my<0? 0:my;

	// 合併圖片
#pragma omp parallel for
	for (int j = 0; j < newH; j++) {
		for (int i = 0; i < newW; i++) {
			// 圖1
			if (i < mx and j<src1.height-1) {
				for (int  rgb = 0; rgb < 3; rgb++) {
					dst.raw_img[((j+myA)*dst.width +i) *3+rgb] = 
						src1.raw_img[(((j))*src1.width +(i+corner[0])) *3+rgb];
				}
			}
			// 重疊區
			else if (i >= mx and i < corner[2]-corner[0]) {
				for (int  rgb = 0; rgb < 3; rgb++) {
					dst.raw_img[(j*dst.width +i) *3+rgb] = 
						blend.raw_img[(j*blend.width+(i-mx)) *3+rgb];
				}
			}
			// 圖2
			else if (i >= corner[2]-corner[0] and j<src2.height) {
				for (int  rgb = 0; rgb < 3; rgb++) {
					dst.raw_img[((j+myB)*dst.width +i) *3+rgb] = 
						src2.raw_img[((j)*src1.width +((i-mx)+corner[0])) *3+rgb];
				}
			}
		}
	}
}
// 混合兩張投影過(未裁減)的圓柱，過程會自動裁減輸出
void WarpCyliMuitBlend(basic_ImgData &dst, 
	const basic_ImgData &src1, const basic_ImgData &src2,
	int mx, int my) 
{
	// 檢測圓柱圖角點(minX, minY, maxX, maxY, mx, my)
	vector<int> corner;
	WarpCyliCorner(src1, corner);
	corner.push_back(mx);
	corner.push_back(my);
	
	// 取出重疊區
	basic_ImgData cut1, cut2;
	getOverlap(src1, src2, cut1, cut2, corner);
	// 混合重疊區
	basic_ImgData blend;
	blendLaplacianImg(blend, cut1, cut2);
	// 合併三張圖片
	mergeOverlap(src1, src2, blend, dst, corner);
}



//==================================================================================
// 公開函式
//==================================================================================
// 混合原始圖
void LapBlender(basic_ImgData &dst, 
	const basic_ImgData &src1, const basic_ImgData &src2,
	double ft, int mx, int my)
{
	basic_ImgData warp1, warp2;
	WarpCylindrical(warp1, src1, ft, 0, 0, 0);
	WarpCylindrical(warp2, src2, ft, 0, 0, 0);
	WarpCyliMuitBlend(dst, warp1, warp2, mx, my);
}



// 範例程式
void LapBlend_Tester() {
	basic_ImgData src1, src2, dst;
	string name1, name2;
	double ft; int Ax, Ay;

	// 籃球 (1334x1000, 237ms) 80~100ms
	name1="srcIMG\\ball_01.bmp", name2="srcIMG\\ball_02.bmp"; ft=2252.97, Ax=539, Ay=-37;
	// 校園 (752x500, 68ms)
	//name1="srcIMG\\sc02.bmp", name2="srcIMG\\sc03.bmp"; ft=676.974, Ax=216, Ay=4;

	// 讀取圖片
	ImgData_read(src1, name1);
	ImgData_read(src2, name2);
	// 混合圖片
	Timer t1;
	LapBlender(dst, src1, src2, ft, Ax, Ay);
	t1.print(" LapBlender");
	// 輸出圖片
	ImgData_write(dst, "_WarpCyliMuitBlend.bmp");
}
//==================================================================================
