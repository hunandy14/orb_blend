#ifndef FAST_H
#define FAST_H


typedef struct { int x, y; } xy; 
typedef unsigned char byte;

int fast9_corner_score(const byte* p, const int pixel[], int bstart);
int fast10_corner_score(const byte* p, const int pixel[], int bstart);
int fast11_corner_score(const byte* p, const int pixel[], int bstart);
int fast12_corner_score(const byte* p, const int pixel[], int bstart);

xy* fast9_detect(const byte* im, int xsize, int ysize, int stride, int b, int* ret_num_corners);
xy* fast10_detect(const byte* im, int xsize, int ysize, int stride, int b, int* ret_num_corners);
xy* fast11_detect(const byte* im, int xsize, int ysize, int stride, int b, int* ret_num_corners);
xy* fast12_detect(const byte* im, int xsize, int ysize, int stride, int b, int* ret_num_corners);

int* fast9_score(const byte* i, int stride, xy* corners, int num_corners, int b);
int* fast10_score(const byte* i, int stride, xy* corners, int num_corners, int b);
int* fast11_score(const byte* i, int stride, xy* corners, int num_corners, int b);
int* fast12_score(const byte* i, int stride, xy* corners, int num_corners, int b);


xy* fast9_detect_nonmax(const byte* im, int xsize, int ysize, int stride, int b, int* ret_num_corners);
xy* fast10_detect_nonmax(const byte* im, int xsize, int ysize, int stride, int b, int* ret_num_corners);
xy* fast11_detect_nonmax(const byte* im, int xsize, int ysize, int stride, int b, int* ret_num_corners);
xy* fast12_detect_nonmax(const byte* im, int xsize, int ysize, int stride, int b, int* ret_num_corners);

xy* nonmax_suppression(const xy* corners, const int* scores, int num_corners, int* ret_num_nonmax);

inline void make_offsets(int pixel[], int row_stride)
{
        pixel[0] = 0 + row_stride * 3;
        pixel[1] = 1 + row_stride * 3;
        pixel[2] = 2 + row_stride * 2;
        pixel[3] = 3 + row_stride * 1;
        pixel[4] = 3 + row_stride * 0;
        pixel[5] = 3 + row_stride * -1;
        pixel[6] = 2 + row_stride * -2;
        pixel[7] = 1 + row_stride * -3;
        pixel[8] = 0 + row_stride * -3;
        pixel[9] = -1 + row_stride * -3;
        pixel[10] = -2 + row_stride * -2;
        pixel[11] = -3 + row_stride * -1;
        pixel[12] = -3 + row_stride * 0;
        pixel[13] = -3 + row_stride * 1;
        pixel[14] = -2 + row_stride * 2;
        pixel[15] = -1 + row_stride * 3;
}


#endif // FAST_H