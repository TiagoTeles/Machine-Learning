#ifndef BITMAP_H
#define BITMAP_H

#include <stdint.h>

struct Image {
    uint8_t* data;
    int label;
    int W;
    int H;
};

int importBMP(const char* filepath, Image& img);
int exportBMP(const char* filepath, Image& img);

#endif