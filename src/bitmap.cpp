#include "bitmap.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BMP_SIZE 14
#define DIB_SIZE 40
#define N_COLOURS 3

#define CHR_INSERT(B, O, C) \
    (B)[(O) + 0] = ((C) >>  0) & 0xFF;

#define SHT_INSERT(B, O, S) \
    (B)[(O) + 0] = ((S) >>  0) & 0xFF; \
    (B)[(O) + 1] = ((S) >>  8) & 0xFF;

#define INT_INSERT(B, O, I) \
    (B)[(O) + 0] = ((I) >>  0) & 0xFF; \
    (B)[(O) + 1] = ((I) >>  8) & 0xFF; \
    (B)[(O) + 2] = ((I) >> 16) & 0xFF; \
    (B)[(O) + 3] = ((I) >> 24) & 0xFF;

#define CHR_READ(B, O) \
    ((B)[(O) + 0] <<  0)

#define SHT_READ(B, O) \
    (((B)[(O) + 0] <<  0) | \
     ((B)[(O) + 1] <<  8))

#define INT_READ(B, O) \
    (((B)[(O) + 0] <<  0) | \
     ((B)[(O) + 1] <<  8) | \
     ((B)[(O) + 2] << 16) | \
     ((B)[(O) + 3] << 24))

#define PAD(I, R) (((I) + (R) - 1) & ~((R) - 1))

#define ASSERT(C) \
    if (!(C)) { \
        printf("Assert Failed!\n"); \
        return EXIT_FAILURE; \
    }

int importBMP(const char* filepath, Image& img) {
    uint8_t bmpHeader[BMP_SIZE];
    uint8_t dibHeader[DIB_SIZE];
    FILE* f;

    // Open File
    f = fopen(filepath, "rb");
    if (f == NULL) {
        printf("Failed to open %s!\n", filepath);
        return EXIT_FAILURE;
    }

    /* Bitmap File Header */
    if (fread(bmpHeader, BMP_SIZE, 1, f) != 1) {
        printf("Failed to read BMP header!\n");
        fclose(f);
        return EXIT_FAILURE;
    }

    ASSERT(CHR_READ(bmpHeader,  0) == 'B');                 // Header 'B'
    ASSERT(CHR_READ(bmpHeader,  1) == 'M');                 // Header 'M'
    ASSERT(INT_READ(bmpHeader, 10) == BMP_SIZE + DIB_SIZE); // Data Offset

    img.label = INT_READ(bmpHeader, 6); // Label
    
    /* Device Independent Bitmap Header */
    if (fread(dibHeader, DIB_SIZE, 1, f) != 1) {
        printf("Failed to read DIB header!\n");
        fclose(f);
        return EXIT_FAILURE;
    }

    ASSERT(INT_READ(dibHeader, 0) == DIB_SIZE);                 // DIB Header Size
    ASSERT(SHT_READ(dibHeader, 12) == 1);                       // Colour Palettes
    ASSERT(SHT_READ(dibHeader, 14) == N_COLOURS * CHAR_BIT);    // Bits Per Pixel
    ASSERT(INT_READ(dibHeader, 16) == 0);                       // Compression Mode

    img.W = INT_READ(dibHeader, 4); // Bitmap width
    img.H = INT_READ(dibHeader, 8); // Bitmap height

    /* Convert .bmp to Image */
    int buffW = PAD(N_COLOURS * img.W, 4);
    int buffH = img.H;
    img.data = new uint8_t[N_COLOURS * img.W * img.H];

    // Allocate buffer
    uint8_t* buffer = new uint8_t[buffW * buffH];

    // Read data to buffer
    if (fread(buffer, buffW * buffH, 1, f) != 1) {
        printf("Failed to read data!\n");
        fclose(f);
        return EXIT_FAILURE;
    }

    // Remove padding
    for (int i = 0; i < buffH; i++) {
        memcpy(&img.data[i * N_COLOURS * img.W], &buffer[i * buffW], N_COLOURS * img.W);
    }

    delete buffer;
    fclose(f);

    return EXIT_SUCCESS;
}

int exportBMP(const char* filepath, Image& img) {
    uint8_t bmpHeader[BMP_SIZE];
    uint8_t dibHeader[DIB_SIZE];
    FILE* f;

    // Open File
    f = fopen(filepath, "wb");
    if (f == NULL) {
        printf("Failed to open %s!\n", filepath);
        return EXIT_FAILURE;
    }

    /* Convert Image to .bmp */
    int buffW = PAD(N_COLOURS * img.W, 4);
    int buffH = img.H; 

    // Allocate Buffer
    uint8_t* buffer = new uint8_t[buffW * buffH];

    // Add Padding
    for (int i = 0; i < img.H; i++) {
        memcpy(&buffer[i * buffW], &img.data[i * N_COLOURS * img.W], N_COLOURS * img.W);
    }

    /* Bitmap File Header */
    CHR_INSERT(bmpHeader,  0, 'B');                                 // Header 'B'
    CHR_INSERT(bmpHeader,  1, 'M');                                 // Header 'M'
    INT_INSERT(bmpHeader,  2, BMP_SIZE + DIB_SIZE + buffW * buffH); // File size
    INT_INSERT(bmpHeader,  6, img.label);                           // Reserved 
    INT_INSERT(bmpHeader, 10, BMP_SIZE + DIB_SIZE);                 // Data Offset

    /* Device Independent Bitmap Header */
    INT_INSERT(dibHeader,  0, DIB_SIZE);                // Header Size
    INT_INSERT(dibHeader,  4, img.W);                   // Bitmap Width
    INT_INSERT(dibHeader,  8, img.H);                   // Bitmap Height
    SHT_INSERT(dibHeader, 12, 1);                       // Number of Colour Palettes
    SHT_INSERT(dibHeader, 14, N_COLOURS * CHAR_BIT);    // Bits Per Pixel
    INT_INSERT(dibHeader, 16, 0);                       // Compression Mode
    INT_INSERT(dibHeader, 20, buffW * buffH);           // Image Size 
    INT_INSERT(dibHeader, 24, 0);                       // Horizontal Resolution
    INT_INSERT(dibHeader, 28, 0);                       // Vertical Resolution
    INT_INSERT(dibHeader, 32, 0);                       // Number of Colours in Colour Palette
    INT_INSERT(dibHeader, 36, 0);                       // Number of Important Colours

    /* Write Data To File */
    if (fwrite(bmpHeader, BMP_SIZE, 1, f) != 1) {
        printf("Failed to write BMP header!\n");
        return EXIT_FAILURE;
    }

    if (fwrite(dibHeader, DIB_SIZE, 1, f) != 1) {
        printf("Failed to write DIP header!\n");
        return EXIT_FAILURE; 
    }

    if (fwrite(buffer, buffW * buffH, 1, f) != 1) {
        printf("Failed to write data!\n");
        return EXIT_FAILURE; 
    }

    delete buffer;
    fclose(f);

    return EXIT_SUCCESS;
}
