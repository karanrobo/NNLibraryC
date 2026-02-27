#ifndef DL_H
#define DL_H
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <raylib.h>
//#include "visuals/Ui.h"
#include "NN.h"
// for loading 3d tensor for the images
typedef struct {
    uint8_t *images; 
    uint8_t *labels;
    int num_images;
    int rows;
    int cols;
}ImgData;


 
ImgData *load_mnist();
// for viewing 
void colourNormaliseArray(Mat m, ImgData *data); 

void MatPrintNum(Mat m, int label);

void show_image(ImgData *data, int img);

void labels_to_onehot(Mat yout, ImgData *data);

// make a 28x28 for testing
void draw_image();

#endif