#include <time.h>
#include "NN.h"
#include "Dataloader.h"
/**
 * Load the mnist data into the model 
 * - for training 
 * Options: 
 * - python data loader (done)
 * - Generic binary loader 
 * - generic loader that can handle binary and csv
 * 
 */


// Mat input_data_mnist(int img_number) {

// }



int main(void) {
    srand(time(0));
    
    //print_model(*nn);
    /*
    0 1 0 1
    0 0 1 1
    */
//    Mat x = MatInit(2,4);
//    MAT_AT(x, 0, 0) = 0;
//    MAT_AT(x, 0, 1) = 0;
//    MAT_AT(x, 0, 2) = 1;
//    MAT_AT(x, 0, 3) = 1;
   
//     MAT_AT(x, 1, 0) = 1;
//     MAT_AT(x, 1, 1) = 0;
//     MAT_AT(x, 1, 2) = 1;
//     MAT_AT(x, 1, 3) = 0;
    
//     Mat y = MatInit(1,4);
//     MAT_AT(y, 0, 0) = 1;
//     MAT_AT(y, 0, 1) = 0;
//     MAT_AT(y, 0, 2) = 0;
//     MAT_AT(y, 0, 3) = 1;
    // for (size_t j = 0; j < 4; j++){
        
    //     int p = j%2;// 1 0 1 0 
    //     MAT_AT(x, 0, j) = p;
    // }
    
    // for (size_t j = 0; j < 2; j++){
        //     int p = j%2;//  0 0 1 1 
        //     MAT_AT(x, 1, j) = p;
        // }
    // float eta = 0.01;
    // int epoch = 10000;

    // int architecture[] = {2,3,4,5,4,3,1};
    // Activation act[] =   {1,1,1,1,1,1,0};
    // int alen = sizeof(architecture)/sizeof(architecture[0]);
    // Function f = {.cost = BINARY_CROSS_ENTROPY, .act = SIGMOID};
    // NN *nn = make_model(architecture, alen, act, BINARY_CROSS_ENTROPY);
    // train_mlp_sgd(x,*nn, y, eta, epoch, 0);

    
    // print_model(*nn);

    // Mat u = MatInit(2,1);
    // MAT_AT(u, 0, 0) = 0;
    // MAT_AT(u, 1, 0) = 1;
    // Mat test = forward(u, *nn);
    // MAT_PRINT(test);
    
    ImgData *img = load_mnist();
    int img_num = 5;
    int label = img->labels[img_num];
    Mat x = MatInit(img->rows, img->cols);
    
    colourNormaliseArray(x, img, img_num); 
    MatPrintNum(x, label);
    
    Mat xin = MatInit(img->rows*img->cols, 1);
    MatFlat(xin, x);
    MatPrintNum(xin, label);
    show_image(img, img_num);

    Mat yout = labels_to_onehot(img);

    // y->labels, x->input
    // uint8_t *y = img->labels;
    
    float eta = 0.01;
    int epoch = 10000;
    // for mnist-> 28x28-> 
    int architecture[] = {784,128,64,10};
    Activation act[] =   {1,1,1,SOFTMAX};
    int alen = sizeof(architecture)/sizeof(architecture[0]);
    NN *nn = make_model(architecture, alen, act, MCLASS_CROSS_ENTROPY);
    train_mlp_sgd(xin,*nn, yout, eta, epoch, 0);


    return 0;
}
