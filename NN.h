#ifndef NN_H
#define NN_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

#define MAT_PRINT(m) MatPrint(m, #m)
#define MAT_AT(m, i, j) (m)->es[(i)*(m)->cols + j] // put in parenthesis for macros

typedef struct mat *Mat;
struct mat {
    size_t rows;
    size_t cols;
    float *es;    
};


// NN models
typedef enum {
    SIGMOID,
    RELU,
    LEAKY_RELU,
    SOFTMAX,
} Activation;

typedef enum {
    MSE,
    BINARY_CROSS_ENTROPY,
    MCLASS_CROSS_ENTROPY,
}Cost;

typedef struct {
    Activation act;
    Cost cost;
} Function;




// For N layers, 0th and (N-1) are input and output respectively
typedef struct{
    // array of matrices for weights(n x m)
    Mat *W;
    
    // array of matrices for biases(n x m)
    // only for hidden neurons 1 to n - 1
    Mat *b;
    
    // W a + b, n x 1
    Mat *pre_activation;   
    
    // sigma(W a + b)  n x 1, also stores the input
    Mat *post_activation;

    // for back propagation
    Mat *delta;
    
    // store layer data at each layer, including input and output 
    int *layer; 
    int layers;

    // Activation function 
    Activation *activations; // define activation per layer
    // Loss function 
    Cost cost;
} NN;


Mat MatInit(size_t rows, size_t cols);
void MatFree(Mat m);
void MatScalar(Mat m, float lambda);

void MatRand(Mat m, float lo, float hi);

// Best for Non linear activations, sigmoid, tanh
void MatRandXavier(Mat m, int input_size);

// Best for ReLU, Leaky ReLU
void MatRandHe(Mat m, int fan_in);

float RandFloat(void);


void MatMul(Mat dest, Mat a, Mat b);
void MatSum(Mat dest, Mat a);
void MatSub(Mat dest, Mat a);
void MatCopy(Mat dest, Mat a);
void MatPrint(Mat a, const char *c);
void Transpose(Mat dest, Mat m);
void Hadamad(Mat dest, Mat a, Mat b);
void GetCol(Mat dest, Mat a, int col);
void MatFlat(Mat dest, Mat src);

// Activations 
float sigmoidf(float x);
float sigmoidfDiff(float x);
float ReLU(float x);
float ReLUDiff(float x);
float LeakyReLU(float x);
float LeakyReLUDiff(float x);


// Loss MSE = { mse_loss, mse_grad };
// Loss L1  = { l1_loss, l1_grad };
// Loss HUBER = { huber_loss, huber_grad };
// Loss CE = { ce_loss, ce_grad };
// Loss HINGE = { hinge_loss, hinge_grad };
// Loss LOGCOSH = { logcosh_loss, logcosh_grad };

float mse_cost(float y, float y_hat);
float mse_cost_diff(float y, float y_hat);


float binary_cross_entropy_cost(float y, float y_hat);
float binary_cross_entropy_cost_diff(float y, float y_hat);


float mclass_cross_entropy_cost(NN nn, Mat y);

void soft_max_output_layer(NN nn);


// float hinge_cost(float y, float y_hat);
// float hinge_cost_diff(float y, float y_hat);

void MatAct(NN nn, Mat dest, Mat m, int layer);
void MatActDiff(NN nn, Mat dest, Mat m, int layer);



NN *make_model(int *nodes, int layer_count, Activation *act, Cost cost);
Mat *INIT_TENSOR_W(int *nodes, int layer_count, Activation *act);
Mat *INIT_ACTIVATION(int *nodes, int layer_count);
Mat *INIT_DELTA(int *nodes, int layer_count);
Mat *MCES_INIT_B(int *nodes, int layer_count);

Mat forward(Mat input, NN nn);
void backprop(NN nn, Mat y, float eta);
float cost(NN nn, Mat y);
void train_mlp_sgd(Mat input, NN nn, Mat y, float eta, int epoch, bool print_error);
void print_model(NN model);


#endif