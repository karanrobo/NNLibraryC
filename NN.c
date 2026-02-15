#include "NN.h"
#include <assert.h>
#include <math.h>
#include <omp.h>


// Change this for Leaky ReLU
#define alpha -0.01

Mat MatInit(size_t rows, size_t cols) {
    Mat m = malloc(sizeof(struct mat));
    m->rows = rows;
    m->cols = cols;
    m->es = calloc(rows * cols, sizeof(*m->es));
    assert(m->es != NULL);
    return m;
}

void MatFree(Mat m) {
    if (m == NULL) {
        return;
    }
    free(m->es);
    free(m);
    return;
}

// Wx, dest = a x b
void MatMul(Mat dest, Mat a, Mat b) {
    assert(a->cols == b->rows);
    size_t n = a->cols;
    assert(dest->rows == a->rows);
    assert(dest->cols == b->cols);
    size_t i, j, k;
    
    // Only parallelize for larger matrices (threshold: 1000+ total elements)
    // #pragma omp parallel for private(i, j, k) if(a->rows * b->cols > 1000)
    for (i = 0; i < a->rows; i++)
    {
        for (j = 0; j < b->cols; j++)
        {
            MAT_AT(dest, i, j) = 0; //already calloced
            for (k = 0; k < n; k++)
            {
                MAT_AT(dest, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
        }
    }
    return;
}


void MatScalar(Mat m, float lambda) {
    size_t i, j;
    // #pragma omp parallel for private(i, j) if(m->rows * m->cols > 1000)
    for (i = 0; i < m->rows; i++){
        for (j = 0; j < m->cols; j++){
            MAT_AT(m, i, j) = MAT_AT(m, i, j) * lambda;
        }
    }
}

float RandFloat(void) {
    return (float) rand() / (float) RAND_MAX;
}

void MatRand(Mat m, float lo, float hi) {
    size_t i, j;
    // #pragma omp parallel for private(i, j) if(m->rows * m->cols > 1000)
    for (i = 0; i < m->rows; i++){
        for (j = 0; j < m->cols; j++){
            MAT_AT(m, i, j) = RandFloat() * (hi - lo) + lo;
        }
    }
}

// Xavier initialization for weights
void MatRandXavier(Mat m, int input_size) {
    float limit = sqrtf(6.0f / (input_size + m->rows));
    size_t i, j;
    for (i = 0; i < m->rows; i++){
        for (j = 0; j < m->cols; j++){
            MAT_AT(m, i, j) = (RandFloat() * 2.0f - 1.0f) * limit;
        }
    }
}


void MatRandHe(Mat m, int fan_in) {
    float limit = sqrtf(6.0f / (fan_in));
    size_t i, j;
    for (i = 0; i < m->rows; i++){
        for (j = 0; j < m->cols; j++){
            MAT_AT(m, i, j) = (RandFloat() * 2.0f - 1.0f) * limit;
        }
    }
}


void MatSum(Mat dest, Mat m) {
    assert(dest->rows == m->rows);
    assert(dest->cols == m->cols);
    size_t i, j;
    // #pragma omp parallel for private(i, j) if(m->rows * m->cols > 1000)
    for (i = 0; i < dest->rows; i++)
    {
        for (j = 0; j < dest->cols; j++){
            MAT_AT(dest, i, j) += MAT_AT(m, i, j);
        }
    }
}

void MatSub(Mat dest, Mat m) {
    assert(dest->rows == m->rows);
    assert(dest->cols == m->cols);
    size_t i, j;
    // #pragma omp parallel for private(i, j) if(m->rows * m->cols > 1000)
    for (i = 0; i < dest->rows; i++)
    {
        for (j = 0; j < dest->cols; j++){
            MAT_AT(dest, i, j) -= MAT_AT(m, i, j);
        }
    }
}


void MatCopy(Mat dest, Mat m) {
    assert(dest->rows == m->rows);
    assert(dest->cols == m->cols);
    size_t i, j;
    // #pragma omp parallel for private(i, j) if(m->rows * m->cols > 1000)
    for (i = 0; i < dest->rows; i++)
    {
        for (j = 0; j < dest->cols; j++){
            MAT_AT(dest, i, j) = MAT_AT(m, i, j);
        }
    }
}

void MatPrint(Mat m, const char *c) {
    printf("%s = [\n", c);
    for (size_t i = 0; i < m->rows; i++)
    {
        for (size_t j = 0; j < m->cols; j++){
            printf("    %f ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("]\n");
}

// different activation fucntions have different f'(x)
float sigmoidf(float x) {
    return 1.0 / (1.0 + expf(-x));  
}

float ReLU(float x) {
    return (x > 0 ? x : 0);
}
float ReLUDiff(float x) {
    return (x > 0 ? 1 : 0); 
}
float LeakyReLU(float x) {
    return (x > 0 ? x : alpha * x);
}
float LeakyReLUDiff(float x) {
    return (x > 0 ? 1 : alpha);
}

// assumes post activation
float sigmoidfDiff(float x) {
    // float x = sigmoidf(x);
    return x * (1 - x);
}


void MatAct(Mat dest, Mat m, Activation act) {
    assert(dest->rows == m->rows);
    assert(dest->cols == m->cols);
    size_t i,j;
    // #pragma omp parallel for private(i, j) if(m->rows * m->cols > 1000)
    for (i = 0; i < m->rows; i++)
    {
        for (j = 0; j < m->cols; j++)
        {

            MAT_AT(dest,i,j) = act == RELU ? ReLU(MAT_AT(m,i,j)) : 
                            act == SIGMOID ? sigmoidf(MAT_AT(m,i,j)) : 
                            LeakyReLU(MAT_AT(m,i,j));
        }
      
    }
}

// for sig(m)
void MatActDiff(Mat dest, Mat m, Activation act) {
    assert(dest->rows == m->rows);
    assert(dest->cols == m->cols);
    size_t i, j;
    // #pragma omp parallel for private(i, j) if(m->rows * m->cols > 1000)
    for (i = 0; i < m->rows; i++)
    {
        for (j = 0; j < m->cols; j++)
        {
            //float val = MAT_AT(m,i,j);
            //MAT_AT(dest,i,j) = val*(1-val);
            MAT_AT(dest, i, j) = act == RELU ? ReLUDiff(MAT_AT(m,i,j)) : 
                            act == SIGMOID ? sigmoidfDiff(MAT_AT(m,i,j)) : 
                            LeakyReLUDiff(MAT_AT(m,i,j));
        }
      
    }
}


void Transpose(Mat dest, Mat m) {
    assert(dest->rows == m->cols);
    assert(dest->cols == m->rows);
    size_t i, j;
    // #pragma omp parallel for private(i, j) if(m->rows * m->cols > 1000)
    for (i = 0; i < m->rows; i++)
    {
        for (j = 0; j < m->cols; j++)
        {
            MAT_AT(dest,j,i) = MAT_AT(m,i,j);
        }
    }
}

// dest = a circle b 
void Hadamad(Mat dest, Mat a, Mat b) {
    assert(dest->rows == a->rows && dest->cols == a->cols
    && a->rows == b->rows && a->cols == b->cols);
    size_t i, j;
    //#pragma omp parallel for private(i, j) if(a->rows * b->cols > 1000)
    for (i = 0; i < a->rows; i++)
    {
        for (j = 0; j < a->cols; j++)
        {
            MAT_AT(dest,i,j) = MAT_AT(a,i,j)*MAT_AT(b,i,j);
        }
        
    } 
}

void GetCol(Mat dest, Mat a, int col) {
    assert(dest->rows == a->rows);
    assert(dest->cols == 1);
    assert(col < a->cols);
    size_t i;
    // #pragma omp parallel for
    for (i = 0; i < a->rows; i++)
    {
      MAT_AT(dest,i,0) = MAT_AT(a,i,col);
    } 

}



// Loss MSE = { mse_loss, mse_grad };
// Loss L1  = { l1_loss, l1_grad };
// Loss HUBER = { huber_loss, huber_grad };
// Loss CE = { ce_loss, ce_grad };
// Loss HINGE = { hinge_loss, hinge_grad };
// Loss LOGCOSH = { logcosh_loss, logcosh_grad };


float mse_cost(float y, float y_hat) {
    float diff = y_hat - y;
    return 0.5f * diff * diff;
}
float mse_cost_diff(float y, float y_hat) {
    float diff = y_hat - y;
    return diff;
}

float binary_cross_entropy_cost(float y, float y_hat) {
    y_hat = fmaxf(1e-7f, fminf(y_hat, 1.0f - 1e-7f));
    float p_class_correct = y * logf(y_hat);
    float p_class_incorrect = (1-y) * logf(1-y_hat);
    return -1.0f * (p_class_correct + p_class_incorrect);
}
float binary_cross_entropy_cost_diff(float y, float y_hat) {
    y_hat = fmaxf(1e-7f, fminf(y_hat, 1.0f - 1e-7f));
    float dC_dy_hat_c = y * 1/(y_hat);
    float dC_dy_hat_ic = (1-y) * 1/(1-y_hat);
    return -1.0f * dC_dy_hat_c + dC_dy_hat_ic;
}


void soft_max_output_layer(NN nn) {
    int size = nn.layers - 1;
    int nodes = nn.layer[size];
    Mat output = nn.post_activation[size];
    float sum = 0.0;
    for (size_t i = 0; i < nodes; i++)
    {
        sum += expf(MAT_AT(output, i, 0));
    }
    
    for (size_t i = 0; i < nodes; i++)
    {
        MAT_AT(output, i, 0) = expf(MAT_AT(output, i, 0))/sum; 
    }
    
    return;
}

// y ->
void cost_diff(Mat dest, Mat y, Mat y_hat, Function f) {
    assert(dest->rows == y->rows);
    assert(dest->cols == y->cols);
    assert(y_hat->cols == y->cols);
    assert(y_hat->rows == y->rows);
    size_t i,j;
    // #pragma omp parallel for private(i, j) if(m->rows * m->cols > 1000)
    for (i = 0; i < y->rows; i++)
    {
        for (j = 0; j < y->cols; j++)
        {

            MAT_AT(dest,i,j) = f.cost == MSE ? mse_cost_diff(MAT_AT(y,i,j), MAT_AT(y_hat,i,j)): 
                             f.cost == BINARY_CROSS_ENTROPY ? 
                             binary_cross_entropy_cost_diff(MAT_AT(y,i,j), MAT_AT(y_hat,i,j)): 
                            0;
        }
      
    }
}

float cost(NN nn, Mat y, Cost c) {
    float cost = 0.f;
    Mat pred = nn.post_activation[nn.layers - 1];
    int output_size = y->rows;
    for (size_t i = 0; i < output_size; i++) {
        cost += c == MSE ? mse_cost(MAT_AT(pred, i, 0), MAT_AT(y, i, 0)):
                     c == BINARY_CROSS_ENTROPY ? binary_cross_entropy_cost(MAT_AT(pred, i, 0), MAT_AT(y, i, 0)):
                     0;
    }
    return cost;
}

// float hinge_cost(float y, float y_hat) {
//     float diff = y - y_hat;
//     diff = diff * diff;
//     return 0.5f * diff;
// }
// float hinge_cost_diff(float y, float y_hat) {
//     float diff = y - y_hat;
//     diff = diff * diff;
//     return 0.5f * diff;
// }





Mat forward(Mat input, NN nn, Activation act) {
    // pre_ac[0] is input 
    MatCopy(nn.post_activation[0], input);
    for (int i = 1; i < nn.layers; i++) {
        // store preact
        MatMul(nn.pre_activation[i], nn.W[i-1], nn.post_activation[i-1]);
        MatSum(nn.pre_activation[i], nn.b[i-1]);

        // store post_activation
        MatCopy(nn.post_activation[i], nn.pre_activation[i]);
        MatAct(nn.post_activation[i], nn.post_activation[i], act);
    }
    return nn.post_activation[nn.layers - 1];
}

void deltaCal(NN nn, Mat y, Function f, int last) {
    Mat diff = MatInit(nn.post_activation[last]->rows, nn.post_activation[last]->cols);

    if (f.act == SIGMOID && f.cost == BINARY_CROSS_ENTROPY) {
        // diff = y_hat - y;
        MatCopy(diff, nn.post_activation[last]);
        MatSub(diff, y);
        MatCopy(nn.delta[last], diff);
    } else {
        // for end matrix
        // dC/dy
        cost_diff(diff, y, nn.post_activation[last], f);
        // hadamad product of diff and delta
        Mat sigdiff = MatInit(nn.post_activation[last]->rows, nn.post_activation[last]->cols);
        MatActDiff(sigdiff, nn.post_activation[last], f.act);
        Hadamad(nn.delta[last], diff, sigdiff);
        MatFree(sigdiff);
    }

    MatFree(diff);
    return;
}


void backprop(NN nn, Mat y, float eta, Function f) {
    // calculate delta
    int last = nn.layers - 1;

    // output layer delta
    deltaCal(nn, y, f, last);

    for (int l = last - 1; l >= 1; l--) {
        Mat Wt = MatInit(nn.W[l]->cols, nn.W[l]->rows);
        Transpose(Wt, nn.W[l]);

        Mat prod = MatInit(nn.delta[l]->rows, nn.delta[l]->cols);
        MatMul(prod, Wt, nn.delta[l + 1]);

        Mat sigdiff_h = MatInit(nn.post_activation[l]->rows, nn.post_activation[l]->cols);
        MatActDiff(sigdiff_h, nn.post_activation[l], f.act);
        Hadamad(nn.delta[l], prod, sigdiff_h);

        MatFree(Wt);
        MatFree(prod);
        MatFree(sigdiff_h);
    }

    
    float val = -1.f * eta;
    for (int l = 0; l <= last - 1; l++) {
        // dW = delta[l+1] * a[l]^T
        Mat aT = MatInit(nn.post_activation[l]->cols, nn.post_activation[l]->rows);
        Transpose(aT, nn.post_activation[l]);

        Mat dW = MatInit(nn.W[l]->rows, nn.W[l]->cols);
        MatMul(dW, nn.delta[l + 1], aT);

        MatScalar(dW, val);
        MatSum(nn.W[l], dW);

        // dB = delta[l+1]
        Mat dB = MatInit(nn.b[l]->rows, nn.b[l]->cols);
        MatCopy(dB, nn.delta[l + 1]);
        MatScalar(dB, val);
        MatSum(nn.b[l], dB);

        MatFree(aT);
        MatFree(dW);
        MatFree(dB);
    }
    return;
}
// should have sig train_mlp_sgd(in, nn, out, eta, epoch, activation, loss)
void train_mlp_sgd(Mat input, NN nn, Mat y, float eta, int epoch, Function f, bool print_error) {
    // each col new input
    // each col maps to each input
    Mat xj = MatInit(input->rows, 1);
    Mat yj = MatInit(y->rows, 1);
    for (int i = 0; i < epoch; i++) {      
        for (int j = 0; j < input->cols; j++)
        {
            GetCol(xj, input, j);
            GetCol(yj, y, j);
            // forward to initialise the current weights and params
            // takes column vector
            forward(xj, nn, f.act);
            // backwards to propagate the changes backwards
            backprop(nn, yj, eta, f);
            
        }      
        if (print_error) printf("Cost = %f\n", cost(nn, yj, f.cost));
    }
    MatFree(xj);
    MatFree(yj);
}

void print_model(NN model) {
    int hidden_layers = model.layers - 1;
    for (int i = 0; i < model.layers; i++) {
        printf("%d", model.layer[i]);
        if (i < model.layers - 1) {
            printf("->");
        }
    }

    printf("\nW values: \n");
    for (int i = 0; i < hidden_layers; i++) {
        printf("Layer %d weights: \n", i);
        MAT_PRINT(model.W[i]);
        printf("\n");
    }
    printf("\nBias values: \n");
    for (int i = 0; i < hidden_layers; i++) {
        printf("Layer %d biases: \n", i);
        MAT_PRINT(model.b[i]);
        printf("\n");
    }
}




NN *make_model(int *nodes, int layer_count, Function f) {
    NN *model = malloc(sizeof(NN));
    model->W = INIT_TENSOR_W(nodes, layer_count, f);
    model->b = MCES_INIT_B(nodes, layer_count);
    // model->delta = INIT_TENSOR(nodes, layer_count);
    model->pre_activation = INIT_ACTIVATION(nodes, layer_count);
    model->post_activation = INIT_ACTIVATION(nodes, layer_count);
    
    model->delta = INIT_DELTA(nodes, layer_count);
    
    int *arc = malloc(sizeof(int) * layer_count);
    for (int i = 0; i < layer_count; i++) {
        arc[i] = nodes[i];
    }
    model->layer = arc;
    model->layers = layer_count;
    return model;
}


Mat *INIT_TENSOR_W(int *nodes, int layer_count, Function f) {
    // init node:
    // n x 1:
    // n x n 
    // nodes contains the start 
    // I x W_1r  x ... x 
    // nodes is layer_count + 1 in size
    Mat *W = malloc(sizeof(struct mat) * (layer_count - 1));
    for (int i = 1; i < layer_count; i++) {
        int col = nodes[i - 1];
        int row = nodes[i];
        W[i - 1] = MatInit(row,col);
        if (f.act == RELU) {
            MatRandHe(W[i - 1], col);
        } else if (f.act == SIGMOID) {
            MatRandXavier(W[i - 1], col);
        } else {
            MatRand(W[i - 1], -1, 1);
        }
    }
    return W;
}
// 
Mat *INIT_ACTIVATION(int *nodes, int layer_count) {
    // init node:
    // n x 1 layers 
    Mat *W = malloc(sizeof(struct mat) * (layer_count));
    for (int i = 0; i < layer_count; i++) {
        int rows = nodes[i];
        W[i] = MatInit(rows,1);
    }
    return W;
}

Mat *INIT_DELTA(int *nodes, int layer_count) {
    // init node:
    // n x 1 layers 
    Mat *D = malloc(sizeof(struct mat) * (layer_count));
    for (int i = 0; i < layer_count; i++) {
        int rows = nodes[i];
        D[i] = MatInit(rows, 1);
    }
    return D;
}


// nth one is: (n,1)
Mat *MCES_INIT_B(int *nodes, int layer_count) {

    Mat *B = malloc(sizeof(struct mat) * (layer_count - 1));
    for (int i = 0; i < layer_count - 1; i++) {
        int m = nodes[i + 1];
        B[i] = MatInit(m, 1);
        MatRand(B[i], -1.0f, 1.0f);  // Small random bias initialization
    }
    return B;
}