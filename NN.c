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
    #pragma omp parallel for private(i, j, k) if(a->rows * b->cols > 1000)
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
    #pragma omp parallel for private(i, j) if(m->rows * m->cols > 1000)
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
    int lim = 100;
    bool rb = m->rows > lim;
    bool cb = m->cols > lim;
    if (rb || cb) {
        int r = (rb) ? 10 : m->rows;
        int c = (cb) ? 10 : m->cols;
        for (size_t i = 0; i < r; i++){
            for (size_t j = 0; j < c; j++){
                printf(" %f ", MAT_AT(m, i, j));
            }
            printf("\n");
        }
        printf("Omitted rows=%lld, cols=%lld\n",
            (rb)? m->rows-lim : 0, (cb)? m->cols-lim :0);
    } else {
        
        for (size_t i = 0; i < m->rows; i++) {
            for (size_t j = 0; j < m->cols; j++){
                printf(" %f ", MAT_AT(m, i, j));
            }
            printf("\n");
        }
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


void MatAct(NN nn, Mat dest, Mat m, int layer) {
    assert(dest->rows == m->rows);
    assert(dest->cols == m->cols);
    size_t i,j;
    // #pragma omp parallel for private(i, j) if(m->rows * m->cols > 1000)
    for (i = 0; i < m->rows; i++)
    {
        for (j = 0; j < m->cols; j++)
        {

            MAT_AT(dest,i,j) = nn.activations[layer] == RELU ? ReLU(MAT_AT(m,i,j)) : 
                            nn.activations[layer] == SIGMOID ? sigmoidf(MAT_AT(m,i,j)) : 
                            LeakyReLU(MAT_AT(m,i,j));
        }
      
    }
}

// for sig(m)
void MatActDiff(NN nn, Mat dest, Mat m, int layer) {
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
            MAT_AT(dest, i, j) = nn.activations[layer] == RELU ? ReLUDiff(MAT_AT(m,i,j)) : 
                            nn.activations[layer] == SIGMOID ? sigmoidfDiff(MAT_AT(m,i,j)) : 
                            nn.activations[layer] == LEAKY_RELU ? LeakyReLUDiff(MAT_AT(m,i,j)) : 0;
            if (nn.activations[layer] == SOFTMAX) {
                // handleed in the forward
                return;
            }
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
    // #pragma omp parallel for private(i, j) if(a->rows * b->cols > 1000)
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

// dest nx1, this is row wise
void MatFlat(Mat dest, Mat src) {
    assert(dest->rows == src->rows * src->cols);
    assert(dest->cols == 1);
    int num = 0;
    for (int i = 0; i < src->rows; i++) {
        for (int j = 0; j < src->cols; j++)
        {

            MAT_AT(dest, num, 0) = MAT_AT(src, i, j);
            num++;
        }
        
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

float mclass_cross_entropy_cost(NN nn, Mat y) {
    int size = nn.layers - 1;
    int nodes = nn.layer[size];
    Mat y_hat = nn.post_activation[size];
    float sum = 0.0;
    for (size_t i = 0; i < nodes; i++)
    {   
        sum += -1.f * MAT_AT(y, i, 0) * logf(MAT_AT(y_hat, i, 0));
    }
    return sum;
}

// for stability, we rubtract the max from each exponent
void soft_max_output_layer(NN nn) {
    int size = nn.layers - 1;
    int nodes = nn.layer[size];
    Mat output = nn.post_activation[size];
    float max = MAT_AT(output, 0, 0);

    for (size_t i = 1; i < nodes; i++)
    {
        if (max < MAT_AT(output, i, 0)) {
            max = MAT_AT(output, i, 0);
        }
    }
    

    // compute exp(x - max) and sum
    float sum = 0.0;
    for (size_t i = 0; i < nodes; i++)
    {
        MAT_AT(output, i, 0) = expf(MAT_AT(output, i, 0) - max);
        sum += MAT_AT(output, i, 0);
    }
    
    if (sum <= 0.0) {
        return;
    }

    for (size_t i = 0; i < nodes; i++)
    {
        MAT_AT(output, i, 0) = MAT_AT(output, i, 0)/sum; 
    }

    return;
}

// y ->
void cost_diff(NN nn, Mat dest, Mat y, Mat y_hat) {
    assert(dest->rows == y->rows);
    assert(dest->cols == y->cols);
    assert(y_hat->cols == y->cols);
    assert(y_hat->rows == y->rows);
    size_t i,j;

     // Note: MCLASS_CROSS_ENTROPY uses the shortcut in deltaCal (y_hat - y)
    // and should never call this function

    // #pragma omp parallel for private(i, j) if(m->rows * m->cols > 1000)
    for (i = 0; i < y->rows; i++)
    {
        for (j = 0; j < y->cols; j++)
        {

            MAT_AT(dest,i,j) = nn.cost == MSE ? mse_cost_diff(MAT_AT(y,i,j), MAT_AT(y_hat,i,j)): 
                             nn.cost == BINARY_CROSS_ENTROPY ? 
                             binary_cross_entropy_cost_diff(MAT_AT(y,i,j), MAT_AT(y_hat,i,j)): 
                            0;
        }
      
    }
}

float cost(NN nn, Mat y) {
    if (nn.cost == MCLASS_CROSS_ENTROPY) {
        return mclass_cross_entropy_cost(nn, y);
    }
    float cost = 0.f;
    Mat pred = nn.post_activation[nn.layers - 1];
    int output_size = y->rows;
    for (size_t i = 0; i < output_size; i++) {
        cost += nn.cost == MSE ? mse_cost(MAT_AT(pred, i, 0), MAT_AT(y, i, 0)):
                     nn.cost == BINARY_CROSS_ENTROPY ? binary_cross_entropy_cost(MAT_AT(pred, i, 0), MAT_AT(y, i, 0)):
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





Mat forward(Mat input, NN nn) {
    // pre_ac[0] is input 
    MatCopy(nn.post_activation[0], input);
    for (int i = 1; i < nn.layers; i++) {
        // store preact
        MatMul(nn.pre_activation[i], nn.W[i-1], nn.post_activation[i-1]);
        MatSum(nn.pre_activation[i], nn.b[i-1]);

        // store post_activation
        MatCopy(nn.post_activation[i], nn.pre_activation[i]);
        if (i == nn.layers - 1 && nn.activations[i] == SOFTMAX) {
            soft_max_output_layer(nn);
        } else {
            MatAct(nn, nn.post_activation[i], nn.post_activation[i], i);
        }
    }
    return nn.post_activation[nn.layers - 1];
}

void deltaCal(NN nn, Mat y, int last) {
    Mat diff = MatInit(nn.post_activation[last]->rows, nn.post_activation[last]->cols);

    if ((nn.activations[last] == SIGMOID && nn.cost == BINARY_CROSS_ENTROPY)) {
        // diff = y_hat - y;
        MatCopy(diff, nn.post_activation[last]);
        MatSub(diff, y);
        MatCopy(nn.delta[last], diff);
    } else if (nn.activations[last] == SOFTMAX && nn.cost == MCLASS_CROSS_ENTROPY) {
        MatCopy(diff, nn.post_activation[last]);
        // MAT_PRINT(diff);
        // MAT_PRINT(y);
        MatSub(diff, y);
        MatCopy(nn.delta[last], diff);
    } else {
        // for end matrix
        // dC/dy
        cost_diff(nn, diff, y, nn.post_activation[last]);
        // hadamad product of diff and delta
        Mat sigdiff = MatInit(nn.post_activation[last]->rows, nn.post_activation[last]->cols);
        MatActDiff(nn, sigdiff, nn.post_activation[last], last);
        Hadamad(nn.delta[last], diff, sigdiff);
        MatFree(sigdiff);
    }

    MatFree(diff);
    return;
}


void backprop(NN nn, Mat y, float eta) {
    // calculate delta
    int last = nn.layers - 1;

    // output layer delta
    deltaCal(nn, y, last);

    for (int l = last - 1; l >= 1; l--) {
        Mat Wt = MatInit(nn.W[l]->cols, nn.W[l]->rows);
        Transpose(Wt, nn.W[l]);

        Mat prod = MatInit(nn.delta[l]->rows, nn.delta[l]->cols);
        MatMul(prod, Wt, nn.delta[l + 1]);

        Mat sigdiff_h = MatInit(nn.post_activation[l]->rows, nn.post_activation[l]->cols);
        MatActDiff(nn, sigdiff_h, nn.post_activation[l], l);
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
void train_mlp_sgd(Mat input, NN nn, Mat y, float eta, int epoch, bool print_error) {
    // each col new input
    // each col maps to each input

    int cols = (int)input->cols;
    int *rand_arr = malloc(sizeof(int) * cols);
    for (int i = 0; i < cols; i++) rand_arr[i] = i;

    printf("xr = %lld xc = %lld yr = %lld yc = %lld\n", input->rows,input->cols, y->rows, y->cols);
    Mat xj = MatInit(input->rows, 1);
    Mat yj = MatInit(y->rows, 1);
    for (int i = 0; i < epoch; i++) { 
        // shuffling per epoch to remove order bias 
        for (int j = cols - 1; j > 0; j--) {
            int k = (int)((double)rand() / ((double)RAND_MAX + 1.0) * (j + 1)); // 0..j
            int tmp = rand_arr[j];
            rand_arr[j] = rand_arr[k];
            rand_arr[k] = tmp;
        }    
        for (int j = 0; j < cols; j++)
        {
            int n = rand_arr[j];
            GetCol(xj, input, n);
            GetCol(yj, y, n);
            // forward to initialise the current weights and params
            // takes column vector
            forward(xj, nn);
            // backwards to propagate the changes backwards
            backprop(nn, yj, eta);
            //printf("on col: %d\n", j);
            
        }      
        if (print_error) printf("Cost = %f\n", cost(nn, yj));
    }
    MatFree(xj);
    MatFree(yj);
    free(rand_arr);
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




NN *make_model(int *nodes, int layer_count, Activation *act, Cost cost) {
    NN *model = malloc(sizeof(NN));
    model->W = INIT_TENSOR_W(nodes, layer_count, act);
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

    model->activations = malloc(sizeof(Activation) * layer_count);
    for (size_t i = 0; i < layer_count; i++)
    {
        model->activations[i] = act[i];
    }
    
    model->cost = cost;


    return model;
}


Mat *INIT_TENSOR_W(int *nodes, int layer_count, Activation *act) {
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
        if (act[i] == RELU) {
            MatRandHe(W[i - 1], col);
        } else if (act[i] == SIGMOID) {
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