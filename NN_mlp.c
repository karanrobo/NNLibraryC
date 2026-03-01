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

void predPrint(Mat res) {
    //assert(res->cols == 1);
    for (int i = 0; i < res->rows; i++) {
        printf("%d: %f, ", i, MAT_AT(res, i, 0));
    }
    printf("\n");
}

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
    int sample_size = 30000;
    // int img_num = 5;
    //int label = img->labels[img_num];
    // Mat x = MatInit(img->rows * img->cols, img->num_images);
    Mat x = MatInit(img->rows * img->cols, sample_size);
    
    colourNormaliseArray(x, img, sample_size); 
    //MatPrintNum(x, label);
    
    
    //MatPrintNum(xin, label);
    //show_image(img, img_num);

    // each column is a new input
    Mat yout = MatInit(10, sample_size);
    labels_to_onehot(yout, img, sample_size);

    // y->labels, x->input
    // uint8_t *y = img->labels;
    
    float eta = 0.005f;
    int epoch = 5;
    // for mnist-> 28x28-> 
    int architecture[] = {784,128,64,10};
    Activation act[] =   {1,1,1,SOFTMAX};
    int alen = sizeof(architecture)/sizeof(architecture[0]);
    NN *nn = make_model(architecture, alen, act, MCLASS_CROSS_ENTROPY);
    // MAT_PRINT(xin);
    // MAT_PRINT(yout);
    train_mlp_sgd(x,*nn, yout, eta, epoch, 0);


    Mat inp = MatInit(28,28);
    //draw_image(inp);
    int window_height = 400;
    int window_width = 300;
    InitWindow(window_width, window_height, "Input");
    SetWindowState(FLAG_WINDOW_RESIZABLE);
    SetTargetFPS(60);
    // Mat inp = MatInit(28,28);
    
    // add later
    // Mat brush = MatInit(9,9);
    // for (int i = 0; i < brush->cols; i++) {
    //     for (int j = 0; j < brush->rows; j++)
    //     {
    //         int dist = abs(i - j);
    //         if (i == j) {
    //             MAT_AT(brush, i, j) = 1.0;
    //         } else if (dist > 0 && dist <= 2) {
    //             MAT_AT(brush, i, j) = 0.5;
    //         } else {
    //             MAT_AT(brush, i, j) = 0;
    //         }
    //     }   
    // }

    

    Mat inp_flat = MatInit(28 * 28, 1);

    //Mat conv = MatInit(9,9);

  

    bool change = false;
    while (!WindowShouldClose()) {
        Vector2 mouse = GetMousePosition();

        // rounding
        int rx = (int)floorf(mouse.x/10.0f);
        int ry = (int)floorf(mouse.y/10.0f);
        bool md = IsMouseButtonDown(MOUSE_BUTTON_LEFT);
         
        BeginDrawing();
        ClearBackground(RAYWHITE);
        //int button_pressed = drawUIDesign(window_width, window_height);
        for (size_t i = 0; i < 28; i++)
        {
            DrawLine(i*10, 0, i*10, 280, BLACK);
            DrawLine(0, i*10, 280, i*10, BLACK);
        }

         if (IsKeyPressed(KEY_C)) {
            for (size_t i = 0; i < inp->cols; i++)
            {
                for (size_t j = 0; j < inp->rows; j++)
                {
                   MAT_AT(inp, i, j) = 0.0;
                }
                
            }
        }
        

        if (md && rx >= 0 && rx < 28 && ry >= 0 && ry < 28) {
            //printf("Pressed: x:%d y:%d", rx, ry);
            //DrawRectangle(rx,ry,10,10,BLACK);

           // MAT_AT(inp, ry, rx) = 1.0;

            if(rx-1 >= 0){
                if(ry-1 >= 0) MAT_AT(inp, ry-1, rx-1) = 0.3; 
                MAT_AT(inp, ry, rx-1) = 0.5; 
                if(ry+1 < 28)MAT_AT(inp, ry+1, rx-1) = 0.3;
            }
            if (ry-1 >= 0) MAT_AT(inp, ry-1, rx) = 0.5;      
            MAT_AT(inp, ry, rx) = 1.0;        
            if (ry+1 < 28) MAT_AT(inp, ry+1, rx) = 0.5;

            if (rx+1 < 28) {

                if(ry-1 >= 0) MAT_AT(inp, ry-1, rx+1) = 0.3;    
                MAT_AT(inp, ry, rx+1) = 0.5;        
                if(rx+1 < 28) MAT_AT(inp, ry+1, rx+1) = 0.3;
            }

       
            change = true;
        }

        for (size_t i = 0; i < inp->cols; i++)
        {
            for (size_t j = 0; j < inp->rows; j++)
            {
                if (MAT_AT(inp, j, i) > 0) {
                    int val = 255 * MAT_AT(inp, j, i);
                    //printf("%d\n\n", val);
                    DrawRectangle(i*10,j*10,10,10,(Color){255-val,255-val,255-val,255});
                }
            }
            
        }

        if (change) {
            MatFlat(inp_flat, inp);
            Mat test = forward(inp_flat, *nn);
            predPrint(test);
            // float max = -1.0f;
            // int val = -1;
            // for (int i = 0; i < test->rows; i++) {
            //     if (MAT_AT(test, i, 0) > max) {
            //         max = MAT_AT(test, i,0);
            //         val = i;
            //     }
            // }
            // printf("Pred: %d, %f\n", val, MAT_AT(test, val, 0));
        }
        
       
        change = false;
        EndDrawing();
    }

    

    MatFree(inp_flat);
    MatFree(inp);
    return 0;
}
