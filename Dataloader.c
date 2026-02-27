#include "Dataloader.h"
#define IMAGE_PIXEL(data, n, i, j) (data)->images[n * (data->rows) * (data->cols) + i * (data->cols) + j]

void MatPrintNum(Mat m, int label) {
    printf("Label: %d\n", label);
    for (size_t i = 0; i < m->rows; i++)
    {
        for (size_t j = 0; j < m->cols; j++){
            printf("%.0f", MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("\n");
}

uint32_t convert_endian(uint32_t val) {
    return ((val & 0xFF000000) >> 24) |
           ((val & 0x00FF0000) >> 8) |
           ((val & 0x0000FF00) << 8) |
           ((val & 0x000000FF) << 24);
}

void colourNormaliseArray(Mat m, ImgData *data) {
    for (int k = 0; k < data->num_images; k++)
    {
        for (int i = 0; i < data->rows; i++) {
            for (int j = 0; j < data->cols; j++)
            { 
                int row = i * data->cols + j;
                MAT_AT(m, row, k) = (float)IMAGE_PIXEL(data, k, i, j) / 255.0f;
            }
        }
    }
    
}


void labels_to_onehot(Mat yout, ImgData *data) {
    // 10 classes;
    // 
    // Mat yout = MatInit(10, data->num_images);
    for (size_t i = 0; i < 10; i++)
    {
        for (size_t j = 0; j < data->num_images; j++)
        {
            if (data->images[j] == i) {
                MAT_AT(yout, i, j) = 1.0;
            }
        }
        
    }
    //return yout;
}

ImgData *load_mnist() {

    // read images
    FILE *f = fopen("MNIST/train-images.idx3-ubyte", "rb");
    if (f == NULL) {
        printf("err");
        fclose(f);
        return NULL;
    }
    
    uint32_t magic_number = 0;
    uint32_t number_of_images = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;

    fread(&magic_number, sizeof(magic_number), 1, f);
    fread(&number_of_images, sizeof(number_of_images), 1, f);
    fread(&rows, sizeof(rows), 1, f);
    fread(&cols, sizeof(cols), 1, f);
    printf("%02x %02x\n", magic_number, number_of_images);
    magic_number = convert_endian(magic_number);
    number_of_images = convert_endian(number_of_images);
    rows = convert_endian(rows);
    cols = convert_endian(cols);
    
    uint8_t *images = malloc(sizeof(uint8_t) * number_of_images * rows * cols);
    
    for (int i = 0; i < number_of_images; i++) {
        for (int j = 0; j < rows; j++)
        {
            for (int k = 0; k < cols; k++)
            {
                uint8_t pixel = 0;
                if (fread(&pixel, sizeof(uint8_t), 1, f) != 1) {
                    printf("Error reading image data\n");
                    free(images);
                    fclose(f);
                    return NULL;
                }
                images[i * rows * cols + j * cols + k] = pixel;
            }
            
        }
    }

    // read labels
    FILE *l = fopen("MNIST/train-labels.idx1-ubyte", "rb");
    if (l == NULL) {
        printf("err");
        fclose(f);
        return NULL;
    }

    int magic_number_labels = 0;
    int number_of_images_labels = 0;
    // int rows_labels = 0;
    // int cols_labels = 0;

    fread(&magic_number_labels, sizeof(magic_number_labels), 1, l);
    fread(&number_of_images_labels, sizeof(number_of_images_labels), 1, l);
    printf("%02x %02x\n", magic_number_labels, number_of_images_labels);
    
    magic_number_labels = convert_endian(magic_number_labels);
    number_of_images_labels = convert_endian(number_of_images_labels);
    
    uint8_t *labels = malloc(sizeof(uint8_t) * number_of_images_labels);

    for (int i = 0; i < number_of_images_labels; i++) {
        uint8_t label = 0;
        if (fread(&label, sizeof(label), 1, l) != 1 ) {
            free(images);
            free(labels);
            printf("Err reading label");
            return NULL;
        }
        labels[i] = label;
    }

    
    ImgData *data = malloc(sizeof(ImgData));
    if (data == NULL) {
        free(images);
        fclose(f);
        return NULL;
    }
    data->images = images;
    data->num_images = number_of_images;
    data->labels = labels;
    data->rows = rows;
    data->cols = cols;
    fclose(f);
    return data;
}


void show_image(ImgData *data, int img) {
    int window_height = 240;
    int window_width = 240;
    InitWindow(window_width, window_height, TextFormat("%d", data->labels[img]));
    SetWindowState(FLAG_WINDOW_RESIZABLE);
    SetTargetFPS(60);

    int x = 10;
    int y = 10;
    while (!WindowShouldClose()) {
        BeginDrawing();
        ClearBackground(BLACK);
        for (size_t i = 0; i < data->rows; i++)
        {
            for (size_t j = 0; j < data->cols; j++)
            {
                int w = IMAGE_PIXEL(data, img, i, j);
                DrawRectangle(j*x, i*y, 24, 24, 
                    (Color){.r = w, .b=w, .g = w, .a = 255});
            }
            
        }
        EndDrawing();
    }
    // printf("Label = %d\n", data->labels[img]);
}

// int drawUIDesign(int screenWidth, int screenHeight) {
//     double wl = (float)GetScreenWidth()/screenWidth;
//     double hl = (float)GetScreenHeight()/screenHeight;

//     Rectangle anchor_0 = (Rectangle){20.000000*wl, 20.000000*hl, 260.000000*wl, 260.000000*hl};    
//     if (GuiButton((Rectangle){20.000000*wl, 300.000000*hl, 140.000000*wl, 40.000000*hl}, "BUTTON 0"))  return 0; 
//     GuiDrawText("SAMPLE TEXT", (Rectangle){180.000000*wl, 300.000000*hl, 100.000000*wl, 40.000000*hl}, TEXT_ALIGN_CENTER, BLACK);
//     return -1;
// }

void draw_image(Mat inp) {
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


   
    while (!WindowShouldClose()) {
        Vector2 mouse = GetMousePosition();

        // rounding
        int rx = (int)roundf(mouse.x/10);
        int ry = (int)roundf(mouse.y/10);
        bool md = IsMouseButtonDown(MOUSE_BUTTON_LEFT);
        BeginDrawing();
        ClearBackground(RAYWHITE);
        //int button_pressed = drawUIDesign(window_width, window_height);
        for (size_t i = 0; i < 28; i++)
        {
            DrawLine(i*10, 0, i*10, 280, BLACK);
            DrawLine(0, i*10, 280, i*10, BLACK);
        }

        if (md) {
            //printf("Pressed: x:%d y:%d", rx, ry);
            //DrawRectangle(rx,ry,10,10,BLACK);
            MAT_AT(inp, rx, ry) = 1.0;
        }

        for (size_t i = 0; i < inp->cols; i++)
        {
            for (size_t j = 0; j < inp->rows; j++)
            {
                if (MAT_AT(inp, i, j) > 0) {
                    DrawRectangle(i*10,j*10,10,10,BLACK);
                }
            }
            
        }
        
        EndDrawing();
    }
    //return inp;
}