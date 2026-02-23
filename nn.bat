@echo off

echo Running NN Engine

gcc -Wall -Werror -fopenmp NN_mlp.c NN.c Dataloader.c -o network -I"C:\Users\karan\Desktop\CompUni\Major Projects\raylib-5.5_win64_mingw-w64\include" -L"C:\Users\karan\Desktop\CompUni\Major Projects\raylib-5.5_win64_mingw-w64\lib" -lraylib -lopengl32 -lgdi32 -lwinmm

if exist network.exe (
    echo running network.exe
    network.exe
) else (
    echo Build failed
)
exit