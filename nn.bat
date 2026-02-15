@echo off

echo Running NN Engine

gcc -Wall -Werror -fopenmp NN_mlp.c NN.c -o network 

if exist network.exe (
    echo running network.exe
    network.exe
) else (
    echo Build failed
)
exit