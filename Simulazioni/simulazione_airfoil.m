rng(1)
clc
clear
load('Datasets/R/airfoil.mat')
[X,Y]=preprocess(airfoil_X,airfoil_Y);
clear airfoil_X;
clear airfoil_Y;
simulaz_regressione