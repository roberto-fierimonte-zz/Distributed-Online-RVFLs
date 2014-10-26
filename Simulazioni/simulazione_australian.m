rng(1)
clc
clear
load('Datasets/BC/australian.mat')
[X,Y]=preprocess(australian_X,australian_Y);
clear australian_X;
clear australian_Y;
simulaz_classbin