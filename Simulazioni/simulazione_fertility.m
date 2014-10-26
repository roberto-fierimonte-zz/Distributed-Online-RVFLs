rng(1)
clc
clear
load('Datasets/BC/fertility.mat')
[X,Y]=preprocess(fertility_X,fertility_Y);
clear fertility_X;
clear fertility_Y;
simulaz_classbin