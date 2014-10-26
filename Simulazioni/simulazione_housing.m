rng(1)
clc
clear
load('Datasets/R/housing.mat')
[X,Y]=preprocess(housing_X,housing_Y);
clear housing_X;
clear housing_Y;
simulaz_regressione