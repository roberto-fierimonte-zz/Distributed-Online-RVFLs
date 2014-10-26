rng(1)
clc
clear
load('Datasets/R/bodyfat.mat')
[X,Y]=preprocess(bodyfat_X,bodyfat_Y);
clear bodyfat_X;
clear bodyfat_Y;
simulaz_regressione