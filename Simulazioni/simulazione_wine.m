rng(1)
clc
clear
load('Datasets/MC/wine.mat');
[X,Y]=preprocess(wine_X,wine_Y);
clear wine_X;
clear wine_Y;
simulaz_classificazione