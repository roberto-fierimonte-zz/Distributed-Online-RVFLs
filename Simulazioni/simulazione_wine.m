rng(1)
clc
clear
load('/Users/robertofierimonte/Documents/MATLAB/Classificazione multiclasse/wine.mat');
[X,Y]=preprocess(wine_X,wine_Y);
clear wine_X;
clear wine_Y;
simulaz_classificazione