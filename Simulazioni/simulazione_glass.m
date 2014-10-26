rng(1)
clc
clear
load('Datasets/MC/glass.mat')
[X,Y]=preprocess(glass_X,glass_Y);
clear glass_X;
clear glass_Y;
simulaz_classificazione