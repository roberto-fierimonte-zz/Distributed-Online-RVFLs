rng(1)
clc
clear
load('/Users/robertofierimonte/Documents/MATLAB/Classificazione multiclasse/glass.mat')
[X,Y]=preprocess(glass_X,glass_Y);
clear glass_X;
clear glass_Y;
simulaz_classificazione