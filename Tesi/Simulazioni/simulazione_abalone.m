rng(1);
clc;
clear;
load('/Users/robertofierimonte/Documents/MATLAB/Regressione/abalone.mat')
[X,Y]=preprocess(abalone_X,abalone_Y);
clear abalone_X;
clear abalone_Y;
simulaz_regressione3