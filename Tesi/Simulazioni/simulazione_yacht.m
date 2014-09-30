rng(1);
clc;
clear;
load('/Users/robertofierimonte/Documents/MATLAB/Regressione/yacht.mat')
[X,Y]=preprocess(yacht_X,yacht_Y);
clear yacht_X;
clear yacht_Y;
simulaz_regressione3