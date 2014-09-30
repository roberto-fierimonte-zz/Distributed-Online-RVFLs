rng(1);
clc;
clear;
load('/Users/robertofierimonte/Documents/MATLAB/Regressione/strength.mat')
[X,Y]=preprocess(strength_X,strength_Y);
clear strength_X;
clear strength_Y;
simulaz_regressione3