rng(1);
clc;
clear;
load('/Users/robertofierimonte/Documents/MATLAB/Regressione/CASP.mat')
[X,Y]=preprocess(CASP_X,CASP_Y);
clear CASP_X;
clear CASP_Y;
simulaz_regressione3