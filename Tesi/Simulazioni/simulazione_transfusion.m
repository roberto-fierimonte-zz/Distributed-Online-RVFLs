rng(1);
clc;
clear;
load('/Users/robertofierimonte/Documents/MATLAB/Classificazione binaria/transfusion.mat')
[X,Y]=preprocess(transfusion_X,transfusion_Y);
clear transfusion_X;
clear transfusion_Y;
simulaz_classbin