rng(1)
clc
clear
load('Datasets/BC/transfusion.mat')
[X,Y]=preprocess(transfusion_X,transfusion_Y);
clear transfusion_X;
clear transfusion_Y;
simulaz_classbin