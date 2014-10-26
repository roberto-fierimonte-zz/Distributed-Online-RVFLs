rng(1)
clc
clear
load('Datasets/R/quake.mat')
[X,Y]=preprocess(quake_X,quake_Y);
clear quake_X;
clear quake_Y;
simulaz_regressione