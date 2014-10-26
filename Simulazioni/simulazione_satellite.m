rng(1)
clc
clear
load('/Users/robertofierimonte/Documents/MATLAB/Classificazione multiclasse/satellite.mat')
[X,Y]=preprocess(satellite_X,satellite_Y);
clear satellite_X;
clear satellite_Y;
simulaz_classificazione