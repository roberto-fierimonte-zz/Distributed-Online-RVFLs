rng(1)
clc
clear
load('Datasets/MC/satellite.mat')
[X,Y]=preprocess(satellite_X,satellite_Y);
clear satellite_X;
clear satellite_Y;
simulaz_classificazione