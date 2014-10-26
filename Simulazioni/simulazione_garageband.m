rng(1)
clc
clear
load('Datasets/MC/garageband.mat')
[X,Y]=preprocess(garageband_X,garageband_Y);
clear garageband_X;
clear garageband_Y;
simulaz_classificazione