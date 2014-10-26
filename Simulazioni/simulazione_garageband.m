rng(1)
clc
clear
load('/Users/robertofierimonte/Documents/MATLAB/Classificazione multiclasse/garageband.mat')
[X,Y]=preprocess(garageband_X,garageband_Y);
clear garageband_X;
clear garageband_Y;
simulaz_classificazione