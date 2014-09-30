rng(1);
clc;
clear;
load('/Users/robertofierimonte/Documents/MATLAB/Classificazione binaria/pimaindians.mat')
[X,Y]=preprocess(pimaindians_X,pimaindians_Y);
clear pimaindians_X;
clear pimaindians_Y;
simulaz_classbin