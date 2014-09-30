rng(1);
clc;
clear;
load('/Users/robertofierimonte/Documents/MATLAB/Classificazione binaria/skin.mat')
[X,Y]=preprocess(skin_X,skin_Y);
clear skin_X;
clear skin_Y;
simulaz_classbin