rng(1);
clc;
clear;
load('/Users/robertofierimonte/Documents/MATLAB/Classificazione binaria/eyestate.mat')
[X,Y]=preprocess(eyestate_X,eyestate_Y);
clear eyestate_X;
clear eyestate_Y;
simulaz_classbin