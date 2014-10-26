rng(1)
clc
clear
load('Datasets/BC/eyestate.mat')
[X,Y]=preprocess(eyestate_X,eyestate_Y);
clear eyestate_X;
clear eyestate_Y;
simulaz_classbin