rng(1)
clc
clear
load('Datasets/BC/liver.mat')
[X,Y]=preprocess(liver_X,liver_Y);
clear liver_X;
clear liver_Y;
simulaz_classbin