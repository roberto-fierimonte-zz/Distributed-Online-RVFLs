rng(1)
clc
clear
load('Datasets/BC/pimaindians.mat')
[X,Y]=preprocess(pimaindians_X,pimaindians_Y);
clear pimaindians_X;
clear pimaindians_Y;
simulaz_classbin