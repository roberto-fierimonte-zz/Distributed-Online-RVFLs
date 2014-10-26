rng(1)
clc
clear
load('Datasets/MC/segment.mat')
[X,Y]=preprocess(segment_X,segment_Y);
clear segment_X;
clear segment_Y;
simulaz_classificazione