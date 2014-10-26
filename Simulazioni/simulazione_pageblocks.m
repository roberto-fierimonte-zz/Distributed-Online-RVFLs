rng(1)
clc
clear
load('Datasets/MC/pageblocks.mat')
[X,Y]=preprocess(pageblocks_X,pageblocks_Y);
clear pageblocks_X;
clear pageblocks_Y;
simulaz_classificazione