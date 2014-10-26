rng(1)
clc
clear
load('/Users/robertofierimonte/Documents/MATLAB/Classificazione multiclasse/pageblocks.mat')
[X,Y]=preprocess(pageblocks_X,pageblocks_Y);
clear pageblocks_X;
clear pageblocks_Y;
simulaz_classificazione