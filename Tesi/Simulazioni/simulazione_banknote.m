rng(1);
clc;
clear;
load('/Users/robertofierimonte/Documents/MATLAB/Classificazione binaria/banknote.mat')
[X,Y]=preprocess(banknote_X,banknote_Y);
clear banknote_X;
clear banknote_Y;
simulaz_classbin