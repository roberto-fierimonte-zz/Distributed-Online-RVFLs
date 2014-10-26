rng(1)
clc
clear
load('Datasets/BC/banknote.mat')
[X,Y]=preprocess(banknote_X,banknote_Y);
clear banknote_X;
clear banknote_Y;
simulaz_classbin