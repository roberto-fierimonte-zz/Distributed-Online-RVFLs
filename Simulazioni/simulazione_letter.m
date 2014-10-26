rng(1)
clc
clear
load('Datasets/MC/letter.mat')
[X,Y]=preprocess(letter_X,letter_Y);
clear letter_X;
clear letter_Y;
simulaz_classificazione