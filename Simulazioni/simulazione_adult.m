rng(1)
clc
clear
load('Datasets/BC/adult.mat')
[X,Y]=preprocess(adult_X,adult_Y);
clear adult_X;
clear adult_Y;
simulaz_classbin