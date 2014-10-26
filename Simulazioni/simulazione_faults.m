rng(1)
clc
clear
load('/Users/robertofierimonte/Documents/MATLAB/Classificazione multiclasse/faults.mat')
[X,Y]=preprocess(faults_X,faults_Y);
clear faults_X;
clear faults_Y;
simulaz_classificazione