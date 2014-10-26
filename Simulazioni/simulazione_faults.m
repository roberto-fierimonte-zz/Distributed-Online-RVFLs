rng(1)
clc
clear
load('Datasets/MC/faults.mat')
[X,Y]=preprocess(faults_X,faults_Y);
clear faults_X;
clear faults_Y;
simulaz_classificazione