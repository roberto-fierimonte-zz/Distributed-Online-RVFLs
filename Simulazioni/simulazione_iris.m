clc
clear
load('/Users/robertofierimonte/Documents/MATLAB/Classificazione multiclasse/iris.mat')
[X,Y]=preprocess(iris_X,iris_Y);
clear iris_X;
clear iris_Y;
simulaz_classificazione