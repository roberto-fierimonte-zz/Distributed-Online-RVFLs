function [X,Y] = preprocess(data_X,data_Y)

X=mapminmax(data_X',0,1);X=X';
Y=data_Y;
r=randperm(size(X,1));
X=X(r,:);
Y=Y(r,:);
end