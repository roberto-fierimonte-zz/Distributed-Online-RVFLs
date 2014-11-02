function [X,Y] = preprocess(dataset)
%PREPROCESS add documentation

    data_X=dataset.X;
    data_Y=dataset.Y;
    X=mapminmax(data_X',-1,1);X=X';
    if strcmp(dataset.type,'MC')
        Y=dummyvar(data_Y);
    else
        Y=data_Y;
    end
    r=randperm(size(X,1));
    X=X(r,:);
    Y=Y(r,:);
end