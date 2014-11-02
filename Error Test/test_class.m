function [error] = test_class(X_test,Y_test,net,beta)
%TEST_CLASS measure test error for a RVFL in multiclassification problems
%
%Input: X_test: (p x n) matrix of input test patterns
%       Y_test: (p x m) matrix of output test patterns (each column
%           correspond to a class)
%       net: struct object that gather the informations about the RVFL
%           (number of hidden node, hidden parameters and regularization
%           parameter)
%       beta: (K x m) matrix of the RVFL output weights
%
%Output: error: percentage of misclassified test patterns over the total
%           of test patterns

    pX=size(X_test,1);
    esp=(exp(-(bsxfun(@plus,X_test*(net.coeff)',net.bias')))+1).^-1;
    exit=(vec2ind((esp*beta)'))';
    error=1/(pX)*sum(sum(exit~=Y_test));
end

