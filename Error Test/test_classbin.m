function [error] = test_classbin(X_test,Y_test,net,beta)
%TEST_CLASSBIN measure test error for a RVFL in binary classification 
%problems
%
%Input: X_test: (p x n) matrix of input test patterns
%       Y_test: (p x 1) vector of output test patterns
%       net: struct object that gather the informations about the RVFL
%           (number of hidden node, hidden parameters and regularization
%           parameter)
%       beta: (K x 1) vector of the RVFL output weights
%
%Output: error: percentage of misclassified test patterns over the total
%           of test patterns

    pX=size(X_test,1);
    esp=(exp(-bsxfun(@plus,X_test*(net.coeff)',net.bias'))+1).^-1;
    exit=sign(esp*beta);
    error=1/(pX)*sum(exit~=Y_test);
end

