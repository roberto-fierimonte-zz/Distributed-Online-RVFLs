function [NRMSE,NSR] = test_reg(X_test,Y_test,net,beta)
%TEST_REG measure test error for a RVFL in regression problems
%
%Input: X_test: (p x n) matrix of input test patterns
%       Y_test: (p x m) matrix of output test patterns (each column
%           correspond to a different output function)
%       net: struct object that gather the informations about the RVFL
%           (number of hidden node, hidden parameters and regularization
%           parameter)
%       beta: (K x m) matrix of the RVFL output weights
%
%Output: NRMSE: Normalized Root Mean Squared-Error
%        NSR: Noise-Signal Ratio

    pX=size(X_test,1);
    esp=(exp(-bsxfun(@plus,X_test*net.coeff',net.bias'))+1).^-1;
    exit=esp*beta;
    MSE=sum((Y_test - exit).^2)/pX;
    NRMSE=sqrt(MSE/var(Y_test));
    NSR=10*log10(sum((Y_test-exit).^2)/sum(Y_test.^2));
end

