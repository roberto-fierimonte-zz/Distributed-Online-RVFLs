function net = generate_RVFL(K,n,lambda)
%GENERA_RETE define a Random Vector Functional-Link for Machine Learning
%applications, hidden parameters (coefficients and biases) are generated
%from normal distribution in [-1,1]
%
%Input: K: number of the hidden nodes
%       n: number of features of the input patterns
%       lambda: positive paramter of regularization
%
%Output: net: struct object that gather the informations about the RVFL

    coeff = -1+2*rand(K,n);
    bias = -1+2*rand(K,1);
    net=struct('bias',bias,'coeff',coeff,'dimension',K,'lambda',lambda);
end

