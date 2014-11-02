function [beta,K1] = rvfl_rls(K0,X1,Y1,beta_prec,net)
%RVFL_RLS is a Random Vector Functional-Link learning algorithm for Machine
%Learning problems in online framework using a Recursive Least Squares (RLS)
%strategy
%
%Input: K0: pseudo-inverse (K x K) matrix relative to the previous iteration
%       X1: (p1 x n) matrix of the new input train patterns
%       Y1: (p1 x m) matrix of the new output train patterns (in case of 
%           multiclassification or regression problems each column
%           correspond to a class or to a different output fuction)
%       beta_prec: (K x m) matrix of the output weights relative to the
%           previous iteration
%       net: struct object that gather the informations about the RVFL
%           (number of hidden node, hidden parameters and regularization
%           parameter)
%
%Output: beta: (K x m) matrix of the output weights relative to the corrent 
%           iteration
%        K1: pseudo-inverse (K x K) matrix relative to the current iteration
    
%Step 1: calculate the number of input and output patterns
    pX1=size(X1,1);
    pY1=size(Y1,1);

%If the number of input patterns is different from the the number of the
%output patterns return an exception
    if pX1 ~= pY1
        error('The sizes of input patterns (%i) and output patterns (%i) are different',pX1,pY1);
    end

%Step 2: calculate the new hidden matrix
    scal1 = X1*net.coeff';
    aff1 = bsxfun(@plus,scal1,net.bias');
    H1 = (exp(-aff1)+1).^-1;

%Step 3: return the new pseudo-inverse matrix    
    K1=(K0+H1'*H1);
    
%Step 4: return the new solution
    beta=beta_prec+K1\H1'*(Y1-H1*beta_prec);
end
