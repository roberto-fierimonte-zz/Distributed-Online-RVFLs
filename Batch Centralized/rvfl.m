function beta = rvfl(X,Y,net)
%RVFL is a Random Vector Functional-Link learning algorithm for Machine
%Learning problems
%
%Input: X: (p x n) matrix of the input train patterns
%       Y: (p x m) matrix of the output train patterns (in case of 
%           multiclassification or regression problems each column
%           correspond to a class or to a different output fuction)
%       net: struct object that gather the informations about the RVFL
%           (number of hidden node, hidden parameters and regularization
%           parameter)
%
%Output: beta: (K x m) matrix of the output weights
    
%Step 1: calculate the number of input and output patterns
    pX=size(X,1);
    pY=size(Y,1);

%If the number of input patterns is different from the the number of the
%output patterns return an exception
    if pX ~= pY
        error('The numbers of input patterns (%i) and output patterns (%i) are different',pX,pY);
    end
    
%Step 2: calculate the hidden matrix
    scal=X*net.coeff';
    aff = bsxfun(@plus,scal,net.bias');
    H = (exp(-aff)+1).^-1;
    
%Step 3: return the output matrix
    if pX>=net.dimension
        beta = (H'*H + net.lambda*eye(net.dimension))\(H'*Y);
    else        
        beta = H'/(net.lambda*eye(pX)+H*H')*Y;
    end
end
