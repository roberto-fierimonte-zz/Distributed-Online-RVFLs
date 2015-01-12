function sol = distributed_rvfl_lms_seriale(X1,Y1,sol_prec,net,W,...
    max_iter,cvpart,count)
%DISTRIBUTED_RVFL_LMS is a Random Vector Functional-Link learning algorithm 
%for Machine Learning problems in distributed systems in online framework,
%where each node estimate the local weights using a Least Mean Square
%(LMS) strategy, and then a global estimate is computed using a consensus 
%algorithm on the local estimates
%THIS IS A SERIAL VERSION USED IN SIMULATIONS
%
%Input: X1: (p1 x n) matrix of the new input train patterns
%       Y1: (p1 x m) matrix of the new output train patterns (in case of 
%           multiclassification or regression problems each column
%           correspond to a class or to a different output fuction)
%       sol_prec: (K x m) matrix of the output weights relative to the
%           previous iteration
%       net: struct object that gather the informations about the RVFL
%           (number of hidden node, hidden parameters and regularization
%           parameter)
%       W: (n_nodes x n_nodes) matrix of the weights of the graph (it must
%           satisfy some conditions)
%       max_iter: max number of consensus iterations
%       cvpart: cvpartition object used for spreading data in the
%           distributed system
%       count: index of the current iteration
%
%Output: sol: (K x m) matrix of the output weights relative to the corrent 
%           iteration

%Step 1: calculate the dimension of input and output patterns and the
%number of nodes in the graph
    pX1=size(X1,1);
    [pY1,m]=size(Y1);
    n_nodes=size(W,1);
    
%If the number of input patterns is different from the the number of the
%output patterns return an exception
    if pX1 ~= pY1
        error('The numbers of input patterns (%i) and output patterns (%i) are different',pX1,pY1);
    end
        
%Step 2: if the system is composed by only one node, the new global solution 
%is equal to the new local solution 
    if n_nodes == 1
        
        scal=X1*net.coeff';
        aff=bsxfun(@plus,scal,net.bias');
        H1=(exp(-aff)+1).^-1; 
        
        if size(X1,1)>0
            alfa=0.01/count;
            sol=sol_prec-alfa*((H1'*H1*sol_prec-H1'*Y1)/pX1...
                +net.lambda*sol_prec);
        else
            sol=sol_prec;
        end
        
    else

%otherwise calculate the new hidden matrix for each node
        beta = zeros(net.dimension,m,n_nodes); 
        
        for kk=1:n_nodes
            
            X1local=X1(cvpart.test(kk),:);
            Y1local=Y1(cvpart.test(kk),:);
            scal = X1local*net.coeff';
            aff1 = bsxfun(@plus,scal,net.bias');
            
            H1 = (exp(-aff1)+1).^-1;
        
%Step 3: calculate the new local output matrix for each node     
            if size(X1local,1)>0
                alfa=0.01/count;
                beta(:,:,kk)=sol_prec-alfa*((H1'*H1*sol_prec-H1'*Y1local)...
                    /size(X1local,1)+net.lambda*sol_prec);
            else
                beta(:,:,kk)=sol_prec;
            end
        end
        
%Step 4: apply consensus algorithm on the new local estimates, check if 
%consensus is reached and return the global solution        
        if max_iter==0
            sol=beta(:,:,1);
        else
            gamma=beta;
            
            for ii = 1:max_iter
                new=gamma;
                for kk=1:n_nodes
                    temp=zeros(net.dimension,m);
                    for qq=1:n_nodes
                        temp=temp+new(:,:,qq)*W(kk,qq);
                    end
                    
                    gamma(:,:,kk)=temp;
                    delta(kk)=(norm(gamma(:,:,kk)-new(:,:,kk)))^2;
                end
                if all(delta<=10^-6)
                    sol=gamma(:,:,1);
                    break
                end
            end
        end
    end
end