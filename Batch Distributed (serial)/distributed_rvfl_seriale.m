function [sol,n_iter] = distributed_rvfl_seriale(X,Y,net,W,max_iter,cvpart)
%DISTRIBUTED_RVFL is a Random Vector Functional-Link learning algorithm for 
%Machine Learning problems in distributed systems, where each node estimate
%the local weights using a RVFL and then a global estimate is computed
%using a consensus algorithm on the local estimates
%THIS IS A SERIAL VERSION USED IN SIMULATIONS
%
%Input: X: (p x n) matrix of the input train patterns
%       Y: (p x m) matrix of the output train patterns (in case of 
%           multiclassification or regression problems each column
%           correspond to a class or to a different output function)
%       net: struct object that gather the informations about the RVFL
%           (number of hidden node, hidden parameters and regularization
%           parameter)
%       W: (n_nodes x n_nodes) matrix of the weights of the graph (it must
%           satisfy some conditions)
%       max_iter: max number of consensus iterations
%       cvpart: cvpartition object used for spreading data in the
%           distributed system
%
%Output: sol: (K x m) matrix of the output weights
%        n_iter: actual number of consensusiterations before the stopping 
%           criteron is satisfied

%Step 1: calculate the dimension of input and output patterns and the
%number of nodes in the graph
    pX=size(X,1);
    [pY,m]=size(Y);
    n_nodes=size(W,1);

%If the number of input patterns is different from the the number of the
%output patterns return an exception
    if pX ~= pY
        error('The numbers of input patterns (%i) and output patterns (%i) are different',pX,pY);
    end

%Step 2: if the system is composed by only one node, the global solution is
%equal to the local solution
    if n_nodes==1
        scal=X*net.coeff';
        aff=bsxfun(@plus,scal,net.bias');
        H=(exp(-aff)+1).^-1;
        
        if pX >= net.dimension
            sol = (H'*H+net.lambda*eye(net.dimension))\H'*Y;
        else
            sol = H'/(net.lambda*eye(pX)+H*H')*Y;
        end
        n_iter=0;
        
%otherwise calculate the hidden matrix for each node    
    else
        local = zeros(net.dimension,m,n_nodes);
        
        for kk=1:n_nodes
            Xlocal=X(cvpart.test(kk),:);
            Ylocal=Y(cvpart.test(kk),:);
            pX_loc=size(Xlocal,1);
            scal = Xlocal*net.coeff';
            aff = bsxfun(@plus,scal,net.bias');
            H = (exp(-aff)+1).^-1;

%Step 3: calculate the local output matrix for each node       
            if pX_loc >= net.dimension
                local(:,:,kk)= (H'*H+net.lambda*eye(net.dimension))\H'*Ylocal;
            else
                local(:,:,kk)= H'/(net.lambda*eye(pX_loc)+H*H')*Ylocal;
            end
        end

%Step 4: apply consensus algorithm on local estimates, check if consensus 
%is reached and return the global solution
        if max_iter==0
            sol=local(:,:,1);
            n_iter=0;
        else
            beta_avg_real = mean(local, 3);
            gamma=local;

            for ii = 1:max_iter
                new=gamma;
                for kk=1:n_nodes
                    temp=zeros(net.dimension,m);
                    for qq=1:n_nodes
                        temp=temp+new(:,:,qq)*W(kk,qq);
                    end
                    gamma(:,:,kk)=temp;
                end
                if all(all(all((abs(repmat(beta_avg_real, 1, 1, ...
                        size(gamma, 3)) - gamma) <= 10^-6))))
                    sol=gamma(:,:,1);
                    n_iter=ii;
                    break
                end
            end
        end
    end
end