function [sol,K1] = distributed_rvfl_rls(K0,X1,Y1,sol_prec,net,W,n_iter)
%DISTRIBUTED_RVFL_RLS is a Random Vector Functional-Link learning algorithm 
%for Machine Learning problems in distributed systems in online framework,
%where each node estimate the local weights using a Recursive Least Squares
%(RLS) strategy, and then a global estimate is computed using a consensus 
%algorithm on the local estimates
%
%Input: K0: distributed pseudo-inverse (K x K) matrix relative to the 
%           previous iteration
%       X1: (p1 x n) matrix of the new input train patterns
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
%
%Output: sol: (K x m) matrix of the output weights relative to the corrent 
%           iteration
%        K1: distributed pseudo-inverse (K x K) matrix relative to the 
%           current iteration

%Step 1: calculate the dimension of input and output patterns and the
%number of nodes in the graph
    pX1=size(X1,1);
    [pY1,m]=size(Y1);
    n_nodes=size(W,1);
    
%If the number of input patterns is different from the the number of the
%output patterns return an exception
    if pX1 ~= pY1
        error('The sizes of input patterns (%i) and output patterns (%i) are different',pX1,pY1);
    end
    
%Step 2: spread the data in the distributed system
    spmd (n_nodes)
        X1_dist=codistributed(X1, codistributor1d(1));
        Y1_dist=codistributed(Y1, codistributor1d(1));
        X1_local=getLocalPart(X1_dist);
        Y1_local=getLocalPart(Y1_dist);
        
%Step 3: calculate the new hidden matrix for each node      
        scal1 = X1_local*net.coeff';
        aff1 = bsxfun(@plus,scal1,net.bias');
        H1=(exp(-aff1)+1).^-1;

%Step 4: return the new pseudo-inverse matrix for each node
        K1=(K0+H1'*H1);

%Step 5: calculate the new local output matrix for each node
        if size(X1_local,1)>0
            local=sol_prec+K1\H1'*(Y1_local-H1*sol_prec);
        else
            local=sol_prec;
        end
        labBarrier;
        
%Step 6: apply consensus algorithm on local estimates  
        neigh = W(labindex, :);
        neigh_idx = find(neigh > 0);
        neigh_idx(neigh_idx == labindex) = [];
        
        current=local;
        
        for ii=1:n_iter
            
            new = neigh(labindex)*current;

            labSend(current, neigh_idx);

            for jj = 1:length(neigh_idx)

                while(~labProbe(neigh_idx(jj)))
                end

                new = new + neigh(neigh_idx(jj))*labReceive(neigh_idx(jj));
            end

            current=new;
            
        end
    end
    
%Step 7: check if consensus is reached and return the global solution
    if n_iter==0
        sol=local{1};
    else
        
        beta = zeros(net.dimension,m,n_nodes);
        gamma = zeros(net.dimension,m,n_nodes);  
        
        for dd=1:n_nodes
            beta(:,:,dd)=local{dd};
            gamma(:,:,dd)=current{dd};
        end
        
        beta_avg_real = mean(beta, 3);
        assert(all(all(all((abs(repmat(beta_avg_real, 1, 1, n_nodes) - ...
            gamma) <= 10^-6)))), 'Error: consensus not reached :(');

        sol=gamma(:,:,1);
    end
end