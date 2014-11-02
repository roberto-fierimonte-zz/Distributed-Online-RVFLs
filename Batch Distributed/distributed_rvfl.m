function sol = distributed_rvfl(X,Y,net,W,max_iter)
%DISTRIBUTED_RVFL is a Random Vector Functional-Link learning algorithm for 
%Machine Learning problems in distributed systems, where each node estimate
%the local weights using a RVFL and then a global estimate is computed
%using a consensus algorithm on the local estimates
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
%
%Output: sol: (K x m) matrix of the output weights

%Step 1: calculate the dimension of input and output patterns and the
%number of nodes in the graph
    pX=size(X,1);
    [pY,m]=size(Y);
    n_nodes=size(W,1);

%If the number of input patterns is different from the the number of the
%output patterns return an exception
    if pX ~= pY
        error('The sizes of input patterns (%i) and output patterns (%i) are different',pX,pY);
    end
    
%Step 2: spread the data in the distributed system
    spmd (n_nodes)
        X_dist=codistributed(X, codistributor1d(1));
        Y_dist=codistributed(Y, codistributor1d(1));
        X_local=getLocalPart(X_dist);
        Y_local=getLocalPart(Y_dist);
        pX_loc=size(X_local,1);
        
%Step 3: calculate the hidden matrix for each node       
        scal = X_local*net.coeff';
        aff = bsxfun(@plus,scal,net.bias');
        H = (exp(-aff)+1).^-1;

%Step 4: calculate the local output matrix for each node    
        if pX_loc >= net.dimension
            local = (H'*H + net.lambda*eye(net.dimension))\(H'*Y_local);
        else
            local = H'/(net.lambda*eye(pX_loc)+H*H')*Y_local;
        end
        
%Step 5: apply consensus algorithm on local estimates
        current=local;
        
        neigh = W(labindex, :);
        neigh_idx = find(neigh > 0);
        neigh_idx(neigh_idx == labindex) = [];
        
        for ii = 1:max_iter
            
            new = neigh(labindex)*current;
            
            labSend(current, neigh_idx);
            
            for jj = 1:length(neigh_idx)
                
                while(~labProbe(neigh_idx(jj)))
                end
                    
                new = new + neigh(neigh_idx(jj))*labReceive(neigh_idx(jj));
            end
            
            current = new;
        end
    end
    
%Step 6: check if consensus is reached and return the global solution
    if max_iter==0
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