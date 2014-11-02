function sol = distributed_rvflR(X,Y,net,W,max_iter)
%DISTRIBUTED_RVFL is a Random Vector Functional-Link learning algorithm for 
%Machine Learning problems in distributed systems, where each node estimate
%the local weights using a RVFL and then a global estimate is computed
%using a consensus algorithm on the local status matrices and vectors
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
        error('The numbers of input patterns (%i) and output patterns (%i) are different',pX,pY);
    end
    
%Step 2: spread the data in the distributed system
    spmd (n_nodes)
        X_dist=codistributed(X, codistributor1d(1));
        Y_dist=codistributed(Y, codistributor1d(1));
        X_local=getLocalPart(X_dist);
        Y_local=getLocalPart(Y_dist);
        
%Step 3: calculate the hidden matrix for each node  
        scal = X_local*net.coeff';
        aff = bsxfun(@plus,scal,net.bias');
        H = (exp(-aff)+1).^-1;

%Step 4: calculate the local output matrix for each node 
        P = (H'*H+lambda*eye(net.dimension));
        q = H'*Y_local;

        local = P\q;
        
%Step 5: apply consensus algorithm on local status matrix and vector
        neigh = W(labindex, :);
        neigh_idx = find(neigh > 0);
        neigh_idx(neigh_idx == labindex) = [];
        
        for ii = 1:max_iter
            
            newP = neigh(labindex)*P;
            newq = neigh(labindex)*q;
            
            labSend(P, neigh_idx,0);

            
            for jj = 1:length(neigh_idx)
                
                while(~labProbe(neigh_idx(jj),0))
                end
                   
                newP = newP + neigh(neigh_idx(jj))*labReceive(neigh_idx(jj),0);
            end
            
            labSend(q, neigh_idx,1);
            
            for jj = 1:length(neigh_idx)
                
                while(~labProbe(neigh_idx(jj),1))
                end
                   
                newq = newq + neigh(neigh_idx(jj))*labReceive(neigh_idx(jj),1);
            end
            
            P = newP;
            q = newq;
            
        end
    end
    
%Step 6: check if consensus is reached and return the global solution
    if max_iter==0
        sol=local{1};
    else
        
        gamma = zeros(net.dimension,m,n_nodes); 
        beta = zeros(net.dimension,m,n_nodes);
        
        for dd=1:n_nodes
            beta(:,:,dd)=local{dd};
            gamma(:,:,dd)=newP{dd}\newq{dd};
        end
        
        beta_avg_real = mean(beta, 3);
        assert(all(all(all((abs(repmat(beta_avg_real, 1, 1, n_nodes) - ...
            gamma) <= 10^-6)))), 'Error: consensus not reached :(');

        sol=gamma(:,:,1);
    end
end