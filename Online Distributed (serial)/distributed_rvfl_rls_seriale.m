function [sol,K1,n_iter] = distributed_rvfl_rls_seriale(K0,X1,Y1,sol_prec,...
    net,W,max_iter,cvpart)
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
%       cvpart: cvpartition object used for spreading data in the
%           distributed system
%
%Output: sol: (K x m) matrix of the output weights relative to the corrent 
%           iteration
%        K1: distributed pseudo-inverse (K x K) matrix relative to the 
%           current iteration
%        n_iter: actual number of consensus iterations before the stopping 
%           criteron is satisfied

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
        A1=(exp(-aff)+1).^-1;        
        K1=(K0+A1'*A1);        
        if size(X1,1)>0
            sol=sol_prec+K1\A1'*(Y1-A1*sol_prec);
        else
            sol=sol_prec;
        end
        n_iter=0;

%otherwise calculate the new hidden matrix for each node        
    else        
        beta = zeros(net.dimension,m,n_nodes);
        K1=zeros(net.dimension,net.dimension,n_nodes);        
        for kk=1:n_nodes
            X1local=X1(cvpart.test(kk),:);
            Y1local=Y1(cvpart.test(kk),:);
            scal = X1local*net.coeff';
            aff1 = bsxfun(@plus,scal,net.bias');
            exit1 = (exp(-aff1)+1).^-1;
            A1 = exit1;
            
%Step 3: return the new pseudo-inverse matrix for each node
            K1(:,:,kk)=(K0(:,:,kk)+A1'*A1);
        
%Step 4: calculate the new local output matrix for each node      
            if size(X1local,1)>0
                beta(:,:,kk)=sol_prec+K1(:,:,kk)\A1'*(Y1local-A1*sol_prec);
            else
                beta(:,:,kk)=sol_prec;
            end
        end
        
%Step 5: apply consensus algorithm on the new local estimates, check if 
%consensus is reached and return the global solution    
        if max_iter==0
            sol=beta(:,:,1);
            n_iter=0;
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
                    n_iter=ii;
                    break
                end
            end
        end
    end
end