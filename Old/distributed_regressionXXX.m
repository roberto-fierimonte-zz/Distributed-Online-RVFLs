function [soluzione,rete] = distributed_regression(X,Y,K,lambda,n_nodi,W,n_iter)
%DISTRIBUTED_REGRESSION definisce un algoritmo di regressione distribuito
%in cui per ogni nodo del sistema la macchina per l'apprendimento è definita 
%da una Random-Vector Functional-Link e i parametri sono calcolati attraverso 
%un algortimo di consensus.
%
%Input: X: matrice p x n dei campioni di ingresso (p campioni di dimensione n)
%       Y: matrice p x m dei campioni di uscita (p campioni di dimensione m)
%       K: intero che definisce la dimensione dell'espansione funzionale
%           della RVFL
%       lambda: scalare che definisce il parametro di regolarizzazione
%           della RVFL
%       n_nodi: intero che definisce il numero di nodi del sistema distribuito
%       W: matrice dei pesi associati al sistema distribuito (deve soddisfare
%           opportune proprietà)
%       n_iter: intero che definisce il numero di iterazioni del consensus
%
%Output:

%Passo 1: estraggo le dimensioni del dataset
    [pX,n]=size(X);
    [pY,m]=size(Y);

%Se i campioni di ingresso e uscita sono di numero diverso restituisco un
%errore
    if pX ~= pY
        error('Il numero di campioni di ingresso (%i) è diverso da quello dei campioni in uscita (%i)',pX,pY);
    end
    
%Passo 2: genero casualmente K*(n+1) numeri reali che rappresentano i pesi
%e le soglie dei termini dell'espansione funzionale
    coeff = randn(K,n);
    soglie = randn(K,1);
    
%Inizializzo le uscite a valori nulli
    sol_distr=zeros(K,n_nodi);
    uscita_distr=[];
    errore_distr=zeros(1,n_nodi);

%Passo 3: determino la matrice delle uscite dell'espansione funzionale
%per ogni nodo del sistema
    spmd (n_nodi)
        X_dist=codistributed(X, codistributor1d(1));
        Y_dist=codistributed(Y, codistributor1d(1));
        X_local=getLocalPart(X_dist);
        Y_local=getLocalPart(Y_dist);
        pX_loc=size(X_local,1);
        scal = X_local*coeff';
        aff = bsxfun(@plus,scal,soglie');
        exit=tanh(aff);
        %A=[X_local exit'];
        A = exit;
        
%Passo 4: calcolo il vettore dei parametri risolvendo il sistema lineare
        if pX_loc >= K
            sol= (A'*A + lambda*eye(K))\(A'*Y_local);
        else
            sol= A'/(lambda*eye(pX_loc)+A*A')*Y_local;
        end
        
        MSE_training=1/(pX_loc*m)*sum(((A*sol)-Y_local).^2);
        
        val_uscita=A*sol;
    end
    
%Restituisco i risultati parziali   
    for ii=1:n_nodi
        sol_distr(:,ii)=sol{ii};
        uscita_distr=[uscita_distr; val_uscita{ii}];
        errore_distr(ii)=MSE_training{ii};
    end
    
    spmd(n_nodi)
       
        neigh = W(labindex, :);
        neigh_idx = find(neigh > 0);
        neigh_idx(neigh_idx == labindex) = [];
        
        d = sol_distr(:,labindex);
        
        diff=zeros(1,n_iter);
        err=zeros(1,n_iter);
        
        for ii = 1:n_iter
            
            new = neigh(labindex)*d;
            labSend(d, neigh_idx);
            
            for jj = 1:length(neigh_idx)
                
                while(~labProbe(neigh_idx(jj)))
                end
                    
                new = new + neigh(neigh_idx(jj))*labReceive(neigh_idx(jj));
      
            end
           
            diff(ii) = norm(d-new);
            %err(ii) = 
            
            d = new;
        end
        
    end
    
%    x=d;
    
    for dd=1:n
        x(:,dd)=d{dd};
        delta(:,dd)=diff{dd};
    end
end

