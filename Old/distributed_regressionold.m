function [beta,out,err,delta,beta_distr,out_distr,err_distr,net,prova] = distributed_regressionold2(X,Y,K,lambda,n_nodi,W,n_iter)
%DISTRIBUTED_REGRESSION definisce un algoritmo di regressione distribuito
%in cui per ogni nodo del sistema la macchina per l'apprendimento è definita 
%da una Random-Vector Functional-Link e i parametri sono determinati attraverso 
%un algortimo di consensus.
%
%Input: X: matrice p x n dei campioni di ingresso (p campioni di dimensione n)
%       Y: matrice p x m dei campioni di uscita (p campioni di dimensione m)
%       K: intero che definisce la dimensione dell'espansione funzionale
%          della RVFL
%       lambda: scalare che definisce il parametro di regolarizzazione
%          della RVFL
%       n_nodi: intero che definisce il numero di nodi del sistema distribuito
%       W: matrice dei pesi associati al sistema distribuito (deve soddisfare
%          opportune proprietà)
%       n_iter: intero che definisce il numero di iterazioni del consensus
%
%Output: beta:
%        out: matrice p x m con elementi corrispondenti alle uscite stimate
%        err: matrice n_iter x n_nodi con elementi corrispondenti all'errore
%             commesso nell'i-esima iterazione del consensus dal j-esimo
%             nodo
%        delta: matrice n_iter x n_nodi con elementi corrispondenti alla
%             differenza in norma della soluzione della i-esima iterazione
%             e l'iterazione precedente del j-esimo nodo
%        beta_distr: DEBUG
%        out_distr: DEBUG
%        err_distr: DEBUG
%        net: struttura contenente i dati della rete RVFL generati in modo
%             aleatorio al passo 2, la dimensione della rete e il parametro
%             di regolarizzazione
%        prova: DEBUG

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

%Ritorno la struttura della rete    
    net=struct('coeff',coeff,'soglie',soglie,'lambda',lambda,'dimensione',K);
    
%Inizializzo gli output a valori nulli
    beta = zeros(K,n_nodi);
    out = [];
    err = zeros(n_iter,n_nodi);
    delta = zeros(n_iter,n_nodi);
    beta_distr=zeros(K,n_nodi);
    out_distr=[];
    err_distr=zeros(1,n_nodi);
    prova=zeros(K,n_nodi);
    
%Passo 3: distribuisco i dati nel sistema
    spmd (n_nodi)
        X_dist=codistributed(X, codistributor1d(1));
        Y_dist=codistributed(Y, codistributor1d(1));
        X_local=getLocalPart(X_dist);
        Y_local=getLocalPart(Y_dist);
        pX_loc=size(X_local,1);
        
%Passo 4: calcolo l'uscita dell'espansione funzionale per ogni nodo del sistema       
        scal = X_local*coeff';
        aff = bsxfun(@plus,scal,soglie');
        exit=tanh(aff);
        %A=[X_local exit'];
        A = exit;
        
%Passo 5: calcolo il vettore dei parametri relativo a ogni nodo risolvendo 
%il sistema lineare
        if pX_loc >= K
            sol= (A'*A + lambda*eye(K))\(A'*Y_local);
        else
            sol= A'/(lambda*eye(pX_loc)+A*A')*Y_local;
        end
        
        MSE_loc=1/(pX_loc*m)*sum(sum((A*sol)-Y_local).^2);
        
        val_uscita=A*sol;

%Passo 6: applico l'algoritmo del consensus per aggiornare i parametri di 
%ogni nodo
        neigh = W(labindex, :);
        neigh_idx = find(neigh > 0);
        neigh_idx(neigh_idx == labindex) = [];
        
        P = (A'*A+lambda*eye(K));
        q = A'*Y_local;
        
        diff=zeros(1,n_iter);
        err_cons=zeros(1,n_iter);
        
        for ii = 1:n_iter
            
            newP = neigh(labindex)*P;
            newq = neigh(labindex)*q;
            
            labSend(P, neigh_idx,0);
            labSend(q, neigh_idx,1);
            
            for jj = 1:length(neigh_idx)
                
                while(~labProbe(neigh_idx(jj)))
                end
                    
                newP = newP + neigh(neigh_idx(jj))*labReceive(neigh_idx(jj),0);
                newq = newq + neigh(neigh_idx(jj))*labReceive(neigh_idx(jj),1);
            end
           
            old=P\q;
            d=(newP)\newq;
            diff(ii) = norm(old-d);
            err_cons(ii) = 1/(pX_loc*m)*sum(sum((A*d)-Y_local).^2);
            
            P = newP;
            q = newq;
            
        end
        
        uscita = A*d;
        
        [soltest,errtest,extest]=rvflreg_test(X_local,Y_local,net);
    end
    
%Ritorno gli output  
    for dd=1:n_nodi
        beta_distr(:,dd)=sol{dd};
        err_distr(:,dd)=MSE_loc{dd};
        out_distr=[out_distr; val_uscita{dd}];
        beta(:,dd)=d{dd};
        delta(:,dd)=diff{dd};
        err(:,dd)=err_cons{dd};
        out=[out;uscita{dd}];
        prova(:,dd)=soltest{dd};
    end
    
end