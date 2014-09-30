function [soluzione,X_dist,NMSE,NSR] = distributed_regression3mod2(X,Y,rete,n_nodi,W,n_iter)
%DISTRIBUTED_REGRESSION definisce un algoritmo di regressione distribuito
%in cui per ogni nodo del sistema la macchina per l'apprendimento � definita 
%da una Random-Vector Functional-Link e i parametri sono determinati attraverso 
%un algortimo di consensus.
%
%Input: X: matrice p x n dei campioni di ingresso (p campioni di dimensione n)
%       Y: vettore p x 1 dei campioni di uscita
%       rete:
%       n_nodi: intero che definisce il numero di nodi del sistema distribuito
%       W: matrice dei pesi associati al sistema distribuito (deve soddisfare
%          opportune propriet�)
%       n_iter: intero che definisce il numero di iterazioni del consensus
%
%Output: soluzione:
%       NMSE: matrice n_iter x n_nodi con elementi corrispondenti all'errore
%       NSR:


%Passo 1: estraggo le dimensioni del dataset
    [pX,n]=size(X);
    pY=size(Y,1);

%Se i campioni di ingresso e uscita sono di numero diverso restituisco un
%errore
    if pX ~= pY
        error('Il numero di campioni di ingresso (%i) � diverso da quello dei campioni in uscita (%i)',pX,pY);
    end
    
%Passo 2:
    coeff = rete.coeff;
    soglie = rete.soglie;
    K = rete.dimensione;
    lambda = rete.lambda;
    
    %esp=[X tanh(bsxfun(@plus,X*coeff',soglie'))];
    esp=[X (exp(-bsxfun(@plus,X*coeff',soglie'))+1).^-1];
    
%Inizializzo gli output a valori nulli
    beta = zeros(n+K,n_nodi);
    
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
        exit = (exp(-aff)+1).^-1;
        %exit=tanh(aff);
        A=[X_local exit];
        %A = exit;
                
        P = (A'*A+lambda*eye(K+n));
        q = A'*Y_local;

%Passo 5: calcolo il vettore dei parametri relativo a ogni nodo risolvendo
%il sistema linare        
        if pX_loc >= K
            corrente= P\q;
        else
            corrente= A'/(lambda*eye(pX_loc)+A*A')*Y_local;
        end
        
%Passo 6: applico l'algoritmo del consensus per aggiornare i parametri di 
%ogni nodo
        neigh = W(labindex, :);
        neigh_idx = find(neigh > 0);
        neigh_idx(neigh_idx == labindex) = [];
        
        for ii = 1:n_iter
            
            new = neigh(labindex)*corrente;
            
            labSend(corrente, neigh_idx);
            
            for jj = 1:length(neigh_idx)
                
                while(~labProbe(neigh_idx(jj)))
                end
                    
                new = new + neigh(neigh_idx(jj))*labReceive(neigh_idx(jj));
            end
            
            corrente = new;
        end
    end
    
%Ritorno gli output  
    for dd=1:n_nodi
        beta(:,dd)=corrente{dd};
    end
    check;
    soluzione=beta(:,1);
    NMSE=1/(pX*var(Y))*sum(sum((esp*soluzione)-Y).^2);
    NSR=10*log10(sum((Y-esp*soluzione).^2)/sum(Y.^2));
end