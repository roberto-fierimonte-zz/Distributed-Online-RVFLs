function [soluzione,NMSE,NSR] = distributed_regression2(X,Y,rete,W,n_iter)
%DISTRIBUTED_REGRESSION definisce un algoritmo di regressione distribuito
%in cui per ogni nodo del sistema la macchina per l'apprendimento è definita 
%da una Random-Vector Functional-Link e i parametri sono determinati attraverso 
%un algortimo di consensus.
%
%Input: X: matrice p x n dei campioni di ingresso (p campioni di dimensione n)
%       Y: vettore dei campioni di uscita
%       rete:
%       W: matrice dei pesi associati al sistema distribuito (deve soddisfare
%          opportune proprietà)
%       n_iter: intero che definisce il numero di iterazioni del consensus
%
%Output: soluzione:
%        NMSE: matrice n_iter x n_nodi con elementi corrispondenti all'errore
%        NSR:

%Passo 1: estraggo le dimensioni del dataset
    pX=size(X,1);
    pY=size(Y,1);
    n_nodi=size(W,1);

%Se i campioni di ingresso e uscita sono di numero diverso restituisco un
%errore
    if pX ~= pY
        error('Il numero di campioni di ingresso (%i) è diverso da quello dei campioni in uscita (%i)',pX,pY);
    end
    
%Passo 2:
    coeff = rete.coeff;
    soglie = rete.soglie;
    K = rete.dimensione;
    lambda = rete.lambda;
    
    esp=(exp(-bsxfun(@plus,X*coeff',soglie'))+1).^-1;
    
%Inizializzo gli output a valori nulli
    beta = zeros(K,n_nodi);
    gamma = zeros(K,n_nodi);
    
%Passo 3: distribuisco i dati nel sistema
    spmd (n_nodi)
        X_dist=codistributed(X, codistributor1d(1));
        Y_dist=codistributed(Y, codistributor1d(1));
        X_local=getLocalPart(X_dist);
        Y_local=getLocalPart(Y_dist);
        
        iniziale=rvflreg(X_local,Y_local,rete);
        
%Passo 6: applico l'algoritmo del consensus per aggiornare i parametri di 
%ogni nodo
        corrente=iniziale;
        
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
        beta(:,dd)=iniziale{dd};
        gamma(:,dd)=corrente{dd};
    end
    check;
    soluzione=gamma(:,1);
    %soluzione=mean(beta,2);
    NMSE=1/(pX*var(Y))*sum(((esp*soluzione)-Y).^2);
    NSR=10*log10(sum((Y-esp*soluzione).^2)/sum(Y.^2));
end