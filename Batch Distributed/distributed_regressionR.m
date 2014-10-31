function soluzione = distributed_regressionR(X,Y,rete,W,max_iter)
%DISTRIBUTED_REGRESSION definisce un algoritmo per problemi di regressione
%e classificazione binaria in sistemi distribuiti in cui per ogni nodo del 
%sistema la macchina per l'apprendimento è definita da una RVFL e i 
%parametri sono determinati attraverso un algortimo di consensus.
%
%Input: X: matrice p x n dei campioni di ingresso (p campioni di dimensione n)
%       Y: vettore dei campioni di uscita (p campioni)
%       rete: struttura che contiene le informazioni relative alla RVFL
%           (dimensione dell'espansione, pesi e soglie della combinazione
%           affine e parametro di regolarizzazione)
%       W: matrice dei pesi associati al sistema distribuito (deve soddisfare
%          opportune proprietà)
%       max_iter: intero che definisce il numero massimo di iterazioni del 
%           consensus
%
%Output: soluzione: vettore dei parametri del modello (K parametri)

%Passo 1: estraggo le dimensioni del dataset
    pX=size(X,1);
    pY=size(Y,1);
    n_nodi=size(W,1);

%Se i campioni di ingresso e uscita sono di numero diverso restituisco un
%errore
    if pX ~= pY
        error('Il numero di campioni di ingresso (%i) è diverso da quello dei campioni in uscita (%i)',pX,pY);
    end
    
%Passo 3: distribuisco i dati nel sistema
    spmd (n_nodi)
        X_dist=codistributed(X, codistributor1d(1));
        Y_dist=codistributed(Y, codistributor1d(1));
        X_local=getLocalPart(X_dist);
        Y_local=getLocalPart(Y_dist);
        
%Passo 4: calcolo l'uscita dell'espansione funzionale per ogni nodo del sistema       
        scal = X_local*rete.coeff';
        aff = bsxfun(@plus,scal,rete.soglie');
        A = (exp(-aff)+1).^-1;
        
        P = (A'*A+lambda*eye(rete.dimensione));
        q = A'*Y_local;

        iniziale = P\q;
        
%Passo 6: applico l'algoritmo del consensus per aggiornare i parametri di 
%ogni nodo
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
    
%Ritorno gli output  
    if max_iter==0
        soluzione=iniziale{1};
    else
        
        gamma = zeros(rete.dimensione,n_nodi); 
        beta = zeros(rete.dimensione,n_nodi);
        
        for dd=1:n_nodi
            beta(:,dd)=iniziale{dd};
            gamma(:,dd)=(newP{dd}-(n_nodi-1)/n_nodi*lambda*eye(K))\newq{dd};
        end
        
        beta_avg_real = mean(beta, 2);
        assert(all(all((abs(repmat(beta_avg_real, 1, size(gamma, 2)) - gamma) <= 10^-6))), 'Errore: consenso non raggiunto :(');
        
        soluzione=gamma(:,1);
    end
end