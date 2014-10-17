function soluzione = distributed_regressionR(X,Y,rete,W,n_iter)
%DISTRIBUTED_REGRESSION definisce un algoritmo per problemi di regressione
%e classificazione binaria in sistemi distribuiti in cui per ogni nodo del 
%sistema la macchina per l'apprendimento è definita da una RVFL e i 
%parametri sono determinati attraverso un algortimo di consensus.
%
%Input: X: matrice p x n dei campioni di ingresso (p campioni di dimensione n)
%       Y: vettore dei campioni di uscita
%       rete: struttura che contiene le informazioni relative alla RVFL
%           (dimensione dell'espansione, pesi e soglie della combinazione
%           affine e parametro di regolarizzazione)
%       W: matrice dei pesi associati al sistema distribuito (deve soddisfare
%          opportune proprietà)
%       n_iter: intero che definisce il numero di iterazioni del consensus
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

%Inizializzo gli output a valori nulli
    beta = zeros(K,n_nodi);
    
%Passo 2: distribuisco i dati nel sistema
    spmd (n_nodi)
        X_dist=codistributed(X, codistributor1d(1));
        Y_dist=codistributed(Y, codistributor1d(1));
        X_local=getLocalPart(X_dist);
        Y_local=getLocalPart(Y_dist);
        
%Passo 3: calcolo l'uscita dell'espansione funzionale per ogni nodo del sistema       
        scal = X_local*rete.coeff';
        aff = bsxfun(@plus,scal,rete.soglie');
        exit = (exp(-aff)+1).^-1;
        A = exit;
        
        K = (A'*A + rete.lambda*eye(rete.dimensione));
        q = A'*Y_local;

%Passo 4: calcolo il vettore dei parametri relativo a ogni nodo risolvendo
%il sistema lineare 
        iniziale = K\q;
        
%Passo 5: applico l'algoritmo del consensus per aggiornare la matrice e il  
%vettore di stato di ogni nodo
        neigh = W(labindex, :);
        neigh_idx = find(neigh > 0);
        neigh_idx(neigh_idx == labindex) = [];
        
        for ii = 1:n_iter
            
            newK = neigh(labindex)*K;
            newq = neigh(labindex)*q;
            
            labSend(K, neigh_idx,0);
            labSend(q, neigh_idx,1);
            
            for jj = 1:length(neigh_idx)
                
                while(~labProbe(neigh_idx(jj),0))
                end
                   
                newK = newK + neigh(neigh_idx(jj))*labReceive(neigh_idx(jj),0);
                newq = newq + neigh(neigh_idx(jj))*labReceive(neigh_idx(jj),1);
            end
            
            K = newK;
            q = newq;
            
        end
        
        corrente = K\q;
    end
    
%Passo 6: controllo se i nodi hanno raggiunto il consenso e restituisco il
%vettore dei parametri
    if n_iter==0
        soluzione=iniziale{1};
    else     
        
    gamma = zeros(rete.dimensione,n_nodi); 
    beta = zeros(rete.dimensione,n_nodi);
    
        for dd=1:n_nodi
            beta(:,dd)=iniziale{dd};
            gamma(:,dd)=corrente{dd};
        end
        
        beta_avg_real = mean(beta, 2);
        assert(all(all((abs(repmat(beta_avg_real, 1, size(gamma, 2)) - gamma) <= 10^-6))), 'Errore: consenso non raggiunto :(');
        
        soluzione=gamma(:,1);
    end
end