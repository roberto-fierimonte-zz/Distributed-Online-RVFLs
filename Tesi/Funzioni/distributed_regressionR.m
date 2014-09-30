function [soluzione,NMSE,NSR] = distributed_regressionR(X,Y,rete,W,n_iter)
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
    pX=size(X,1);
    pY=size(Y,1);
    n_nodi=size(W,1);

%Se i campioni di ingresso e uscita sono di numero diverso restituisco un
%errore
    if pX ~= pY
        error('Il numero di campioni di ingresso (%i) è diverso da quello dei campioni in uscita (%i)',pX,pY);
    end
    
%Passo 2: genero casualmente K*(n+1) numeri reali che rappresentano i pesi
%e le soglie dei termini dell'espansione funzionale
    coeff = rete.coeff;
    soglie = rete.soglie;
    K = rete.dimensione;
    lambda = rete.lambda;
    
    esp=(exp(-bsxfun(@plus,X*coeff',soglie'))+1).^-1;

%Inizializzo gli output a valori nulli
    beta = zeros(K,n_nodi);
    
%Passo 3: distribuisco i dati nel sistema
    spmd (n_nodi)
        X_dist=codistributed(X, codistributor1d(1));
        Y_dist=codistributed(Y, codistributor1d(1));
        X_local=getLocalPart(X_dist);
        Y_local=getLocalPart(Y_dist);
        
%Passo 4: calcolo l'uscita dell'espansione funzionale per ogni nodo del sistema       
        scal = X_local*coeff';
        aff = bsxfun(@plus,scal,soglie');
        exit = (exp(-aff)+1).^-1;
        A = exit;
        
        P = (A'*A+lambda*eye(K));
        q = A'*Y_local;

%Passo 6: applico l'algoritmo del consensus per aggiornare i parametri di 
%ogni nodo
        neigh = W(labindex, :);
        neigh_idx = find(neigh > 0);
        neigh_idx(neigh_idx == labindex) = [];
        
        for ii = 1:n_iter
            
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
        d=(newP)\newq;
    end
    
%Ritorno gli output  
    for dd=1:n_nodi
        beta(:,dd)=d{dd};
    end
    %check;
    soluzione=beta(:,1);
    NMSE=1/(pX*var(Y))*sum(sum((esp*soluzione)-Y).^2);
    NSR=10*log10(sum((Y-esp*soluzione).^2)/sum(Y.^2));
end