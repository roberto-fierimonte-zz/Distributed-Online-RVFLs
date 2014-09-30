function [soluzione,K1] = distributed_regression3online2(K0,X1,Y1,sol_prec,rete,W,n_iter)
%DISTRIBUTED_REGRESSION definisce un algoritmo di regressione distribuito
%in cui per ogni nodo del sistema la macchina per l'apprendimento è definita 
%da una Random-Vector Functional-Link e i parametri sono determinati attraverso 
%un algortimo di consensus.
%
%Input: 
%       n_nodi: intero che definisce il numero di nodi del sistema distribuito
%       W: matrice dei pesi associati al sistema distribuito (deve soddisfare
%          opportune proprietà)
%       n_iter: intero che definisce il numero di iterazioni del consensus
%
%Output: beta:
%        out: vettore p x 1 con elementi corrispondenti alle uscite stimate
%        NMSE: matrice n_iter x n_nodi con elementi corrispondenti all'errore
%             in media quadratica normalizzato commesso nell'i-esima
%             iterazione del consensus dal j-esimo nodo
%        delta: matrice n_iter x n_nodi con elementi corrispondenti alla
%             differenza in norma della soluzione della i-esima iterazione
%             e l'iterazione precedente del j-esimo nodo
%        net: struttura contenente i dati della rete RVFL generati in modo
%             aleatorio al passo 2, la dimensione della rete e il parametro
%             di regolarizzazione

%Passo 1: estraggo le dimensioni del dataset
    [pX1,n]=size(X1);
    pY1=size(Y1,1);
    n_nodi=size(W,1);
    
%Se i campioni di ingresso e uscita sono di numero diverso restituisco un
%errore
    if pX1 ~= pY1
        error('Il numero dei nuovi campioni di ingresso (%i) è diverso da quello dei campioni in uscita (%i)',pX1,pY1);
    end
    
%Passo 2: genero casualmente K*(n+1) numeri reali che rappresentano i pesi
%e le soglie dei termini dell'espansione funzionale
    coeff = rete.coeff;
    soglie = rete.soglie;
    
%Inizializzo gli output a valori nulli
    beta = zeros(rete.dimensione+n,n_nodi);
    
%Passo 3: distribuisco i dati nel sistema
    spmd (n_nodi)
        X1_dist=codistributed(X1, codistributor1d(1));
        Y1_dist=codistributed(Y1, codistributor1d(1));
        X1_local=getLocalPart(X1_dist);
        Y1_local=getLocalPart(Y1_dist);
        
%Passo 4: calcolo l'uscita dell'espansione funzionale per ogni nodo del sistema       
        scal1 = X1_local*coeff';
        aff1 = bsxfun(@plus,scal1,soglie');
        %exit1=tanh(aff1);
        exit1=(exp(-aff1)+1).^-1;
        A1=[X1_local exit1];
        %A1 = exit1;
        
        K1=(K0+A1'*A1);

        neigh = W(labindex, :);
        neigh_idx = find(neigh > 0);
        neigh_idx(neigh_idx == labindex) = [];
        
        if size(X1_local,1)>0
            soluzione=sol_prec+K1\A1'*(Y1_local-A1*sol_prec);
        else
            soluzione=sol_prec;
        end
        labBarrier;
        
        for ii=1:n_iter
            
            new = neigh(labindex)*soluzione;

            labSend(soluzione, neigh_idx);

            for jj = 1:length(neigh_idx)

                while(~labProbe(neigh_idx(jj)))
                end

                new = new + neigh(neigh_idx(jj))*labReceive(neigh_idx(jj));
            end

            d=new;
            
            soluzione = d;
        end
    end
    
%Ritorno gli output  
    for dd=1:n_nodi
        beta(:,dd)=d{dd};
    end
    check;
    soluzione=beta(:,1);
end