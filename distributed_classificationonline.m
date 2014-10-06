function [soluzione,K1] = distributed_classificationonline(K0,X1,Y1,sol_prec,rete,W,n_iter)
%DISTRIBUTED_CLASSIFICATIONONLINE definisce un algoritmo per problemi di 
%classificazione in sistemi distribuiti in cui per ogni nodo del sistema la
%macchina per l'apprendimento è definita da una RVFL, i parametri sono 
%determinati attraverso un algortimo di consensus e in cui siano forniti 
%nuovi dati e si desideri aggiornare la stima dei parametri
%
%Input: K0: pseudoinversa (K x K) distribuita relativa all'iterazione 
%           precedente
%       X1: matrice p1 x n dei nuovi campioni di ingresso
%       Y1: matrice p1 x m dei nuovi campioni di uscita
%       sol_prec: matrice K x m dei parametri della rete stimati attraverso
%           i campioni già noti
%       rete: struttura che contiene le informazioni relative alla RVFL
%           (dimensione dell'espansione, pesi e soglie della combinazione
%           affine e parametro di regolarizzazione)
%       W: matrice dei pesi associati al sistema distribuito (deve soddisfare
%           opportune proprietà)
%       n_iter: intero che definisce il numero di iterazioni del consensus
%
%Output: soluzione: matrice K x m dei parametri del modello
%        K1: pseudoinversa (K x K) rdistribuita elativa all'iterazione 
%           corrente

%Passo 1: estraggo le dimensioni del dataset
    pX1=size(X1,1);
    pY1=size(Y1,1);
    n_nodi=size(W,1);
    aus=dummyvar(Y1);
    m=size(aus,2);

%Se i campioni di ingresso e uscita sono di numero diverso restituisco un
%errore
    if pX1 ~= pY1
        error('Il numero dei nuovi campioni di ingresso (%i) è diverso da quello dei campioni in uscita (%i)',pX1,pY1);
    end
    
%Passo 2: converto i valori della variabile di uscita in valori booleani 
%utilizzando variabili ausiiarie
    aus=dummyvar(Y1);
        
%Passo 3: distribuisco i dati nel sistema
    spmd (n_nodi)
        X1_dist=codistributed(X1, codistributor1d(1));
        Y1_dist=codistributed(aus, codistributor1d(1));
        X1_local=getLocalPart(X1_dist);
        Y1_local=getLocalPart(Y1_dist);
        
%Passo 4: calcolo l'uscita dell'espansione funzionale per ogni nodo del sistema       
        scal = X_local*rete.coeff';
        aff = bsxfun(@plus,scal,rete.soglie');
        exit = (exp(-aff)+1).^-1;
        A1 = exit;
        
        K1=(K0+A1'*A1);

        neigh = W(labindex, :);
        neigh_idx = find(neigh > 0);
        neigh_idx(neigh_idx == labindex) = [];

%Passo 5: aggiorno la soluzione utilizzando i nuovi dati        
        if size(X1_local,1)>0
            iniziale=sol_prec+K1\A1'*(Y1_local-A1*sol_prec);
        else
            iniziale=sol_prec;
        end
        labBarrier;
        
%Passo 6: applico l'algoritmo del consensus per aggiornare i parametri di 
%ogni nodo         
        corrente=iniziale;

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
        
%Passo 6: controllo se i nodi hanno raggiunto il consenso e restituisco la
%matrice dei parametri 
    if n_iter==0
        soluzione=iniziale{1};
    else
        
        beta = zeros(rete.dimensione,m,n_nodi);
        gamma = zeros(rete.dimensione,m,n_nodi); 
        
        for dd=1:n_nodi
            beta(:,:,dd)=iniziale{dd};
            gamma(:,:,dd)=corrente{dd};
        end
        
        beta_avg_real = mean(beta, 3);
        assert(all(all(all((abs(repmat(beta_avg_real, 1, 1, size(gamma, 3)) - gamma) <= 10^-6)))), 'Errore: consenso non raggiunto :(');
        
        soluzione=gamma(:,:,1);
    end
end