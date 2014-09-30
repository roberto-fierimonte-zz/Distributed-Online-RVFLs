function [soluzione,K1] = distributed_regressiononline(K0,X1,Y1,sol_prec,rete,W,n_iter)
%DISTRIBUTED_REGRESSIONONLINE definisce un algoritmo di regressione distribuito
%
%Input:
%
%Output:
%        

%Passo 1: estraggo le dimensioni del dataset
    pX1=size(X1,1);
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
    K = rete.dimensione;
    
%Inizializzo gli output a valori nulli
    beta = zeros(K,n_nodi);
    gamma = zeros(K,n_nodi);
    
%Passo 3: distribuisco i dati nel sistema
    spmd (n_nodi)
        X1_dist=codistributed(X1, codistributor1d(1));
        Y1_dist=codistributed(Y1, codistributor1d(1));
        X1_local=getLocalPart(X1_dist);
        Y1_local=getLocalPart(Y1_dist);
        
%Passo 4: calcolo l'uscita dell'espansione funzionale per ogni nodo del sistema       
        scal1 = X1_local*coeff';
        aff1 = bsxfun(@plus,scal1,soglie');
        exit1=(exp(-aff1)+1).^-1;
        A1 = exit1;
        
        K1=(K0+A1'*A1);

        neigh = W(labindex, :);
        neigh_idx = find(neigh > 0);
        neigh_idx(neigh_idx == labindex) = [];
        
        if size(X1_local,1)>0
            iniziale=sol_prec+K1\A1'*(Y1_local-A1*sol_prec);
        else
            iniziale=sol_prec;
        end
        labBarrier;
        corrente=iniziale;
        
        for ii=1:n_iter
            
            new = neigh(labindex)*corrente;

            labSend(corrente, neigh_idx);

            for jj = 1:length(neigh_idx)

                while(~labProbe(neigh_idx(jj)))
                end

                new = new + neigh(neigh_idx(jj))*labReceive(neigh_idx(jj));
            end

            corrente=new;
            
        end
    end
    
%Ritorno gli output
    for dd=1:n_nodi
        beta(:,dd)=iniziale{dd};
        gamma(:,dd)=corrente{dd};
    end
    check;
    soluzione=gamma(:,1);
end