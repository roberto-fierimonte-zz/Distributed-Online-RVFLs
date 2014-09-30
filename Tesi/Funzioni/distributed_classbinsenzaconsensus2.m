function [errtest,errtrain] = distributed_classbinsenzaconsensus2(X,Y,X_test,Y_test,rete,n_nodi)
%DISTRIBUTED_CLASSBIN definisce un algoritmo di 
%
%Input: X: matrice p x n dei campioni di ingresso (p campioni di dimensione n)
%       Y: vettore p x 1 dei campioni di uscita
%       X_test:
%       Y_test:
%       rete:
%       n_iter: intero che definisce il numero di iterazioni del consensus
%
%Output: errtest:
%        errtrain:

%Passo 1: estraggo le dimensioni del dataset
    pX=size(X,1);
    pY=size(Y,1);
    pXtest=size(X_test,1);

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
    
    esp=(exp(-bsxfun(@plus,X_test*coeff',soglie'))+1).^-1;
    
    errtest=0;
    errtrain=0;
    
%Passo 3: distribuisco i dati nel sistema
    spmd (n_nodi)
        X_dist=codistributed(X, codistributor1d(1));
        Y_dist=codistributed(Y, codistributor1d(1));
        X_local=getLocalPart(X_dist);
        Y_local=getLocalPart(Y_dist);
        pX_loc=size(X_local,1);
        
        sol=rvflreg(X_local,Y_local,rete);
        
        loctest=1/(pXtest)*sum(Y_test~=sign(esp*sol));
        loctrain=1/(pX_loc)*sum(Y_local~=sign(A*sol));
    end
    
%Ritorno gli output  
    for dd=1:n_nodi
       errtest=errtest+loctest{dd};
       errtrain=errtrain+loctrain{dd};
    end
    errtest=errtest/n_nodi;
    errtrain=errtrain/n_nodi;
end