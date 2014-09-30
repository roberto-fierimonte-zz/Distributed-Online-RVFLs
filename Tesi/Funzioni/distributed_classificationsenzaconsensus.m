function [errtest,errtrain] = distributed_classificationsenzaconsensus(X,Y,X_test,Y_test,rete,n_nodi)
%DISTRIBUTED_CLASSIFICATION definisce un algoritmo di classificazione distribuito
%
%Input: X: matrice p x n dei campioni di ingresso (p campioni di dimensione n)
%       Y: vettore dei campioni di uscita
%       rete:
%       W: matrice dei pesi associati al sistema distribuito (deve soddisfare
%          opportune proprietà)
%       n_iter: intero che definisce il numero di iterazioni del consensus
%
%Output: soluzione:
%        errore:

%Passo 1: estraggo le dimensioni del dataset
    pX=size(X,1);
    pY=size(Y,1);
    pXtest=size(X_test,1);

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
    
    esp=(exp(-bsxfun(@plus,X_test*coeff',soglie'))+1).^-1;
    
    errtest=0;
    errtrain=0;

%Definisco delle variabili ausiliarie per l'uscita
    aus=dummyvar(Y);
        
%Passo 3: distribuisco i dati nel sistema
    spmd (n_nodi)
        X_dist=codistributed(X, codistributor1d(1));
        Y_dist=codistributed(aus, codistributor1d(1));
        Y1_dist=codistributed(Y, codistributor1d(1));
        X_local=getLocalPart(X_dist);
        Y_local=getLocalPart(Y_dist);
        Y1_local=getLocalPart(Y1_dist);
        pX_loc=size(X_local,1);
        
%Passo 4: calcolo l'uscita dell'espansione funzionale per ogni nodo del sistema       
        scal = X_local*coeff';
        aff = bsxfun(@plus,scal,soglie');
        exit = (exp(-aff)+1).^-1;
        A = exit;
        
%Passo 5: calcolo il vettore dei parametri relativo a ogni nodo risolvendo 
%il sistema lineare
        if pX_loc >= K
            sol= (A'*A + lambda*eye(K))\(A'*Y_local);
        else
            sol= A'/(lambda*eye(pX_loc)+A*A')*Y_local;
        end

        loctest=1/(pXtest)*sum(sum((vec2ind((esp*sol)'))'~=Y_test));
        loctrain=1/(pX_loc)*sum(sum((vec2ind((A*sol)'))'~=Y1_local));
        
    end
        
%Ritorno gli output  
    for dd=1:n_nodi
        errtest=errtest+loctest{dd};
        errtrain=errtrain+loctrain{dd};
    end
    errtest=errtest/n_nodi;
    errtrain=errtrain/n_nodi;
end