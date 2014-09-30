function [soluzione,errore,uscita,rete] = rvflclass_distr(X,Y,K,lambda,n_nodi)
%RVFLREG_DISTR definisce un algoritmo di addestramento per una rete neurale di
%tipo random vector functional-link da utilizzare in problemi di
%classificazione binaria distribuita
%
%Input: X: matrice p x n dei campioni di ingresso (p campioni di dimensione n)
%       Y: matrice p x m dei campioni di uscita (p campioni di dimensione m)
%       K: intero che definisce la dimensione dell'espansione funzionale
%       lambda: scalare che definisce il parametro di regolarizzazione
%       n: intero che definisce il numero di nodi del sistema distribuito
%
%Output: soluzione: Una matrice (n + K) x m dei parametri della rete
%        errore: scalare che definisce il MSE della RVFL
%        uscita: matrice p x m delle uscite stimate dalla rete RVFL
%        rete: struttura contenente i parametri della rete RVFL generati
%           in modo aleatorio al passo 2 e la dimensione dell'espansione

%Passo 1: estraggo le dimensioni del dataset
    [pX,n]=size(X);
    [pY,m]=size(Y);

%Se i campioni di ingresso e uscita sono di numero diverso restituisco un
%errore
    if pX ~= pY
        error('Il numero di campioni di ingresso (%i) è diverso da quello dei campioni in uscita (%i)',pX,pY);
    end

%Passo 2: genero casualmente K*(n+1) numeri reali che rappresentano i pesi
%e le soglie dei termini dell'espansione funzionale(uguali per tutti i
%nodi),inizializzo l'uscita a un valore vuoto
    coeff = randn(K,n);
    soglie = randn(K,1);
    soluzione=zeros(K,n_nodi);
    uscita=[];
    errore=zeros(1,n_nodi);
    
%Passo 3: determino la matrice delle uscite dell'espansione funzionale
%della rete corrispondente a ogni nodo
    spmd (n_nodi)
        X_dist=codistributed(X, codistributor1d(1));
        Y_dist=codistributed(Y, codistributor1d(1));
        X_local=getLocalPart(X_dist);
        Y_local=getLocalPart(Y_dist);
        pX_loc=size(X_local,1);
        scal = X_local*coeff';
        aff = bsxfun(@plus,scal,soglie');
        exit=tanh(aff);
        %A=[X_local exit'];
        A = exit;
        
%Passo 4: calcolo il vettore dei parametri risolvendo il sistema lineare
        if pX_loc >= K
            sol= (A'*A + lambda*eye(K))\(A'*Y_local);
        else
            sol= A'/(lambda*eye(pX_loc)+A*A')*Y_local;
        end
        
        val_uscita=A*sol;
        
        err=1/(pX_loc*m)*sum(sum(abs(val_uscita-Y_local)./2));

    end
 
%Restituisco i risultati    
    for ii=1:n_nodi
        soluzione(:,ii)=sol{ii};
        uscita=[uscita; val_uscita{ii}];
        errore(ii)=err{ii};
    end

    rete=struct('coeff',coeff,'soglie',soglie,'lambda',lambda,'dimensione',K);
end
