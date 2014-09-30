function [soluzione,errore,rete] = rvflreg(X,Y,K,lambda)
%RVFLREG definisce un algoritmo di addestramento per una rete neurale di
%tipo random vector functional-link da utilizzare in problemi di
%regressione
%
%Input: X: matrice p x n dei campioni di ingresso (p campioni di dimensione n)
%       Y: matrice p x m dei campioni di uscita (p campioni di dimensione m)
%       K: scalare che definisce la dimensione dell'espansione funzionale
%       lambda: scalare che definisce il parametro di regolarizzazione
%
%Output: soluzione: matrice K x m dei parametri della rete RVFL
%        errore: scalare che definisce il MSE della RVFL
%        uscita: matrice p x m delle uscite stimate dalla rete RVFL
%        rete: struttura contenente i parametri della rete RVFL generati
%              in modo aleatorio al passo 2, la dimensione dell'espansione e
%              il parametro di regolarizzazione
    
%Passo 1: estraggo le dimensioni del dataset
    [pX,n]=size(X);
    [pY,m]=size(Y);

%Se i campioni di ingresso e uscita sono di numero diverso restituisco un
%errore
    if pX ~= pY
        error('Il numero di campioni di ingresso (%i) è diverso da quello dei campioni in uscita (%i)',pX,pY);
    end

%Passo 2: genero casualmente K*(n+1) numeri reali che rappresentano i pesi
%e le soglie dei termini dell'espansione funzionale
    coeff = randn(K,n);
    soglie = randn(K,1);

%Passo 3: determino la matrice delle uscite dell'espansione funzionale
    scal=X*coeff';
    aff = bsxfun(@plus,scal,soglie');
    exit=tanh(aff);
    %A=[X exit];
    A = exit;
    
%Passo 4: calcolo il vettore dei parametri risolvendo il sistema lineare e
%discriminando il procedimento in base alla dimensione del dataset
    if pX>=K
        sol=(A'*A + lambda*eye(K))\(A'*Y);
    else        
        sol=A'/(lambda*eye(pX)+A*A')*Y;
    end
    
%Restituisco i risultati
    soluzione=sol;
    %uscita=A*sol;
    errore=1/(pX*var(Y))*sum(sum((A*sol-Y).^2));
    rete=struct('coeff',coeff,'soglie',soglie,'lambda',lambda,'dimensione',K);
end
