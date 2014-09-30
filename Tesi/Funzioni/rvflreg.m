function [soluzione,NMSE,NSR] = rvflreg(X,Y,rete)
%RVFLREG definisce un algoritmo di addestramento per una rete neurale di
%tipo random vector functional-link da utilizzare in problemi di
%regressione
%
%Input: X: matrice p x n dei campioni di ingresso (p campioni di dimensione n)
%       Y: vettore dei campioni di uscita (p campioni)
%       rete:
%
%Output: soluzione: vettore dei parametri della rete RVFL
%        NMSE: scalare che definisce il NMSE della RVFL sui dati di
%        training
%        NSR: scalare che definisce il rapporto rumore segnale
    
%Passo 1: estraggo le dimensioni del dataset
    pX=size(X,1);
    pY=size(Y,1);

%Se i campioni di ingresso e uscita sono di numero diverso restituisco un
%errore
    if pX ~= pY
        error('Il numero di campioni di ingresso (%i) è diverso da quello dei campioni in uscita (%i)',pX,pY);
    end

%Passo 2: 
    coeff = rete.coeff;
    soglie = rete.soglie;

%Passo 3: determino la matrice delle uscite dell'espansione funzionale
    scal=X*coeff';
    aff = bsxfun(@plus,scal,soglie');
    exit = (exp(-aff)+1).^-1;
    A = exit;
    
%Passo 4: calcolo il vettore dei parametri risolvendo il sistema lineare e
%discriminando il procedimento in base alla dimensione del dataset
    if pX>=rete.dimensione
        soluzione=(A'*A + rete.lambda*eye(rete.dimensione))\(A'*Y);
    else        
        soluzione=A'/(rete.lambda*eye(pX)+A*A')*Y;
    end
    
%Restituisco i risultati
    NMSE=1/(pX*var(Y))*sum((A*soluzione-Y).^2);
    NSR=10*log10(sum((Y-A*soluzione).^2)/sum(Y.^2));
end
