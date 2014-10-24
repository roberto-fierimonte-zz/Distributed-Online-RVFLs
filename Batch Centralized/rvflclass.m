function soluzione = rvflclass(X,Y,rete)
%RVFLCLASS definisce un algoritmo di addestramento per una rete neurale di
%tipo random vector functional-link da utilizzare in problemi di
%classificazione
%
%Input: X: matrice p x n dei campioni di ingresso (p campioni di dimensione n)
%       Y: vettore dei campioni di uscita (p campioni)
%       rete: struttura che contiene le informazioni relative alla RVFL
%           (dimensione dell'espansione, pesi e soglie della combinazione
%           affine e parametro di regolarizzazione)
%
%Output: soluzione: matrice K x m dei parametri del modello
    
%Passo 1: estraggo le dimensioni del dataset
    pX=size(X,1);
    pY=size(Y,1);
    
%Se i campioni di ingresso e uscita sono di numero diverso restituisco un
%errore
    if pX ~= pY
        error('Il numero di campioni di ingresso (%i) è diverso da quello dei campioni in uscita (%i)',pX,pY);
    end

%Passo 2: converto i valori della variabile di uscita in valori booleani 
%utilizzando variabili ausiiarie
    aus=dummyvar(Y);
        
%Passo 3: determino la matrice delle uscite dell'espansione funzionale
    scal=X*rete.coeff';
    aff = bsxfun(@plus,scal,rete.soglie');
    A = (exp(-aff)+1).^-1;
    
%Passo 4: calcolo la matrice dei parametri risolvendo il sistema lineare e
%discriminando il procedimento in base alla dimensione del dataset
    if pX>=rete.dimensione
        soluzione=(A'*A + rete.lambda*eye(rete.dimensione))\(A'*aus);
    else        
        soluzione=A'/(rete.lambda*eye(pX)+A*A')*aus;
    end
end
