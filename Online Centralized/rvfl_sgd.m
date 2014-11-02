function [soluzione,aus,err] = rvfl_sgd(X,Y,X1,Y1,alfa,sol_prec,aus_prec,err_prec,count,rete)
%RVFL_RLS definisce un algoritmo di addestramento per una rete neurale di
%tipo random vector functional-link da utilizzare in problemi di Machine
%Learning in cui siano forniti nuovi dati e si desideri aggiornare la stima 
%dei parametri attraverso una tecnica di Recursive Least Squares (RLS)
%
%Input: K0: pseudoinversa (K x K) relativa all'iterazione precedente
%       X1: matrice p1 x n dei nuovi campioni di ingresso
%       Y1: matrice p1 x m dei nuovi campioni di uscita
%       sol_prec: matrice K x m dei parametri della rete stimati attraverso
%           i campioni già noti
%       rete: struttura che contiene le informazioni relative alla RVFL
%           (dimensione dell'espansione, pesi e soglie della combinazione
%           affine e parametro di regolarizzazione)
%
%Output: soluzione: matrice dei parametri del modello (K x m parametri)
%        K1: pseudoinversa (K x K) relativa all'iterazione corrente
    
%Passo 1: estraggo le dimensioni del nuovo dataset
    pX1=size(X1,1);
    pY1=size(Y1,1);

%Se i campioni di ingresso e uscita sono di numero diverso restituisco un
%errore
    if pX1 ~= pY1
        error('Il numero dei nuovi campioni di ingresso (%i) è diverso da quello dei campioni in uscita (%i)',pX1,pY1);
    end

%Passo 3: determino la matrice delle uscite dell'espansione funzionale
%relativamente ai nuovi dati
    scal1 = X1*rete.coeff';
    aff1 = bsxfun(@plus,scal1,rete.soglie');
    A1 = (exp(-aff1)+1).^-1;
    
%Passo 4: aggiorno la soluzione utilizzando i nuovi dati
    temp=aus_prec-alfa*((A1'*A1*aus_prec-A1'*Y1)/size(X1,1)+...
                rete.lambda*aus_prec);
    esp=(exp(-bsxfun(@plus,X*(rete.coeff)',rete.soglie'))+1).^-1;
    uscita=sign(esp*temp);
    err=1/(size(X,1))*sum(uscita~=Y);
    
    while err>err_prec
        alfa=alfa/2;
        temp=aus_prec-alfa*((A1'*A1*aus_prec-A1'*Y1)/size(X1,1)+...
                rete.lambda*aus_prec);
        uscita=sign(esp*temp);
        err=1/(size(X,1))*sum(uscita~=Y);
    end
    soluzione=temp;
    aus=soluzione+count/(count+3)*(soluzione-sol_prec);
end
