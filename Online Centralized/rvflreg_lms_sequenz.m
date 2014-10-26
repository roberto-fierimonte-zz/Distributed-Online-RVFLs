function soluzione = rvflreg_lms_sequenz(X1,Y1,sol_prec,rete,mu)
%RVFLREG_LMS_SEQUENZ definisce un algoritmo di addestramento per una rete neurale di
%tipo random vector functional-link da utilizzare in problemi di regressione
%e classificazione binaria in cui siano forniti nuovi dati e si desideri 
%aggiornare la stima dei parametri attraverso una tecnica di Least-Mean
%Squares (LMS)
%
%Input: X1: matrice p1 x n dei nuovi campioni di ingresso
%       Y1: matrice p1 x n dei nuovi campioni di uscita
%       sol_prec: vettore dei parametri della rete stimati attraverso i
%           campioni già noti
%       rete: struttura che contiene le informazioni relative alla RVFL
%           (dimensione dell'espansione, pesi e soglie della combinazione
%           affine e parametro di regolarizzazione)
%       mu: scalare che definisce il passo lungo la direzione
%           dell'antigradiente
%
%Output: soluzione: vettore dei parametri del modello (K parametri)
    
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
    soluzione=sol_prec-mu*(A1'*A1*sol_prec-A1'*Y1+rete.lambda*sol_prec);
end
