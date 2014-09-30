function [soluzione,K1] = rvflreg_sequenz(K0,X1,Y1,sol_prec,rete)
%RVFLREG_SEQUENZ definisce un algoritmo di addestramento per una rete neurale di
%tipo random vector functional-link da utilizzare in problemi di regressione
%in cui siano forniti nuovi dati e si desideri aggiornare la stima dei
%parametri
%
%Input: K0:
%       X1: matrice p1 x n dei nuovi campioni di ingresso
%       Y1: matrice p1 x n dei nuovi campioni di uscita
%       sol_prec: vettore (K + n) x m dei parametri della rete
%           stimati attraverso i campioni già noti
%       rete: struttura contenente i parametri aleatori della rete
%
%Output: soluzione: matrice (n + K) x m dei parametri della rete RVFL
%        K1:
    
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
    exit1 = (exp(-aff1)+1).^-1;
    A1 = exit1;
    
    K1=(K0+A1'*A1);
    
%Restituisco i risultati
    soluzione=sol_prec+K1\A1'*(Y1-A1*sol_prec);
end
