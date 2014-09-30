function [soluzione,K1] = rvflreg_sequenz(X0,Y0,X1,Y1,sol_prec,rete)
%RVFLREG_SEQUENZ definisce un algoritmo di addestramento per una rete neurale di
%tipo random vector functional-link da utilizzare in problemi di regressione
%in cui siano forniti nuovi dati e si desideri aggiornare la stima dei
%parametri
%
%Input: X0: matrice p0 x n dei campioni di ingresso (p0 campioni di dimensione n) 
%           già noti alla rete
%       Y0: matrice p0 x m dei campioni di uscita (p0 campioni di dimensione m)
%           già noti alla rete
%       X1: matrice p1 x n dei nuovi campioni di ingresso
%       Y1: matrice p1 x n dei nuovi campioni di uscita
%       sol_prec: vettore (K + n) x m dei parametri della rete
%           stimati attraverso i campioni già noti
%       rete: struttura contenente i parametri aleatori della rete
%
%Output: soluzione: matrice (n + K) x m dei parametri della rete RVFL
%        errore: scalare che definisce il MSE della RVFL
%        uscita: matrice p x m delle uscite stimate dalla rete RVFL
    
%Passo 1: estraggo le dimensioni del nuovo dataset
    [pX1]=size(X1,1);
    [pY1]=size(Y1,1);
    [pX0]=size(X0,1);
    [pY0]=size(Y0,1);

%Se i campioni di ingresso e uscita sono di numero diverso restituisco un
%errore
    if pX1 ~= pY1
        error('Il numero dei nuovi campioni di ingresso (%i) è diverso da quello dei campioni in uscita (%i)',pX1,pY1);
    end
    
    if pX0 ~= pY0
        error('Il numero dei vecchi campioni di ingresso (%i) è diverso da quello dei campioni in uscita (%i)',pX0,pY0);
    end

%Passo 2: estraggo i parametri aleatori della rete dall'input
    coeff= rete.coeff;
    soglie = rete.soglie;

%Passo 3: determino la matrice delle uscite dell'espansione funzionale
    scal0 = X0*coeff';
    aff0 = bsxfun(@plus,scal0,soglie');
    exit0 = (exp(-aff0)+1).^-1;
    %exit0=tanh(aff0);
    %A0=[X0 exit0];
    A0 = exit0;
    scal1 = X1*coeff';
    aff1 = bsxfun(@plus,scal1,soglie');
    %exit1=tanh(aff1);
    exit1 = (exp(-aff1)+1).^-1;
    %A1=[X1 exit1];
    A1 = exit1;
    
    K0=(A0'*A0 + rete.lambda*eye(rete.dimensione));
    
    %P0=sol_prec/(A0'*Y0);
    %P0=inv(K0);
    %P1=P0-P0*A1'/(eye(pX1)+A1*P0*A1')*A1*P0;
    %P1=inv(K0+A1'*A1);
    
%Restituisco i risultati
    soluzione=sol_prec+(K0+A1'*A1)\A1'*(Y1-A1*sol_prec);
end
