function [soluzione,K1] = distributed_regressiononlineseriale(K0,X1,Y1,sol_prec,rete,W,n_iter,cvpart)
%DISTRIBUTED_REGRESSIONONLINE definisce un algoritmo per problemi di 
%regressione e classificazione binaria in sistemi distribuiti in cui per 
%ogni nodo del sistema la macchina per l'apprendimento è definita da una 
%RVFL, i parametri sono determinati attraverso un algortimo di consensus
%e in cui siano forniti nuovi dati e si desideri aggiornare la stima dei
%parametri
%
%Input: K0: pseudoinversa (K x K) distribuita relativa all'iterazione 
%           precedente
%       X1: matrice p1 x n dei nuovi campioni di ingresso
%       Y1: vettore dei nuovi campioni di uscita (p1 campioni)
%       sol_prec: vettore dei parametri della rete stimati attraverso i
%           campioni già noti (K parametri)
%       rete: struttura che contiene le informazioni relative alla RVFL
%           (dimensione dell'espansione, pesi e soglie della combinazione
%           affine e parametro di regolarizzazione)
%       W: matrice dei pesi associati al sistema distribuito (deve soddisfare
%           opportune proprietà)
%       n_iter: intero che definisce il numero di iterazioni del consensus
%       cvpart: oggetto di tipo cvpartition usato per distribuire i dati nel
%           sistema
%
%Output: soluzione: vettore dei parametri del modello (K parametri)
%        K1: pseudoinversa (K x K) distribuita elativa all'iterazione 
%           corrente   

%Passo 1: estraggo le dimensioni del dataset
    pX1=size(X1,1);
    pY1=size(Y1,1);
    n_nodi=size(W,1);
    
%Se i campioni di ingresso e uscita sono di numero diverso restituisco un
%errore
    if pX1 ~= pY1
        error('Il numero dei nuovi campioni di ingresso (%i) è diverso da quello dei campioni in uscita (%i)',pX1,pY1);
    end
        
%Passo 4: calcolo l'uscita dell'espansione funzionale per ogni nodo del sistema  
    if n_nodi == 1
        scal=X1*rete.coeff';
        aff=bsxfun(@plus,scal,rete.soglie');
        A1=(exp(-aff)+1).^-1;
        
        K1=(K0+A1'*A1);
        
        if size(X1,1)>0
            soluzione=sol_prec+K1\A1'*(Y1-A1*sol_prec);
        else
            soluzione=sol_prec;
        end
    else
        
        beta = zeros(rete.dimensione,n_nodi);
        K1=zeros(rete.dimensione,rete.dimensione,n_nodi);
        
        for kk=1:n_nodi
            X1local=X1(cvpart.test(kk),:);
            Y1local=Y1(cvpart.test(kk),:);
            scal = X1local*coeff';
            aff1 = bsxfun(@plus,scal,soglie');
            exit1 = (exp(-aff1)+1).^-1;
            A1 = exit1;

            K1(:,:,kk)=(K0(:,:,kk)+A1'*A1);
        
%Passo 5: calcolo il vettore dei parametri relativo a ogni nodo risolvendo
%il sistema linare        
            if size(X1local,1)>0
                beta(:,kk)=sol_prec+K1(:,:,kk)\A1'*(Y1local-A1*sol_prec);
            else
                beta(:,kk)=sol_prec;
            end
        end
    
        if n_iter==0
            soluzione=beta(:,1);
        else
            gamma=beta;

            for ii = 1:n_iter
                    nuovo=gamma;
                    gamma=nuovo*W;
            end

            check;

            soluzione=gamma(:,1);
        end
    end
end


