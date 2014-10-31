function [soluzione,n_iter,train_time] = distributed_classificationseriale(X,Y,rete,W,max_iter,cvpart)
%DISTRIBUTED_CLASSIFICATION definisce un algoritmo per problemi di
%classificazione in sistemi distribuiti in cui per ogni nodo del sistema la
%macchina per l'apprendimento è definita da una RVFL e i parametri sono 
%attraverso un algortimo di consensus.
%
%Input: X: matrice p x n dei campioni di ingresso (p campioni di dimensione n)
%       Y: matrice p x m dei campioni di uscita
%       rete: struttura che contiene le informazioni relative alla RVFL
%           (dimensione dell'espansione, pesi e soglie della combinazione
%           affine e parametro di regolarizzazione)
%       W: matrice dei pesi associati al sistema distribuito (deve soddisfare
%          opportune proprietà)
%       max_iter: intero che definisce il numero massimo di iterazioni del 
%           consensus
%       cvpart: oggetto di tipo cvpartition usato per distribuire i dati nel
%           sistema
%
%Output: soluzione: matrice K x m dei parametri del modello

%Passo 1: estraggo le dimensioni del dataset e converto i valori della 
%variabile di uscita in valori booleani utilizzando variabili ausiliarie
    tic;
    pX=size(X,1);
    pY=size(Y,1);
    n_nodi=size(W,1);
    aus=dummyvar(Y);
    m=size(aus,2);

%Se i campioni di ingresso e uscita sono di numero diverso restituisco un
%errore
    if pX ~= pY
        error('Il numero di campioni di ingresso (%i) è diverso da quello dei campioni in uscita (%i)',pX,pY);
    end
   
%Passo 2: calcolo l'uscita dell'espansione funzionale per ogni nodo del sistema
    if n_nodi==1
        scal=X*rete.coeff';
        aff=bsxfun(@plus,scal,rete.soglie');
        A=(exp(-aff)+1).^-1;
        
        if pX >= rete.dimensione
            soluzione = (A'*A+rete.lambda*eye(rete.dimensione))\A'*aus;
        else
            soluzione = A'/(rete.lambda*eye(pX)+A*A')*aus;
        end
    else
        
        beta = zeros(rete.dimensione,m,n_nodi);
        
        for kk=1:n_nodi
            Xlocal=X(cvpart.test(kk),:);
            Ylocal=aus(cvpart.test(kk),:);
            pX_loc=size(Xlocal,1);
            scal = Xlocal*rete.coeff';
            aff = bsxfun(@plus,scal,rete.soglie');
            A = (exp(-aff)+1).^-1;
        
%Passo 3: calcolo il vettore dei parametri relativo a ogni nodo risolvendo 
%il sistema lineare
            if pX_loc >= rete.dimensione
                beta(:,:,kk)= (A'*A+rete.lambda*eye(rete.dimensione))\A'*Ylocal;
            else
                beta(:,:,kk)= A'/(rete.lambda*eye(pX_loc)+A*A')*Ylocal;
            end
        end
    
%Passo 4: applico l'algoritmo del consensus per aggiornare i parametri di 
%ogni nodo
        if max_iter==0
            soluzione=beta(:,:,1);
        else
            beta_avg_real = mean(beta, 3);
            gamma=beta;

            for ii = 1:max_iter
                nuovo=gamma;
                for kk=1:n_nodi
                    temp=zeros(rete.dimensione,m);
                    for qq=1:n_nodi
                        temp=temp+nuovo(:,:,qq)*W(kk,qq);
                    end
                    gamma(:,:,kk)=temp;
                end
            end

            assert(all(all(all((abs(repmat(beta_avg_real, 1, 1, size(gamma, 3)) - gamma) <= 10^-6)))), 'Errore: consenso non raggiunto :(');

            soluzione=gamma(:,:,1);
        end
    end
end