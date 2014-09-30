function [soluzione,out,err,delta,net] = distributed_classification3(X,Y,K,lambda,n_nodi,W,n_iter)
%DISTRIBUTED_CLASSIFICATION definisce un algoritmo di classificazione distribuito
%in cui per ogni nodo del sistema la macchina per l'apprendimento � definita 
%da una Random-Vector Functional-Link e i parametri sono determinati attraverso 
%un algortimo di consensus.
%
%Input: X: matrice p x n dei campioni di ingresso (p campioni di dimensione n)
%       Y: vettore p x 1 dei campioni di uscita
%       K: intero che definisce la dimensione dell'espansione funzionale
%          della RVFL
%       lambda: scalare che definisce il parametro di regolarizzazione
%          della RVFL
%       n_nodi: intero che definisce il numero di nodi del sistema distribuito
%       W: matrice dei pesi associati al sistema distribuito (deve soddisfare
%          opportune propriet�)
%       n_iter: intero che definisce il numero di iterazioni del consensus
%
%Output: beta:
%        soluzione: vettore p x 1 con elementi corrispondenti alle uscite stimate
%        err: matrice n_iter x n_nodi con elementi corrispondenti all'errore
%             commesso nell'i-esima iterazione del consensus dal j-esimo
%             nodo
%        delta: matrice n_iter x n_nodi con elementi corrispondenti alla
%             differenza in norma della soluzione della i-esima iterazione
%             e l'iterazione precedente del j-esimo nodo
%        net: struttura contenente i dati della rete RVFL generati in modo
%             aleatorio al passo 2, la dimensione della rete e il parametro
%             di regolarizzazione

%Passo 1: estraggo le dimensioni del dataset
    [pX,n]=size(X);
    pY=size(Y,1);

%Se i campioni di ingresso e uscita sono di numero diverso restituisco un
%errore
    if pX ~= pY
        error('Il numero di campioni di ingresso (%i) � diverso da quello dei campioni in uscita (%i) :(',pX,pY);
    end
    
%Passo 2: genero casualmente K*(n+1) numeri reali che rappresentano i pesi
%e le soglie dei termini dell'espansione funzionale
    coeff = randn(K,n);
    soglie = randn(K,1);
    
    esp=tanh(bsxfun(@plus,X*coeff',soglie'));

%Ritorno la struttura della rete    
    net=struct('coeff',coeff,'soglie',soglie,'lambda',lambda,'dimensione',K);
    
%Inizializzo gli output a valori nulli
    beta = zeros(K,n_nodi);
    out = [];
    err = zeros(n_iter,n_nodi);
    delta = zeros(n_iter,n_nodi);

%Definisco delle variabili ausiliarie per l'uscita
    aus=dummyvar(Y);
        
%Passo 3: distribuisco i dati nel sistema
    spmd (n_nodi)
        X_dist=codistributed(X, codistributor1d(1));
        Y_dist=codistributed(aus, codistributor1d(1));
        X_local=getLocalPart(X_dist);
        Y_local=getLocalPart(Y_dist);
        pX_loc=size(X_local,1);
        
%Passo 4: calcolo l'uscita dell'espansione funzionale per ogni nodo del sistema       
        scal = X_local*coeff';
        aff = bsxfun(@plus,scal,soglie');
        exit=tanh(aff);
        %A=[X_local exit'];
        A = exit;
           
        P = (A'*A+lambda*eye(K));
        q = A'*Y_local;
        
%Passo 5: calcolo il vettore dei parametri relativo a ogni nodo risolvendo 
%il sistema lineare
        %if pX_loc >= K
            corrente= P\q;
        %else
        %    corrente= A'/(lambda*eye(pX_loc)+A*A')*Y_local;
        %end
        
        %err_loc=1/(pX_loc)*sum(sum((A*corrente)-Y_local).^2);

        %val_uscita=A*corrente;

%Passo 6: applico l'algoritmo del consensus per aggiornare i parametri di 
%ogni nodo
        neigh = W(labindex, :);
        neigh_idx = find(neigh > 0);
        neigh_idx(neigh_idx == labindex) = [];

        diff=zeros(1,n_iter);
        err_cons=zeros(1,n_iter);
        
        for ii = 1:n_iter

            new = neigh(labindex)*corrente;

            labSend(corrente, neigh_idx,0);

            for jj = 1:length(neigh_idx)

                while(~labProbe(neigh_idx(jj)))
                end

                new = new + neigh(neigh_idx(jj))*labReceive(neigh_idx(jj),0);
            end

            d=new;
            diff(ii) = norm(corrente-d);
            err_cons(ii) = 1/(pX)*sum(sum((vec2ind((esp*d)'))'~=Y));

            corrente = d;

        end

        uscita = (vec2ind((A*d)'))';

        %[soltest,errtest,extest]=rvflreg_test(X_local,Y_local,net);
    end
        
%Ritorno gli output  
    for dd=1:n_nodi
        %beta_distr(:,dd)=corrente{dd};
        %err_distr(:,dd)=err_loc{dd};
        %out_distr=[out_distr; val_uscita{dd}];
        %beta(:,dd)=d{dd};
        delta(:,dd)=diff{dd};
        err(:,dd)=err_cons{dd};
        out=[out;uscita{dd}];
        %prova(:,dd)=soltest{dd};
    end
end