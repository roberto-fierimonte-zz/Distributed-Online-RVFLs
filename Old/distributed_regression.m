function [soluzione,NMSE,delta,net] = distributed_regression(X,Y,K,lambda,n_nodi,W,n_iter)
%DISTRIBUTED_REGRESSION definisce un algoritmo di regressione distribuito
%in cui per ogni nodo del sistema la macchina per l'apprendimento è definita 
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
%          opportune proprietà)
%       n_iter: intero che definisce il numero di iterazioni del consensus
%
%Output: soluzione: vettore K x 1 con elementi corrispondenti ai parametri
%             della rete
%        out: vettore p x 1 con elementi corrispondenti alle uscite stimate
%        NMSE: matrice n_iter x n_nodi con elementi corrispondenti all'errore
%             in media quadratica normalizzato commesso nell'i-esima
%             iterazione del consensus dal j-esimo nodo
%        delta: matrice n_iter x n_nodi con elementi corrispondenti alla
%             differenza in norma della soluzione della i-esima iterazione
%             e l'iterazione precedente del j-esimo nodo
%        net: struttura contenente i dati della rete RVFL generati in modo
%             aleatorio al passo 2, la dimensione della rete e il parametro
%             di regolarizzazione

    fprintf('Avvio simulazione di apprendimento distribuito...\n');
%Passo 1: estraggo le dimensioni del dataset
    [pX,n]=size(X);
    pY=size(Y,1);

%Se i campioni di ingresso e uscita sono di numero diverso restituisco un
%errore
    if pX ~= pY
        error('Il numero di campioni di ingresso (%i) è diverso da quello dei campioni in uscita (%i) :(',pX,pY);
    end
    
%Passo 2: genero casualmente K*(n+1) numeri reali che rappresentano i pesi
%e le soglie dei termini dell'espansione funzionale
    fprintf('Inizializzo la rete RVFL...\nGenero casualmente i pesi e le soglie dell espansione funzionale\n');
    tic;
    coeff = randn(K,n);
    soglie = randn(K,1);
    
    esp=tanh(bsxfun(@plus,X*coeff',soglie'));

%Ritorno la struttura della rete    
    net=struct('coeff',coeff,'soglie',soglie,'lambda',lambda,'dimensione',K);
    fprintf('Rete inizializzata, trascorsi %.2f secondi\n', toc); 
    
%Inizializzo gli output a valori nulli
    beta = zeros(K,n_nodi);
    %out = [];
    NMSE = zeros(n_iter,n_nodi);
    delta = zeros(n_iter,n_nodi);
    
%Passo 3: distribuisco i dati nel sistema
    fprintf('Calcolo dei parametri in corso...\n');
    tic;
    spmd (n_nodi)
        X_dist=codistributed(X, codistributor1d(1));
        Y_dist=codistributed(Y, codistributor1d(1));
        X_local=getLocalPart(X_dist);
        Y_local=getLocalPart(Y_dist);
        
%Passo 4: calcolo l'uscita dell'espansione funzionale per ogni nodo del sistema       
        scal = X_local*coeff';
        aff = bsxfun(@plus,scal,soglie');
        exit=tanh(aff);
        %A=[X_local exit'];
        A = exit;
                
        P = (A'*A+lambda*eye(K));
        q = A'*Y_local;

%Passo 5: calcolo il vettore dei parametri relativoa ogni nodo risolvendo
%il sistema linare        
        %if pX_loc >= K
            corrente= P\q;
        %else
        %    corrente= A'/(lambda*eye(pX_loc)+A*A')*Y_local;
        %end
        
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
            err_cons(ii) = 1/(pX*var(Y))*sum(sum((esp*d)-Y).^2);
            
            corrente = d;
        end
        
        uscita = A*d;
    end
    
%Ritorno gli output  
    fprintf('Calcolo dei parametri completato,trascorsi %.2f secondi\n', toc);
    for dd=1:n_nodi
        beta(:,dd)=d{dd};
        delta(:,dd)=diff{dd};
        NMSE(:,dd)=err_cons{dd};
        %out=[out;uscita{dd}];
    end
    check;
    soluzione=beta(:,1);
end