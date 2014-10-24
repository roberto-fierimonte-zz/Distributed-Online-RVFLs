function soluzione = distributed_regressionRseriale(X,Y,rete,W,n_iter,cvpart)
%DISTRIBUTED_REGRESSION definisce un algoritmo per problemi di regressione
%e classificazione binaria in sistemi distribuiti in cui per ogni nodo del 
%sistema la macchina per l'apprendimento è definita da una RVFL e i 
%parametri sono determinati attraverso un algortimo di consensus.
%
%Input: X: matrice p x n dei campioni di ingresso (p campioni di dimensione n)
%       Y: vettore dei campioni di uscita (p campioni)
%       rete: struttura che contiene le informazioni relative alla RVFL
%           (dimensione dell'espansione, pesi e soglie della combinazione
%           affine e parametro di regolarizzazione)
%       W: matrice dei pesi associati al sistema distribuito (deve soddisfare
%          opportune proprietà)
%       n_iter: intero che definisce il numero di iterazioni del consensus
%       cvpart: oggetto di tipo cvpartition usato per distribuire i dati nel
%           sistema
%
%Output: soluzione: vettore dei parametri del modello (K parametri)


%Passo 1: estraggo le dimensioni del dataset
    pX=size(X,1);
    pY=size(Y,1);
    n_nodi=size(W,1);

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
        
        if pX >= K
            soluzione = (A'*A+rete.lambda*eye(rete.dimensione))\A'*Y;
        else
            soluzione = A'/(rete.lambda*eye(pX)+A*A')*Y;
        end
    else
        
        P = zeros(rete.dimensione,rete.dimensione,n_nodi);
        q = zeros(rete.dimensione,n_nodi);
        beta=zeros(rete.dimensione,n_nodi);
        
        for kk=1:n_nodi
            Xlocal=X(cvpart.test(kk),:);
            Ylocal=Y(cvpart.test(kk),:);
            scal = Xlocal*rete.coeff';
            aff = bsxfun(@plus,scal,rete.soglie');
            A = (exp(-aff)+1).^-1;

            P(:,:,kk) = (A'*A+lambda*eye(rete.dimensione));
            q(:,kk) = A'*Ylocal;
            
            beta(:,kk) = P(:,:,kk)\q(:,:,kk);
        end
        
%Passo 3: applico l'algoritmo del consensus per aggiornare i parametri di 
%ogni nodo
        if n_iter==0
            soluzione=P(:,:,1)\q(:,1);
        else

            gamma=zeros(rete.dimensione,n_nodi);

            for ii=1:n_iter
                oldP=P;
                oldq=q;
                for kk=1:n_nodi
                    tempP=zeros(K,K);
                    tempq=zeros(K,1);
                    for qq=1:n_nodi
                        tempP=tempP+oldP(:,:,qq)*W(kk,qq);
                        tempq=tempq+oldq(:,qq)*W(kk,qq);
                    end
                    P(:,:,kk)=tempP;
                    q(:,kk)=tempq;
                end
            end

            for kk=1:n_nodi
                gamma(:,kk)=(P(:,:,kk)-(n_nodi-1)/n_nodi*rete.lambda*eye(rete.dimensione))\q(:,kk);
            end

            beta_avg_real = mean(beta, 2);
            assert(all(all((abs(repmat(beta_avg_real, 1, size(gamma, 2)) - gamma) <= 10^-6))), 'Errore: consenso non raggiunto :(');

            soluzione=gamma(:,1);
        end
    end
end