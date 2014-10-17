function soluzione = distributed_regressionRseriale(X,Y,rete,W,n_iter,cvpart)
%DISTRIBUTED_REGRESSION definisce un algoritmo di regressione distribuito
%in cui per ogni nodo del sistema la macchina per l'apprendimento è definita 
%da una Random-Vector Functional-Link e i parametri sono determinati attraverso 
%un algortimo di consensus.
%
%Input: X: matrice p x n dei campioni di ingresso (p campioni di dimensione n)
%       Y: vettore p x 1 dei campioni di uscita
%       rete:
%       W: matrice dei pesi associati al sistema distribuito (deve soddisfare
%          opportune proprietà)
%       n_iter: intero che definisce il numero di iterazioni del consensus
%
%Output: soluzione:


%Passo 1: estraggo le dimensioni del dataset
    pX=size(X,1);
    pY=size(Y,1);
    n_nodi=size(W,1);

%Se i campioni di ingresso e uscita sono di numero diverso restituisco un
%errore
    if pX ~= pY
        error('Il numero di campioni di ingresso (%i) è diverso da quello dei campioni in uscita (%i)',pX,pY);
    end
    
%Passo 2:
    coeff = rete.coeff;
    soglie = rete.soglie;
    K = rete.dimensione;
    lambda = rete.lambda;
    
%Inizializzo gli output a valori nulli
    P = zeros(K,K,n_nodi);
    q = zeros(K,1,n_nodi);
    
%Passo 4: calcolo l'uscita dell'espansione funzionale per ogni nodo del sistema  
    for kk=1:n_nodi
        Xlocal=X(cvpart.test(kk),:);
        Ylocal=Y(cvpart.test(kk),:);
        scal = Xlocal*coeff';
        aff = bsxfun(@plus,scal,soglie');
        exit = (exp(-aff)+1).^-1;
        A = exit;

        P(:,:,kk) = (A'*A+lambda*eye(K));
        q(:,:,kk) = A'*Ylocal;
    end
        
%Passo 6: applico l'algoritmo del consensus per aggiornare i parametri di 
%ogni nodo
    if n_iter==0
        soluzione=P(:,:,1)\q(:,:,1);
    else
        for ii=1:n_iter
            oldP=P;
            oldq=q;
            for kk=1:n_nodi
                tempP=zeros(K,K);
                tempq=zeros(K,1);
                for qq=1:n_nodi
                    tempP=tempP+oldP(:,:,qq)*W(kk,qq);
                    tempq=tempq+oldq(:,:,qq)*W(kk,qq);
                end
                P(:,:,kk)=tempP;
                q(:,:,kk)=tempq;
            end
        end
        %soluzione=P(:,:,1)\q(:,:,1);
        soluzione=(P(:,:,1)-(n_nodi-1)/n_nodi*lambda*eye(K))\q(:,:,1);
        %soluzione=(mean(P,3)-(n_nodi-1)/n_nodi*lambda*eye(K))\mean(q,3);
    end
end