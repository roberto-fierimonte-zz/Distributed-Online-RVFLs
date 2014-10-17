function soluzione = distributed_classificationseriale(X,Y,rete,W,n_iter,cvpart)
%DISTRIBUTED_CLASSIFICATION definisce un algoritmo di classificazione distribuito
%in cui per ogni nodo del sistema la macchina per l'apprendimento è definita 
%da una Random-Vector Functional-Link e i parametri sono determinati attraverso 
%un algortimo di consensus.
%
%Input: X: matrice p x n dei campioni di ingresso (p campioni di dimensione n)
%       Y: vettore dei campioni di uscita
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
    aus=dummyvar(Y);
    m=size(aus,2);

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
    beta = zeros(K,m,n_nodi);

%Definisco delle variabili ausiliarie per l'uscita
    aus=dummyvar(Y);
        
%Passo 4: calcolo l'uscita dell'espansione funzionale per ogni nodo del sistema  
    for kk=1:n_nodi
        Xlocal=X(cvpart.test(kk),:);
        Ylocal=aus(cvpart.test(kk),:);
        pX_loc=size(Xlocal,1);
        scal = Xlocal*coeff';
        aff = bsxfun(@plus,scal,soglie');
        exit = (exp(-aff)+1).^-1;
        A = exit;
        
%Passo 5: calcolo il vettore dei parametri relativo a ogni nodo risolvendo 
%il sistema lineare
        if pX_loc >= K
            beta(:,:,kk)= (A'*A+lambda*eye(K))\A'*Ylocal;
        else
            beta(:,:,kk)= A'/(lambda*eye(pX_loc)+A*A')*Ylocal;
        end
    end
    
%Passo 6: applico l'algoritmo del consensus per aggiornare i parametri di 
%ogni nodo
    if n_iter==0
        soluzione=beta(:,:,1);
    else    
        gamma=beta;
        
        for ii = 1:n_iter
            nuovo=gamma;
            for kk=1:n_nodi
                temp=zeros(K,m);
                for qq=1:n_nodi
                    temp=temp+nuovo(:,:,qq)*W(kk,qq);
                end
                gamma(:,:,kk)=temp;
            end
        end
        
        beta_avg_real = mean(beta, 3);
        assert(all(all(all((abs(repmat(beta_avg_real, 1, 1, size(gamma, 3)) - gamma) <= 10^-6)))), 'Errore: consenso non raggiunto :(');
        
        soluzione=gamma(:,:,1);
    end
end