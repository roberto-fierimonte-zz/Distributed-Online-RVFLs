function [NMSEtest,NSRtest,NMSEtrain,NSRtrain] = distributed_regressionsenzaconsensus(X,Y,X_test,Y_test,rete,n_nodi)
%DISTRIBUTED_REGRESSION definisce un algoritmo di regressione distribuito
%in cui per ogni nodo del sistema la macchina per l'apprendimento è definita 
%da una Random-Vector Functional-Link e i parametri sono determinati attraverso 
%un algortimo di consensus.
%
%Input:
%
%Output:

%Passo 1: estraggo le dimensioni del dataset
    pX=size(X,1);
    pY=size(Y,1);
    pXtest=size(X_test,1);

%Se i campioni di ingresso e uscita sono di numero diverso restituisco un
%errore
    if pX ~= pY
        error('Il numero di campioni di ingresso (%i) è diverso da quello dei campioni in uscita (%i)',pX,pY);
    end
    
%Passo 2: genero casualmente K*(n+1) numeri reali che rappresentano i pesi
%e le soglie dei termini dell'espansione funzionale
    coeff = rete.coeff;
    soglie = rete.soglie;
    K = rete.dimensione;
    lambda = rete.lambda;
    
    esp=(exp(-bsxfun(@plus,X_test*coeff',soglie'))+1).^-1;
    
%Inizializzo gli output a valori nulli
    NMSEtrain = 0;
    NSRtrain = 0;
    NMSEtest = 0;
    NSRtest = 0;
    
%Passo 3: distribuisco i dati nel sistema
    spmd (n_nodi)
        X_dist=codistributed(X, codistributor1d(1));
        Y_dist=codistributed(Y, codistributor1d(1));
        X_local=getLocalPart(X_dist);
        Y_local=getLocalPart(Y_dist);
        pX_loc=size(X_local,1);
        
%Passo 4: calcolo l'uscita dell'espansione funzionale per ogni nodo del sistema       
        scal = X_local*coeff';
        aff = bsxfun(@plus,scal,soglie');
        exit = (exp(-aff)+1).^-1;
        A = exit;
                
%Passo 5: calcolo il vettore dei parametri relativoa ogni nodo risolvendo
%il sistema linare        
        if pX_loc >= K
            sol= (A'*A + lambda*eye(K))\(A'*Y_local);
        else
            sol= A'/(lambda*eye(pX_loc)+A*A')*Y_local;
        end
        
        nmsetest=1/(pXtest*var(Y_test))*sum(sum((Y_test-esp*sol).^2));
        nsrtest=10*log10(sum((Y_test-esp*sol).^2)/sum(Y_test.^2));
        nmsetrain=1/(pX_loc*var(Y_local))*sum(sum((Y_local-A*sol).^2));
        nsrtrain=10*log10(sum((Y_local-A*sol).^2)/sum(Y_local.^2));
    end
    
%Ritorno gli output  
    for dd=1:n_nodi
        NMSEtest=NMSEtest+nmsetest{dd};
        NSRtest=NSRtest+nsrtest{dd};
        NMSEtrain=NMSEtrain+nmsetrain{dd};
        NSRtrain=NSRtrain+nsrtrain{dd};
    end
    NMSEtest=NMSEtest/n_nodi;
    NSRtest=NSRtest/n_nodi;
    NMSEtrain=NMSEtrain/n_nodi;
    NSRtrain=NSRtrain/n_nodi;
end