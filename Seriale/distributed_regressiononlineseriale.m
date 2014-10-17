function [soluzione,K1] = distributed_regressiononlineseriale(K0,X1,Y1,sol_prec,rete,W,n_iter,cvpart)
%DISTRIBUTED_REGRESSIONONLINE definisce un algoritmo di regressione distribuito
%
%Input:
%
%Output:
%        

%Passo 1: estraggo le dimensioni del dataset
    pX1=size(X1,1);
    pY1=size(Y1,1);
    n_nodi=size(W,1);
    
%Se i campioni di ingresso e uscita sono di numero diverso restituisco un
%errore
    if pX1 ~= pY1
        error('Il numero dei nuovi campioni di ingresso (%i) è diverso da quello dei campioni in uscita (%i)',pX1,pY1);
    end
    
%Passo 2: genero casualmente K*(n+1) numeri reali che rappresentano i pesi
%e le soglie dei termini dell'espansione funzionale
    coeff = rete.coeff;
    soglie = rete.soglie;
    K = rete.dimensione;
    
%Inizializzo gli output a valori nulli
    beta = zeros(K,n_nodi);
        
%Passo 4: calcolo l'uscita dell'espansione funzionale per ogni nodo del sistema  
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


