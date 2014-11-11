function [sol,K1,n_iter] = distributed_rvfl_rls_seriale(K0,X1,Y1,sol_prec,net,W,max_iter,cvpart)
%DISTRIBUTED_RVFL_RLS definisce un algoritmo per problemi di Machine
%Learningmin sistemi distribuiti in cui per ogni nodo del sistema la 
%macchina per l'apprendimento è definita da una RVFL, i parametri sono 
%determinati attraverso un algortimo di consensus e in cui siano forniti 
%nuovi dati e si desideri aggiornare la stima dei parametri
%
%Input: K0: pseudoinversa (K x K) distribuita relativa all'iterazione 
%           precedente
%       X1: matrice p1 x n dei nuovi campioni di ingresso
%       Y1: matrice p1 x m dei nuovi campioni di uscita
%       sol_prec: vettore dei parametri della rete stimati attraverso i
%           campioni già noti (K x m parametri)
%       rete: struttura che contiene le informazioni relative alla RVFL
%           (dimensione dell'espansione, pesi e soglie della combinazione
%           affine e parametro di regolarizzazione)
%       W: matrice dei pesi associati al sistema distribuito (deve soddisfare
%           opportune proprietà)
%       max_iter: intero che definisce il numero massimo di iterazioni del 
%           consensus
%       cvpart: oggetto di tipo cvpartition usato per distribuire i dati nel
%           sistema
%
%Output: soluzione: matrice dei parametri del modello (K x m parametri)
%        K1: pseudoinversa (K x K) distribuita elativa all'iterazione 
%           corrente   

%Passo 1: estraggo le dimensioni del dataset e del grafo
    pX1=size(X1,1);
    [pY1,m]=size(Y1);
    n_nodes=size(W,1);
    
%Se i campioni di ingresso e uscita sono di numero diverso restituisco un
%errore
    if pX1 ~= pY1
        error('Il numero dei nuovi campioni di ingresso (%i) è diverso da quello dei campioni in uscita (%i)',pX1,pY1);
    end
        
%Passo 4: calcolo l'uscita dell'espansione funzionale per ogni nodo del sistema  
    if n_nodes == 1
        scal=X1*net.coeff';
        aff=bsxfun(@plus,scal,net.bias');
        A1=(exp(-aff)+1).^-1;
        
        K1=(K0+A1'*A1);
        
        if size(X1,1)>0
            sol=sol_prec+K1\A1'*(Y1-A1*sol_prec);
        else
            sol=sol_prec;
        end
        n_iter=0;
        
    else
        
        beta = zeros(net.dimension,m,n_nodes);
        K1=zeros(net.dimension,net.dimension,n_nodes);
        
        for kk=1:n_nodes
            X1local=X1(cvpart.test(kk),:);
            Y1local=Y1(cvpart.test(kk),:);
            scal = X1local*net.coeff';
            aff1 = bsxfun(@plus,scal,net.bias');
            exit1 = (exp(-aff1)+1).^-1;
            A1 = exit1;

            K1(:,:,kk)=(K0(:,:,kk)+A1'*A1);
        
%Passo 5: calcolo il vettore dei parametri relativo a ogni nodo risolvendo
%il sistema linare        
            if size(X1local,1)>0
                beta(:,:,kk)=sol_prec+K1(:,:,kk)\A1'*(Y1local-A1*sol_prec);
            else
                beta(:,:,kk)=sol_prec;
            end
        end
    
        if max_iter==0
            sol=beta(:,:,1);
            n_iter=0;
        else
            gamma=beta;

            for ii = 1:max_iter
                new=gamma;
                for kk=1:n_nodes
                    temp=zeros(net.dimension,m);
                    for qq=1:n_nodes
                        temp=temp+new(:,:,qq)*W(kk,qq);
                    end
                    gamma(:,:,kk)=temp;
                    delta(kk)=(norm(gamma(:,:,kk)-new(:,:,kk)))^2;
                end
                if all(delta<=10^-6)
                    sol=gamma(:,:,1);
                    n_iter=ii;
                    break
                end
            end
        end
    end
end


