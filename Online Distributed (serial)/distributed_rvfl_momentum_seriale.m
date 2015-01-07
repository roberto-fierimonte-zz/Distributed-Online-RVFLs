function [sol,aus] = distributed_rvfl_momentum_seriale(X1,Y1,sol_prec,aus_prec,net,W,alfa,eta,...
    max_iter,cvpart)
%DISTRIBUTED_RVFL_SGD definisce un algoritmo per problemi di 
%regressione e classificazione binaria in sistemi distribuiti in cui per 
%ogni nodo del sistema la macchina per l'apprendimento è definita da una 
%RVFL, in cui siano forniti nuovi dati e si desideri aggiornare la stima dei
%parametri attraverso una tecnica di Gradiente Stocastico (SGD) e 
%successivamente di un algoritmo di Consensus
%
%Input: X1: matrice p1 x n dei nuovi campioni di ingresso
%       Y1: vettore dei nuovi campioni di uscita (p1 campioni)
%       sol_prec: vettore dei parametri della rete stimati attraverso i
%           campioni già noti (K parametri)
%       aus_prec: vettore ausiliario di dimensione K usato nella iterazione
%           precedente
%       C: costante positiva usata nel calcolo del passo
%       mu_zero: scalare che definisce il passo iniziale lungo la direzione
%           dell'antigradiente
%       rete: struttura che contiene le informazioni relative alla RVFL
%           (dimensione dell'espansione, pesi e soglie della combinazione
%           affine e parametro di regolarizzazione)
%       W: matrice dei pesi associati al sistema distribuito (deve soddisfare
%           opportune proprietà)
%       count: indice dell'iterazione corrente
%       max_iter: intero che definisce il numero di iterazioni del consensus
%       cvpart: oggetto di tipo cvpartition usato per distribuire i dati nel
%           sistema
%
%Output: soluzione: vettore dei parametri del modello (K parametri)
%        aus: vettore ausiliario di dimensione K usato nella modifica di 
%           Nesterov

%Passo 1: estraggo le dimensioni del dataset
    pX1=size(X1,1);
    [pY1,m]=size(Y1);
    n_nodes=size(W,1);
    
%Se i campioni di ingresso e uscita sono di numero diverso restituisco un
%errore
    if pX1 ~= pY1
        error('Il numero dei nuovi campioni di ingresso (%i) è diverso da quello dei campioni in uscita (%i)'...
            ,pX1,pY1);
    end
        
%Passo 4: calcolo l'uscita dell'espansione funzionale per ogni nodo del sistema  
    if n_nodes == 1
        scal=X1*net.coeff';
        aff=bsxfun(@plus,scal,net.bias');
        H1=(exp(-aff)+1).^-1;
        
        if size(X1,1)>0
%             alfa=1/norm((H1'*H1)/pX1+net.lambda*eye(net.dimension));
            sol=sol_prec-alfa*((H1'*H1*sol_prec-H1'*Y1)/pX1...
                +net.lambda*sol_prec)+eta(sol_prec-aus_prec);
            aus=sol_prec;
        else
            sol=sol_prec;
            aus=aus_prec;
        end
    else
        
        beta = zeros(net.dimension,m,n_nodes);
        
        for kk=1:n_nodes
            X1local=X1(cvpart.test(kk),:);
            Y1local=Y1(cvpart.test(kk),:);
            scal = X1local*net.coeff';
            aff1 = bsxfun(@plus,scal,net.bias');
            H1 = (exp(-aff1)+1).^-1;
        
%Passo 5: calcolo il vettore dei parametri relativo a ogni nodo risolvendo
%il sistema linare        
            if size(X1local,1)>0
%                 alfa=0.5/norm((H1'*H1)/size(X1local,1)+net.lambda*eye(net.dimension));
                beta(:,:,kk)=sol_prec-alfa*((H1'*H1*sol_prec-H1'*Y1local)...
                    /size(X1local,1)+net.lambda*sol_prec)+eta*(sol_prec-aus_prec);
            else
                beta(:,:,kk)=sol_prec;
            end
        end
    
        if max_iter==0
            sol=beta(:,:,1);
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
%                     fprintf('Convergenza raggiunta in %i iterazioni\n',ii);
                    aus=sol_prec;
                    break
                end
            end
        end
    end
end


