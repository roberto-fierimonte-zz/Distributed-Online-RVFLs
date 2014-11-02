function [soluzione,aus] = rvfl_sgd_distributed_seriale(X1,Y1,sol_prec,...
aus_prec,C,mu_zero,rete,W,count,max_iter,cvpart)
%DISTRIBUTED_REGRESSION_LMS definisce un algoritmo per problemi di 
%regressione e classificazione binaria in sistemi distribuiti in cui per 
%ogni nodo del sistema la macchina per l'apprendimento è definita da una 
%RVFL, in cui siano forniti nuovi dati e si desideri aggiornare la stima dei
%parametri attraverso una tecnica di Gradiente Stocastico (SGD) con modifica
%di Nesterov e successivamente di un algoritmo di Consensus
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
    n_nodi=size(W,1);
    
%Se i campioni di ingresso e uscita sono di numero diverso restituisco un
%errore
    if pX1 ~= pY1
        error('Il numero dei nuovi campioni di ingresso (%i) è diverso da quello dei campioni in uscita (%i)'...
            ,pX1,pY1);
    end
        
%Passo 4: calcolo l'uscita dell'espansione funzionale per ogni nodo del sistema  
    if n_nodi == 1
        scal=X1*rete.coeff';
        aff=bsxfun(@plus,scal,rete.soglie');
        A1=(exp(-aff)+1).^-1;
        
        if size(X1,1)>0
            soluzione=aus_prec-C*(mu_zero)^-count*((A1'*A1*aus_prec-...
                A1'*Y1)/size(X1,1)+ rete.lambda*aus_prec);
            aus=soluzione+count/(count+3)*(soluzione-sol_prec);
        else
            soluzione=sol_prec;
            aus=aus_prec;
        end
    else
        
        beta = zeros(rete.dimensione,m,n_nodi);
        temp = zeros(rete.dimensione,m,n_nodi);
        
        for kk=1:n_nodi
            X1local=X1(cvpart.test(kk),:);
            Y1local=Y1(cvpart.test(kk),:);
            scal = X1local*rete.coeff';
            aff1 = bsxfun(@plus,scal,rete.soglie');
            A1 = (exp(-aff1)+1).^-1;
        
%Passo 5: calcolo il vettore dei parametri relativo a ogni nodo risolvendo
%il sistema linare        
            if size(X1local,1)>0
                beta(:,:,kk)=aus_prec-C*(mu_zero)^-count*((A1'*A1*aus_prec-...
                    A1'*Y1local)/size(X1,1)+rete.lambda*aus_prec);
                temp(:,:,kk)=beta(:,:,kk)+count/(count+3)*(beta(:,:,kk)-sol_prec);
            else
                beta(:,:,kk)=sol_prec;
                temp(:,:,kk)=aus_prec;
            end
        end
    
        if max_iter==0
            soluzione=beta(:,:,1);
            aus=temp(:,:,1);
        else
            beta_avg_real = mean(beta, 3);
            temp_avg_real = mean(temp,3);
            gamma=beta;
            vu=temp;

            for ii = 1:max_iter
                nuovo=gamma;
                gamma=nuovo*W;
                nuovo2=vu;
                vu=nuovo2*W;
                if all(all((abs(repmat(beta_avg_real, 1, size(gamma, 2)) - ...
                    gamma) <= 10^-6)))
                    if all(all((abs(repmat(temp_avg_real, 1, size(vu, 2)) - ...
                        vu) <= 10^-6)))
                        soluzione=gamma(:,1);
                        aus=vu(:,1);
                        break
                    end
                end
            end
        end
        %aus=soluzione+count/(count+3)*(soluzione-sol_prec);
    end
end


