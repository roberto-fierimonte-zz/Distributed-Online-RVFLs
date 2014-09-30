function [soluzione] = consensusRob(A,b, W, n_iter)
%CONSENSUS_BASE definisce un semplice algoritmo di consenso per reti di
%sensori, in cui ogni sensore stima un proprio parametro sulla base di una
%porzione del training set. Per ora funziona 
%
%Input: x0: matrice (sxn) dei parametri inziali del grafo (s parametri per
%           n nodi)
%       W: matrice dei pesi associati agli archi del grafo (deve soddisfare
%           opportune proprietà)
%       n_iter: intero che definisce il numero di iterazioni dell'algoritmo
%
%Output: soluzione: vettore di n parametri finali corrispondenti ai parametri del
%           modello sottostante i dati
    
    n =size(x0,2);
    x=x0;
    
    spmd(n)
       
        neigh = W(labindex, :);
        neigh_idx = find(neigh > 0);
        neigh_idx(neigh_idx == labindex) = [];
        
        d = x(:,labindex);
        
        for ii = 1:n_iter
            
            new = neigh(labindex)*d;
            labSend(d, neigh_idx);
            
            for jj = 1:length(neigh_idx)
                
                while(~labProbe(neigh_idx(jj)))
                end
                    
                new = new + neigh(neigh_idx(jj))*labReceive(neigh_idx(jj));
      
            end
            d = new;
        end
        
    end
    
%    x=d;
    
    for dd=1:n
        beta(:,dd)=d{dd};
    end
    check;
    soluzione=beta(:,1);
end