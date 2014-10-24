function [errore] = test_class(X_test,Y_test,rete,beta)
%TEST_CLASS misura l'errore di classificazione del modello definito da una
%RVFL sul test set
%
%Input: X_test: matrice p x n dei campioni di test (p campioni di
%           dimensione n)
%       Y_test: vettore dei campioni di uscita (p campioni)
%       rete: struttura che contiene le informazioni relative alla RVFL
%           (dimensione dell'espansione, pesi e soglie della combinazione
%           affine e parametro di regolarizzazione)
%       beta: vettore dei parametri associati al modello
%
%Output: Errore: scalare che misura la frazione di campioni erroneamente
%           classificati sul totale dei campioni di test

pX=size(X_test,1);
esp=(exp(-(bsxfun(@plus,X_test*(rete.coeff)',rete.soglie')))+1).^-1;
uscita=(vec2ind((esp*beta)'))';
errore=1/(pX)*sum(sum(uscita~=Y_test));
end

