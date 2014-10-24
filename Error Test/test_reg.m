function [NMSE,NSR] = test_reg(X_test,Y_test,rete,beta)
%TEST_REG misura l'errore di regressione del modello definito da una RVFL
%sul test set
%
%Input: X_test: matrice p x n dei campioni di test (p campioni di
%           dimensione n)
%       Y_test: vettore dei campioni di uscita (p campioni)
%       rete: struttura che contiene le informazioni relative alla RVFL
%           (dimensione dell'espansione, pesi e soglie della combinazione
%           affine e parametro di regolarizzazione)
%       beta: vettore dei parametri associati al modello
%
%Output: NMSE: scalare che misura il rapporto tra l'errore quadratico e la
%           varianza del test set
%        NSR: scalare che misura il rapporto rumore-segnale sul test set

esp=(exp(-bsxfun(@plus,X_test*rete.coeff',rete.soglie'))+1).^-1;
uscita=esp*beta;
NMSE=sum(((uscita-Y_test).^2))/(size(X_test,1)*var(Y_test));
NSR=10*log10(sum((Y_test-uscita).^2)/sum(Y_test.^2));
end

