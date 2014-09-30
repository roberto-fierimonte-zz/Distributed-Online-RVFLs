function [errore] = test_classbin(X_test,Y_test,rete,beta)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
pX=size(X_test,1);
esp=(exp(-bsxfun(@plus,X_test*(rete.coeff)',rete.soglie'))+1).^-1;
%esp=tanh(bsxfun(@plus,X_test*(rete.coeff)',rete.soglie'));
uscita=sign(esp*beta);
errore=1/(pX)*sum(uscita~=Y_test);
end

