function [NMSE,NSR] = test_reg(X_test,Y_test,rete,beta)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
pX=size(X_test,1);
esp=(exp(-bsxfun(@plus,X_test*rete.coeff',rete.soglie'))+1).^-1;
%esp=tanh(bsxfun(@plus,X_test*rete.coeff',rete.soglie'));
uscita=esp*beta;
NMSE=1/(pX*var(Y_test))*sum((uscita-Y_test).^2);
NSR=10*log10(sum((Y_test-uscita).^2)/sum(Y_test.^2));
end

