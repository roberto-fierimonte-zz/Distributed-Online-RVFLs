c = cvpartition(size(X,1),'kfold',10);
fprintf('Avvio simulazione di apprendimento distribuito...\n');
[a,b,e,d]=rvflreg(X,Y,5000,0.0001);
errore=zeros(1,c.NumTestSets);
snr=zeros(1,c.NumTestSets);
for i = 1:c.NumTestSets
    fprintf('Fold: %i\n', i);
    X_train=X(c.training(i),:);
    Y_train=Y(c.training(i),:);
    X_test=X(c.test(i),:);
    Y_test=Y(c.test(i),:);
    fprintf('Calcolo dei parametri in corso...\n');
    tic;
    [sol,uscita,NMSE,delta,net]=distributed_regression3mod(X_train,Y_train,d,8,ones(8)/8,5);
    fprintf('Calcolo dei parametri completato,trascorsi %.2f secondi\n', toc);
    [exit,err,err2]=test_reg(X_test,Y_test,net,sol(:,1));
    errore(i)=err;
    snr(i)=err2;
    fprintf('Prestazioni di test : \nNMSE: %.4f \nNSR: %.4f \n \n', err,err2);
end
erroremedio=mean(errore);
snrmedio=mean(snr);
fprintf('Media errore NMSE di test: %.4f \n',erroremedio);
fprintf('Media NSR di test: %.4f \n',snrmedio);