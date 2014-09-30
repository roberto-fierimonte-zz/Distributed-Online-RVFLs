c = cvpartition(size(X,1),'kfold',10);
fprintf('Avvio simulazione di apprendimento distribuito...\n');
[a,b,e,d]=rvflclass(X,Y,1500,0.0001);
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
    [sol,uscita,error,delta,net]=distributed_classification3mod(X_train,Y_train,d,8,ones(8)/8,5);
    fprintf('Calcolo dei parametri completato,trascorsi %.2f secondi\n', toc);
    [exit,err]=test_class(X_test,Y_test,net,sol);
    errore(i)=err;
    fprintf('Prestazioni di test : \nFrazione di errori di classificazione: %.4f \n \n', err);
end
erroremedio=mean(errore);
snrmedio=mean(snr);
fprintf('Frazione media di errori di classificazione di test: %.4f \n',erroremedio);