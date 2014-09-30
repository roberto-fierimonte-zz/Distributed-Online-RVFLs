k=input('Inserisci la dimensione della k-fold cross-validation (k): ');
fprintf('\nEffettuo una %i-fold cross-validation per testare la bont� dell algoritmo\n\n',k);
c = cvpartition(size(X,1),'kfold',k);
t1=0;t2=0;t3=0;
for ii = 1:c.NumTestSets
    X_train=X(c.training(ii),:);
    Y_train=Y(c.training(ii),:);
    X_test=X(c.test(ii),:);
    Y_test=Y(c.test(ii),:);
    tic;
    [batchsol,batchtrainNMSE,batchtrainNSR]=rvflreg(X_train,Y_train,net);
    t1=t1+toc;
    tic;
    [distrsol,distrtrainNMSE,distrtrainNSR]=distributed_regression(X_train,Y_train,net,W,n_iter);
    t2=t2+toc;
    tic;
    [batcherr]=test_classbin(X_test,Y_test,net,batchsol);
    [distrerr]=test_classbin(X_test,Y_test,net,distrsol);
        NMSE=NMSE+[batcherr,distrerr];
    fprintf('%i0 percento\n',ii*10/k);
    %fprintf('Prestazioni di test :                NMSE:      NSR:\n\n');
    %fprintf('Dati non distribuiti:                %.4f     %.4f\n\n',batcherr,batcherr2);
    %fprintf('Dati distribuiti con consensus:      %.4f     %.4f\n\n',distrerr,distrerr2);
    %fprintf('Dati distribuiti senza consensus:    %.4f     %.4f\n\n',testNMSE,testNSR);   
end
fprintf('Riepilogo simulazione:\n---------------------------------------------------------------------------------------------------------------\n');
fprintf('                                    Media NMSE:   Media NSR:   Media Training Time:\n\n');
fprintf('Dati non distribuiti:               %.4f        %.4f     %.4f\n\n',NMSE(1)/k,NSR(1)/k,t1/k);
fprintf('Dati distribuiti con consensus:     %.4f        %.4f     %.4f\n\n',NMSE(2)/k,NSR(2)/k,t2/k);
fprintf('Dati distribuiti senza consensus:   %.4f        %.4f     %.4f\n\n',NMSE(3)/k,NSR(3)/k,t3/k);