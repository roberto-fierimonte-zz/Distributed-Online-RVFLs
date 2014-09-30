fprintf('Il tuo data set è composto da %i dati\n',size(X,1));
pX0=input('Scegli quanti dati vuoi usare per l addestramento (pX0): ');
pX1=input('Scegli quanti dati vuoi usare per il test (pX1): ');
if pX0+pX1>size(X,1)
    error('Errore: stai cercando di usare più dati (%i) di quelli disponibili (%i)',pX0+pX1,size(X,1));
end
fprintf('Utilizzo %i dati per l addestramento e %i dati per testare la bontà dell algoritmo\n\n',pX0,pX1);
X_train=X(1:pX0,:);Y_train=Y(1:pX0,:);
X_test=X(pX0+1:pX0+pX1,:);Y_test=Y(pX0+1:pX0+pX1,:);
tic;
[batchsol,batchtrainNMSE,batchtrainNSR]=rvflregmod(X_train,Y_train,net);
tic;
[distrsol,distrtrainNMSE,distrtrainNSR]=distributed_regression3mod(X_train,Y_train,net,n_nodi,W,n_iter);
tic;
[trainNMSE,trainNSR,testNMSE,testNSR]=distributed_regression3modsenzaconsensus(X_train,Y_train,X_test,Y_test,net,n_nodi);
[batcherr,batcherr2]=test_reg(X_test,Y_test,net,batchsol);
[distrerr,distrerr2]=test_reg(X_test,Y_test,net,distrsol);
NMSE=NMSE+[batcherr,distrerr,testNMSE];
NSR=NSR+[batcherr2,distrerr2,testNSR];
fprintf('\nRiepilogo simulazione:\n---------------------------------------------------------------------------------------------------------------\n');
fprintf('                                     NMSE:   NSR:\n\n');
fprintf('Dati non distribuiti:                %.4f     %.4f\n\n',batcherr,batcherr2);
fprintf('Dati distribuiti con consensus:      %.4f     %.4f\n\n',distrerr,distrerr2);
fprintf('Dati distribuiti senza consensus:    %.4f     %.4f\n\n',testNMSE,testNSR);   