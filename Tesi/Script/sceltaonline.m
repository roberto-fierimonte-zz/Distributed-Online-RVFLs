fprintf('Il tuo data set è composto da %i dati\n',size(X,1));
pX0=input('Scegli quanti dati vuoi usare per un primo addestramento batch(pX0): ');
pX1=input('Scegli quanti dati vuoi usare per l addestramento online (pX1): ');
pXtest=input('Scegli quanti dati vuoi usare per il test (pXtest): ');
if pX0+pX1>size(X,1)
    error('Errore: stai cercando di usare più dati (%i) di quelli disponibili (%i)',pX0+pX1,size(X,1));
end
fprintf('Utilizzo %i dati per l addestramento e %i dati per testare la bontà dell algoritmo\n\n',pX0,pX1);
X_train=X(1:pX0,:);Y_train=Y(1:pX0,:);
X_test=X(pX0+1:pX0+pX1,:);Y_test=Y(pX0+1:pX0+pX1,:);
fprintf('Calcolo dei parametri in corso...\n');
tic;
[batchsol,batcherror]=rvflregmod(X_train,Y_train,net);
fprintf('Calcolo dei parametri in maniera non distribuita completato,trascorsi %.2f secondi\n', toc);
tic;
[distrsol,distrout,distrNMSE,distrdelta]=distributed_regression3mod(X_train,Y_train,net,n_nodi,W,n_iter);
fprintf('Calcolo dei parametri in maniera distribuita con consensus completato,trascorsi %.2f secondi\n', toc);
tic;
[testerr,testerr2]=distributed_regression3modsenzaconsensus(X_train,Y_train,X_test,Y_test,net,n_nodi);
fprintf('Calcolo dei parametri in maniera distribuita senza consensus completato,trascorsi %.2f secondi\n', toc);
[batchexit,batcherr,batcherr2]=test_reg(X_test,Y_test,net,batchsol);
[distrexit,distrerr,distrerr2]=test_reg(X_test,Y_test,net,distrsol);
NMSE=NMSE+[batcherr,distrerr,testerr];
NSR=NSR+[batcherr2,distrerr2,testerr2];
fprintf('\nRiepilogo simulazione:\n---------------------------------------------------------------------------------------------------------------\n');
fprintf('                                     NMSE:   NSR:\n\n');
fprintf('Dati non distribuiti:                %.4f     %.4f\n\n',batcherr,batcherr2);
fprintf('Dati distribuiti con consensus:      %.4f     %.4f\n\n',distrerr,distrerr2);
fprintf('Dati distribuiti senza consensus:    %.4f     %.4f\n\n',testerr,testerr2);   