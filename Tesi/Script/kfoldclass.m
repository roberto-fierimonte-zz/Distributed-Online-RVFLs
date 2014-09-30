c = cvpartition(size(X,1),'kfold',k);
t1temp=0;t2temp=0;t3temp=0;erroretemp=[0,0,0];
for ii = 1:c.NumTestSets
    X_train=X(c.training(ii),:);
    Y_train=Y(c.training(ii),:);
    X_test=X(c.test(ii),:);
    Y_test=Y(c.test(ii),:);
    tic;
    batchsol=rvflclass(X_train,Y_train,net);
    t1temp=t1temp+toc;
    tic;
    distrsol=distributed_classification(X_train,Y_train,net,W,n_iter);
    t2temp=t2temp+toc;
    batcherr=test_class(X_test,Y_test,net,batchsol);
    distrerr=test_class(X_test,Y_test,net,distrsol);
    tic;
    errtest=distributed_classificationsenzaconsensus(X_train,Y_train,X_test,Y_test,net,n_nodi);
    t3temp=t3temp+toc;
    erroretemp=erroretemp+[batcherr,distrerr,errtest];
    errore=errore+[batcherr,distrerr,errtest];
    t=t+[t1temp,t2temp,t3temp];
    fprintf('%i percento\n',((jj-1)*k+ii)*100/(k*n));  
end
fprintf('Riepilogo run %i:\n---------------------------------------------------------------------------------------------------------------\n',jj);
fprintf('                                    Media errore:   Media Training Time:\n\n');
fprintf('Dati non distribuiti:               %.4f        %.4f\n\n',erroretemp(1)/k,t1temp/k);
fprintf('Dati distribuiti con consensus:     %.4f        %.4f\n\n',erroretemp(2)/k,t2temp/k);
fprintf('Dati distribuiti senza consensus:   %.4f        %.4f\n\n',erroretemp(3)/k,t3temp/k);