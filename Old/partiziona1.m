c = cvpartition(size(X,1),'kfold',10);
[a,b,e,d]=rvflreg(X,Y,100,1);
errore=zeros(1,c.NumTestSets);
for i = 1:c.NumTestSets
    X_train=X(c.training(i),:);
    Y_train=Y(c.training(i),:);
    X_test=X(c.test(i),:);
    Y_test=Y(c.test(i),:);
    [sol,uscita,NMSE,delta,net]=distributed_regressionmod(X_train,Y_train,d,8,ones(8)/8,5);
    [exit,err,err2]=test_reg(X_test,Y_test,net,sol(:,1));
    errore(i)=err;
end
