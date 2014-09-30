clc
clear
load('/Users/robertofierimonte/Documents/MATLAB/Regressione/abalone.mat')
[X,Y]=preprocess(abalone_X,abalone_Y);
X0=X(1:100,:);Y0=Y(1:100,:);
X1=X(101:200,:);Y1=Y(101:200,:);
X2=X(201:300,:); Y2=Y(201:300,:);
X3=X(301:400,:); Y3=Y(301:400,:);
X400=X(1:400,:);Y400=Y(1:400,:);
X_test=X(1000:2000,:);Y_test=Y(1000:2000,:);
K=500;
n_nodi=5;
p=0.5;
generagrafo;
[coeff,soglie]=genera_rete(K,8);
net=struct('coeff',coeff,'soglie',soglie,'dimensione',K,'lambda',100);
spmd (n_nodi)
    for ii=1:n_nodi
       K0=net.lambda*eye(K);
    end
end
solbatch100=distributed_regression(X0,Y0,net,W,500);
%solRob100=distributed_regressionR(X0,Y0,net,W,500);
solcentr100=rvflreg(X0,Y0,net);
[solonline0,K1]=distributed_regressiononline(K0,X0,Y0,zeros(K,1),net,W,500);
[solonline1,K2]=distributed_regressiononline(K1,X1,Y1,solonline0,net,W,500);
[solonline2,K3]=distributed_regressiononline(K2,X2,Y2,solonline1,net,W,500);
[solonline3,K4]=distributed_regressiononline(K3,X3,Y3,solonline2,net,W,500);
solbatch400=distributed_regression(X400,Y400,net,W,500);
%solRob400=distributed_regressionR(X400,Y400,net,W,500);
solcentr400=rvflreg(X400,Y400,net);
[onlinerr01,onlinerr02]=test_reg(X_test,Y_test,net,solonline0);
[onlinerr11,onlinerr12]=test_reg(X_test,Y_test,net,solonline1);
[onlinerr21,onlinerr22]=test_reg(X_test,Y_test,net,solonline2);
[onlinerr31,onlinerr32]=test_reg(X_test,Y_test,net,solonline3);
[batcherr401,batcherr402]=test_reg(X_test,Y_test,net,solbatch400);
%[Roberr401,Roberr402]=test_reg(X_test,Y_test,net,solRob400);
[centrerr401,centrerr402]=test_reg(X_test,Y_test,net,solcentr400);