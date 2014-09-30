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
K=200;
[coeff,soglie]=genera_rete(200,8);
net=struct('coeff',coeff,'soglie',soglie,'dimensione',K,'lambda',1);
solbatch=rvflreg(X0,Y0,net);
[solonline0,K1]=rvflreg_sequenz(net.lambda*eye(K),X0,Y0,zeros(K,1),net);
[solonline1,K2]=rvflreg_sequenz(K1,X1,Y1,solonline0,net);
[solonline2,K3]=rvflreg_sequenz(K2,X2,Y2,solonline1,net);
[solonline3,K4]=rvflreg_sequenz(K3,X3,Y3,solonline2,net);
solbatch400=rvflreg(X400,Y400,net);
[onlinerr01,onlinerr02]=test_reg(X_test,Y_test,net,solonline0);
[onlinerr11,onlinerr12]=test_reg(X_test,Y_test,net,solonline1);
[onlinerr21,onlinerr22]=test_reg(X_test,Y_test,net,solonline2);
[onlinerr31,onlinerr32]=test_reg(X_test,Y_test,net,solonline3);
[batcherr401,batcherr402]=test_reg(X_test,Y_test,net,solbatch400);