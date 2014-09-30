n_nodi=5;
load('/Users/robertofierimonte/Documents/MATLAB/Regressione/g50c.mat')
[coeff,soglie]=genera_rete(200,50);
net=struct('coeff',coeff,'soglie',soglie,'dimensione',200,'lambda',100);
generagrafo;