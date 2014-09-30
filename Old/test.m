[soluzione,errore,exit,rete]=rvflreg(X,Y,5000,0.00001);
[sol_dist,err_dist,exit_dist,rete_dist]=rvflreg_distr(X,Y,5000,0.00001,6);
A=ones(6); A=A/6;
[consenso,delta]=consensus(sol_dist,A,100);
[sol_test,err_test,exit_test]=rvflreg_test(X,Y,rete);
[sol_testdist,err_testdist,exit_testdist]=rvflreg_test(X,Y,rete_dist);