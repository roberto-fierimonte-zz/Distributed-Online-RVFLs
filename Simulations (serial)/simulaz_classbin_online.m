function [] = simulaz_classbin_online(X,Y,n_fold,n_run,K,lambda,n_iter,n_nodi,batch)

    for jj=1:n_run
        c = cvpartition(Y,'kfold',n_fold);
        [coeff,soglie]=genera_rete(K,size(X,2));
        net=struct('soglie',soglie,'coeff',coeff,'dimensione',K,'lambda',lambda);
        generagrafo;
        for ii = 1:c.NumTestSets

            X_train=X(c.training(ii),:);
            Y_train=Y(c.training(ii),:);
            X_test=X(c.test(ii),:);
            Y_test=Y(c.test(ii),:);

            start=1;
            n_online=ceil(size(X_train,1)/batch);
            batchsol=zeros(K,1);
            distrsol=zeros(K,1);
            distrsol2=zeros(K,1);
            mom_sol=zeros(K,1);
            lms_sol2=zeros(K,1);
            aus=zeros(K,1);
            aus2=aus;
            K0=net.lambda*eye(K);

            for cc=1:n_nodi
                K0dist(:,:,cc)=net.lambda*eye(K);
                K0dist2=K0dist;
            end

            for kk=1:n_online
                if kk==n_online
                    Xtemp=X_train(start:end,:);
                    Ytemp=Y_train(start:end,:);
                else
                    Xtemp=X_train(start:(start+batch-1),:);
                    Ytemp=Y_train(start:(start+batch-1),:);
                end

                if size(Xtemp,1)>=n_nodi
                    if n_nodi == 1
                        distributor = 0;
                    else
                        distributor = cvpartition(Ytemp,'K',n_nodi);
                    end

                    [batchsol,K0]=rvflreg_sequenz(K0,Xtemp,Ytemp,batchsol,net);

                    [distrsol,K0dist]=distributed_regressiononlineseriale(K0dist,Xtemp,Ytemp,distrsol,net,W,n_iter,distributor);

                    [distrsol2,K0dist2]=distributed_regressiononlineseriale(K0dist2,Xtemp,Ytemp,distrsol2,net,W,0,distributor);
                    
                    [mom_sol,aus]=lms_momentum_seriale(Xtemp,Ytemp,mom_sol,aus,10^-3,1,net,W,kk,n_iter,distributor);  
                    
                    [lms_sol2,aus2]=lms_momentum_seriale(Xtemp,Ytemp,lms_sol2,aus2,10^-3,1,net,W,kk,0,distributor);

                    batcherr(kk,ii)=test_classbin(X_test,Y_test,net,batchsol);
                    distrerr(kk,ii)=test_classbin(X_test,Y_test,net,distrsol);
                    errtest(kk,ii)=test_classbin(X_test,Y_test,net,distrsol2);
                    lmserr(kk,ii)=test_classbin(X_test,Y_test,net,mom_sol);
                    lmserr2(kk,ii)=test_classbin(X_test,Y_test,net,lms_sol2);

                    start=(start+batch);
                else
                    n_online=n_online-1;
                end
            end  
        end
        errore(:,1,jj)=mean(batcherr,2);
        errore(:,2,jj)=mean(distrerr,2);
        errore(:,3,jj)=mean(errtest,2);
        errore(:,4,jj)=mean(lmserr,2);
        errore(:,5,jj)=mean(lmserr2,2);
        fprintf('run %i di %i completo\n',jj,n_run);
    end
    baseline=1/2*ones(1,5,n_run);
    errore=[baseline; errore];
    devst=std(errore,1,3);
    fprintf('Riepilogo simulazione:\n--------------------------------------\n');
    fprintf('                                    Media errore:   Dev.St.:\n\n');
    fprintf('Centralized RVFL:                   %.4f          %.4f\n\n',mean(errore(n_online+1,1),3),devst(n_online,1));
    fprintf('RLS-Consensus RVFL:                 %.4f          %.4f\n\n',mean(errore(n_online+1,2),3),devst(n_online,2));
    fprintf('RLS-Local RVFL:                     %.4f          %.4f\n\n',mean(errore(n_online+1,3),3),devst(n_online,3));
    
    prepare_plot_online;
end

