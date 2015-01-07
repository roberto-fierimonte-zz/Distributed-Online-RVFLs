function [] = simulaz_reg_online(dataset,n_fold,n_run,K,lambda,n_iter,n_nodi,batch)

    X=dataset.X; Y=dataset.Y; m=size(Y,2);
    X=[X;X]; Y=[Y;Y];
    train_time=zeros(4,n_run*n_fold);
    for jj=1:n_run
        c = cvpartition(size(X,1),'kfold',n_fold);
        generagrafo;
        net=generate_RVFL(K,size(X,2),lambda);
        for ii = 1:c.NumTestSets

            X_train=X(c.training(ii),:);
            Y_train=Y(c.training(ii),:);
            X_test=X(c.test(ii),:);
            Y_test=Y(c.test(ii),:);

            start=1;
            n_online=ceil(size(X_train,1)/batch);
            
            centr_sol=zeros(K,m); 
            distr_sol=zeros(K,m); 
            local_sol=zeros(K,m); 
            lms_sol=zeros(K,m);
            
            K_batch=net.lambda*eye(K);
            K_dist=repmat(K_centr,1,1,n_nodi); 
            K_local=K_dist;

            for kk=1:n_online
                if kk==n_online
                    Xtemp=X_train(start:end,:);
                    Ytemp=Y_train(start:end,:);
                else
                    Xtemp=X_train(start:(start+batch-1),:);
                    Ytemp=Y_train(start:(start+batch-1),:);
                end

                if size(Xtemp,1)>=n_nodi
                    distributor = cvpartition(size(Xtemp,1),'K',n_nodi);
                    
                    tic;
                    [centr_sol,K_batch]=rvfl_rls(K_batch,Xtemp,Ytemp,centr_sol,net);
                    train_time(1,(jj-1)*n_fold+ii)=train_time(1,(jj-1)*n_fold+ii)...
                        +toc;
                    
                    tic;
                    [distr_sol,K_dist]=distributed_rvfl_rls_seriale(K_dist,Xtemp,Ytemp,distr_sol,net,W,n_iter,distributor);
                    train_time(2,(jj-1)*n_fold+ii)=train_time(2,(jj-1)*n_fold+ii)...
                        +toc;
                    
                    tic;
                    [local_sol,K_local]=distributed_rvfl_rls_seriale(K_local,Xtemp,Ytemp,local_sol,net,W,0,distributor);
                    train_time(3,(jj-1)*n_fold+ii)=train_time(3,(jj-1)*n_fold+ii)+toc;
                    
                    tic;
                    lms_sol=distributed_rvfl_sgd_seriale(Xtemp,Ytemp,lms_sol,net,W,n_iter,distributor,kk);
                    train_time(4,(jj-1)*n_fold+ii)=train_time(4,(jj-1)*n_fold+ii)...
                        +toc;
                    
                    centr_NRMSE(kk,ii)=test_reg(X_test,Y_test,net,centr_sol);
                    distr_NRMSE(kk,ii)=test_reg(X_test,Y_test,net,distr_sol);
                    local_NRMSE(kk,ii)=test_reg(X_test,Y_test,net,local_sol);
                    lms_NRMSE(kk,ii)=test_reg(X_test,Y_test,net,lms_sol);

                    start=(start+batch);
                else
                    n_online=n_online-1;
                end
            end  
        end
        NRMSE(:,1,jj)=mean(centr_NRMSE,2);
        NRMSE(:,2,jj)=mean(distr_NRMSE,2);
        NRMSE(:,3,jj)=mean(local_NRMSE,2);
        NRMSE(:,4,jj)=mean(lms_NRMSE,2);
        fprintf('run %i di %i completo\n',jj,n_run);
    end
    
    baseline=ones(1,4,n_run);
    errore=[baseline; NRMSE];
    devst=std(errore,1,3);
    
    fprintf('Simulation results:\n-------------------------------------------\n');
    fprintf('                              Mean Error:   St.Dev.:  Total Time:\n\n');
    fprintf('Centralized RVFL:             %.4f          %.4f          %.4f\n\n'...
        ,mean(errore(n_online+1,1),3),devst(n_online,1),mean(train_time(1,:)));
    fprintf('RLS-Consensus RVFL:           %.4f          %.4f          %.4f\n\n'...
        ,mean(errore(n_online+1,2),3),devst(n_online,2),mean(train_time(2,:)));
    fprintf('LMS-Consensus RVFL:           %.4f          %.4f          %.4f\n\n'...
        ,mean(errore(n_online+1,4),3),devst(n_online,4),mean(train_time(4,:)));
    fprintf('RLS-Local RVFL:               %.4f          %.4f          %.4f\n\n'...
        ,mean(errore(n_online+1,3),3),devst(n_online,3),mean(train_time(3,:)));
    
    prepare_plot_online;
end

