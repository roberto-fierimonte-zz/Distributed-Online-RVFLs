function [] = simulaz_class_online(dataset,n_fold,n_run,K,lambda,n_iter,n_nodi,batch)

    X=dataset.X; Y=dataset.Y; m=size(Y,2);
    for jj=1:n_run
        if strcmp(dataset.type,'BC')
            c = cvpartition(Y,'kfold',n_fold);
        else
            c = cvpartition(size(X,1),'kfold',n_fold);
        end
        net=genera_rete(K,size(X,2),lambda);
        generagrafo;
        for ii = 1:c.NumTestSets

            X_train=X(c.training(ii),:);
            Y_train=Y(c.training(ii),:);
            X_test=X(c.test(ii),:);
            Y_test=Y(c.test(ii),:);

            start=1;
            n_online=ceil(size(X_train,1)/batch);
            batch_sol=zeros(K,m); distr_sol=zeros(K,m); local_sol=zeros(K,m);
            K_batch=net.lambda*eye(K);

            for cc=1:n_nodi
                K_dist(:,:,cc)=net.lambda*eye(K);
                K_local=K_dist;
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
                        if strcmp(dataset.type,'BC')
                            distributor = cvpartition(Ytemp,'kfold',n_nodi);
                        else
                            distributor = cvpartition(size(Xtemp,1),'kfold',n_nodi);
                        end
                    end

                    [batch_sol,K_batch]=rvfl_rls(K_batch,Xtemp,Ytemp,batch_sol,net);

                    [distr_sol,K_dist]=distributed_rvfl_rls_seriale(K_dist,Xtemp,Ytemp,distr_sol,net,W,n_iter,distributor);

                    [local_sol,K_local]=distributed_rvfl_rls_seriale(K_local,Xtemp,Ytemp,local_sol,net,W,0,distributor);

                    if strcmp(dataset.type,'BC')
                        batch_err(kk,ii)=test_classbin(X_test,Y_test,net,batch_sol);
                        distr_err(kk,ii)=test_classbin(X_test,Y_test,net,distr_sol);
                        local_err(kk,ii)=test_classbin(X_test,Y_test,net,local_sol);
                    else
                        batch_err(kk,ii)=test_class(X_test,vec2ind(Y_test')',net,batch_sol);
                        distr_err(kk,ii)=test_class(X_test,vec2ind(Y_test')',net,distr_sol);
                        local_err(kk,ii)=test_class(X_test,vec2ind(Y_test')',net,local_sol);
                    end

                    start=(start+batch);
                else
                    n_online=n_online-1;
                end
            end  
        end
        errore(:,1,jj)=mean(batch_err,2);
        errore(:,2,jj)=mean(distr_err,2);
        errore(:,3,jj)=mean(local_err,2);
        fprintf('run %i di %i completo\n',jj,n_run);
    end
    
    if strcmp(dataset.type,'BC')
        baseline=1/2*ones(1,3,n_run);
    else
        baseline=(m-1)/m*ones(1,3,n_run);
    end
    
    errore=[baseline; errore];
    devst=std(errore,1,3);
    fprintf('Riepilogo simulazione:\n--------------------------------------\n');
    fprintf('                                    Media errore:   Dev.St.:\n\n');
    fprintf('Centralized RVFL:                   %.4f          %.4f\n\n',mean(errore(n_online+1,1),3),devst(n_online,1));
    fprintf('RLS-Consensus RVFL:                 %.4f          %.4f\n\n',mean(errore(n_online+1,2),3),devst(n_online,2));
    fprintf('RLS-Local RVFL:                     %.4f          %.4f\n\n',mean(errore(n_online+1,3),3),devst(n_online,3));
    
    prepare_plot_online;
end
