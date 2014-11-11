function [] = simulaz_class_bigdata(dataset,train_size,n_run,K,lambda,n_iter,n_nodi,batch)

    X=dataset.X; Y=dataset.Y; m=size(Y,2);
    for jj=1:n_run
        net=generate_RVFL(K,size(X,2),lambda);
        generagrafo;

        X_train=X(1:train_size,:);
        Y_train=Y(1:train_size,:);
        X_test=X(train_size+1:end,:);
        Y_test=Y(train_size+1:end,:);

        start=1;
        n_online=ceil(size(X_train,1)/batch);
        centr_sol=zeros(K,m); distr_sol=zeros(K,m); local_sol=zeros(K,m);
        K_centr=net.lambda*eye(K);
            
        sgdcentr_sol=zeros(K,m); sgdcentr_aus=zeros(K,m);

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

                [centr_sol,K_centr]=rvfl_rls(K_centr,Xtemp,Ytemp,centr_sol,net);

                [distr_sol,K_dist]=distributed_rvfl_rls_seriale(K_dist,Xtemp,Ytemp,distr_sol,net,W,n_iter,distributor);

                [local_sol,K_local]=distributed_rvfl_rls_seriale(K_local,Xtemp,Ytemp,local_sol,net,W,0,distributor);

                [sgdcentr_sol,sgdcentr_aus]=rvfl_sgd(Xtemp,Ytemp,1,1,sgdcentr_sol,sgdcentr_aus,kk,net);

                if strcmp(dataset.type,'BC')
                    centr_err(kk)=test_classbin(X_test,Y_test,net,centr_sol);
                    distr_err(kk)=test_classbin(X_test,Y_test,net,distr_sol);
                    local_err(kk)=test_classbin(X_test,Y_test,net,local_sol);
                    sgdcentr_err(kk)=test_classbin(X_test,Y_test,net,sgdcentr_sol);
                else
                    centr_err(kk)=test_class(X_test,vec2ind(Y_test')',net,centr_sol);
                    distr_err(kk)=test_class(X_test,vec2ind(Y_test')',net,distr_sol);
                    local_err(kk)=test_class(X_test,vec2ind(Y_test')',net,local_sol);
                    sgdcentr_err(kk)=test_class(X_test,vec2ind(Y_test')',net,sgdcentr_sol);
                end

                start=(start+batch);
            else
                n_online=n_online-1;
            end 
        end
        errore(:,1,jj)=centr_err;
        errore(:,2,jj)=distr_err;
        errore(:,3,jj)=local_err;
        errore(:,4,jj)=sgdcentr_err;
        fprintf('run %i di %i completo\n',jj,n_run);
    end
    
    if strcmp(dataset.type,'BC')
        baseline=1/2*ones(1,4,n_run);
    else
        baseline=(m-1)/m*ones(1,4,n_run);
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

