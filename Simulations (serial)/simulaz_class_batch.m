function [] = simulaz_class_batch(dataset,n_fold,n_run,K,lambda,n_iter,vett_nodi)
    
    X=dataset.X; Y=dataset.Y;
    errore=zeros(size(vett_nodi,2),4,n_run*n_fold);
    train_time=zeros(size(vett_nodi,2),4,n_run*n_fold);
    cons_iter=zeros(size(vett_nodi,2),n_run*n_fold+1);
    for ind=1:size(vett_nodi,2)
        
        n_nodi=vett_nodi(ind);
        
        for jj=1:n_run
            
            if strcmp(dataset.type,'BC')
                c = cvpartition(Y,'kfold',n_fold);
            else
                c = cvpartition(size(X,1),'kfold',n_fold);
            end
            net=genera_rete(K,size(X,2),lambda);
            generagrafo;

            errore(ind,1)= n_nodi;
            train_time(ind,1)= n_nodi;
            cons_iter(ind,1)=n_nodi;
            
            for ii = 1:c.NumTestSets

                X_train=X(c.training(ii),:);
                Y_train=Y(c.training(ii),:);
                X_test=X(c.test(ii),:);
                Y_test=Y(c.test(ii),:);
                
                if n_nodi == 1
                    distributor = 0;
                else
                    if strcmp(dataset.type,'BC')
                        distributor = cvpartition(Y_train,'kfold',n_nodi);
                    else
                        distributor = cvpartition(size(X_train,1),'kfold',n_nodi);
                    end
                end
                    tic;
                    batch_sol=rvfl(X_train,Y_train,net);
                    batch_time=toc;
                
                    tic;
                    [distr_sol,iterations]=distributed_rvfl_seriale(X_train,Y_train,net,W,n_iter,distributor);
                    distr_time=toc;

                    tic;
                    local_sol=distributed_rvfl_seriale(X_train,Y_train,net,W,0,distributor);
                    local_time=toc;
                    if strcmp(dataset.type,'BC')
                        batch_err=test_classbin(X_test,Y_test,net,batch_sol);
                        distr_err=test_classbin(X_test,Y_test,net,distr_sol);
                        local_err=test_classbin(X_test,Y_test,net,local_sol);
                    else
                        batch_err=test_class(X_test,vec2ind(Y_test')',net,batch_sol);
                        distr_err=test_class(X_test,vec2ind(Y_test')',net,distr_sol);
                        local_err=test_class(X_test,vec2ind(Y_test')',net,local_sol);
                    end

                errore(ind,:,(jj-1)*n_fold+ii)=[0,batch_err,distr_err,local_err];
                train_time(ind,:,(jj-1)*n_fold+ii)=[0,batch_time,distr_time/n_nodi,local_time/n_nodi];
                cons_iter(ind,1+(jj-1)*n_fold+ii)=iterations;
            end
        end
        fprintf('simulazione con %i nodi completa\n',n_nodi);
    end
    devst_err=std(errore,1,3);
    fprintf('Riepilogo simulazione con 5 nodi:\n---------------------------------------------------------------------------------------------------------------\n');
    fprintf('                   Media errore:   Dev.St.:   Training time (s):\n\n');
    fprintf('Centralized:       %.4f          %.4f     %.4f\n\n',mean(errore(2,2,:),3),devst_err(2,2),mean(train_time(2,2,:),3));
    fprintf('Consensus:         %.4f          %.4f     %.4f\n\n',mean(errore(2,3,:),3),devst_err(2,3),mean(train_time(2,3,:),3));
    fprintf('Local:             %.4f          %.4f     %.4f\n\n',mean(errore(2,4,:),3),devst_err(2,4),mean(train_time(2,4,:),3));
    fprintf('Numero medio di iterazioni del consenso: %d\n',round(mean(cons_iter(2,2:end),2)));
    
    prepare_plot_batch_err;
end

