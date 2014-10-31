function [] = simulaz_classbin_batch(X,Y,n_fold,n_run,K,lambda,n_iter,vett_nodi)

    errore=zeros(size(vett_nodi,2),4,n_run*n_fold);
    train_time=zeros(size(vett_nodi,2),4,n_run*n_fold);
    cons_iter=zeros(size(vett_nodi,2),n_run*n_fold+1);
    for ind=1:size(vett_nodi,2)
        
        n_nodi=vett_nodi(ind);
        
        for jj=1:n_run
            
            c = cvpartition(Y,'kfold',n_fold);
            [coeff,soglie]=genera_rete(K,size(X,2));
            net=struct('soglie',soglie,'coeff',coeff,'dimensione',K,'lambda',lambda);
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
                    distributor = cvpartition(Y_train,'K',n_nodi);
                end
                    tic;
                    batch_sol=distributed_regressionseriale(X_train,Y_train,net,1,n_iter,distributor);
                    batch_time=toc;
                    batch_err=test_classbin(X_test,Y_test,net,batch_sol);
                
                    tic;
                    [distr_sol,iterations]=distributed_regressionseriale(X_train,Y_train,net,W,n_iter,distributor);
                    distr_time=toc;
                    distr_err=test_classbin(X_test,Y_test,net,distr_sol);

                    tic;
                    local_sol=distributed_regressionseriale(X_train,Y_train,net,W,0,distributor);
                    local_time=toc;
                    local_err=test_classbin(X_test,Y_test,net,local_sol);

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

