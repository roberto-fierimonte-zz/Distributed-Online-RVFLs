function [] = simulaz_reg_online(dataset,n_fold,n_run,K,lambda,n_iter,n_nodi,batch)

    X=dataset.X; Y=dataset.Y; m=size(Y,2);
    for jj=1:n_run
        c = cvpartition(size(X,1),'kfold',n_fold);
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
                    distributor = cvpartition(size(Xtemp,1),'K',n_nodi);

                    [batch_sol,K_batch]=rvfl_rls(K_batch,Xtemp,Ytemp,batch_sol,net);

                    [distr_sol,K_dist]=distributed_rvfl_rls_seriale(K_dist,Xtemp,Ytemp,distr_sol,net,W,n_iter,distributor);

                    [local_sol,K_local]=distributed_rvfl_rls_seriale(K_local,Xtemp,Ytemp,local_sol,net,W,0,distributor);

                    batch_NRMSE(kk,ii)=test_reg(X_test,Y_test,net,batch_sol);
                    distr_NRMSE(kk,ii)=test_reg(X_test,Y_test,net,distr_sol);
                    local_NRMSE(kk,ii)=test_reg(X_test,Y_test,net,local_sol);

                    start=(start+batch);
                else
                    n_online=n_online-1;
                end
            end  
        end
        NMSE(:,1,jj)=mean(batch_NRMSE,2);
        NMSE(:,2,jj)=mean(distr_NRMSE,2);
        NMSE(:,3,jj)=mean(local_NRMSE,2);
        fprintf('run %i di %i completo\n',jj,n_run);
    end
    devst=std(NMSE,0,3);
    fprintf('Riepilogo simulazione:\n---------------------------------------------------------------------------------------------------------------\n');
    fprintf('                                    Media NMSE:   Dev.St.:\n\n');
    fprintf('Dati non distribuiti:               %.4f        %.4f\n\n',mean(NMSE(n_online,1),3),devst(n_online,1));
    fprintf('Dati distribuiti con consensus:     %.4f        %.4f\n\n',mean(NMSE(n_online,2),3),devst(n_online,2));
    fprintf('Dati distribuiti senza consensus:   %.4f        %.4f\n\n',mean(NMSE(n_online,3),3),devst(n_online,3));
    
    errorbar(1:size(batch_NRMSE,1),mean(NMSE(:,1),3),devst(:,1),'k--','LineWidth',2);
    hold on
    errorbar(1:size(batch_NRMSE,1),mean(NMSE(:,2),3),devst(:,2),'b','LineWidth',2);
    hold on
    errorbar(1:size(batch_NRMSE,1),mean(NMSE(:,3),3),devst(:,3),'r','LineWidth',2);
    
    box on;
    grid on;
    
    legend('centralizzato','distribuito','distribuito senza consensus','Location', 'NorthEast')
end

