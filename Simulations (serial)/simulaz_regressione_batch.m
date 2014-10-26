function [] = simulaz_regressione_batch(X,Y,n_fold,n_run,K,lambda,n_iter,vett_nodi)

    NMSE=zeros(size(vett_nodi,2),4,n_run*n_fold);
    NSR=zeros(size(vett_nodi,2),4,n_run*n_fold);
    train_time=zeros(size(vett_nodi,2),4,n_run*n_fold);
    for ind=1:size(vett_nodi,2)
        
        n_nodi=vett_nodi(ind);
        
        for jj=1:n_run
            
            c = cvpartition(size(X,1),'kfold',n_fold);
            [coeff,soglie]=genera_rete(K,size(X,2));
            net=struct('soglie',soglie,'coeff',coeff,'dimensione',K,'lambda',lambda);
            generagrafo;

            NMSE(ind,1)= n_nodi;
            NSR(ind,1)= n_nodi;
            train_time(ind,1)= n_nodi;
            
            for ii = 1:c.NumTestSets

                X_train=X(c.training(ii),:);
                Y_train=Y(c.training(ii),:);
                X_test=X(c.test(ii),:);
                Y_test=Y(c.test(ii),:);

                tic;
                batchsol=rvflreg(X_train,Y_train,net);
                time_batch=toc;
                [batchNMSE,batchNSR]=test_reg(X_test,Y_test,net,batchsol);
                
                if n_nodi == 1
                    distributor = 0;
                else
                    distributor = cvpartition(Y_train,'K',n_nodi);
                end
                    tic;
                    distrsol=distributed_regressionseriale(X_train,Y_train,net,W,n_iter,distributor);
                    time_distr=toc;
                    [distrNMSE,distrNSR]=test_reg(X_test,Y_test,net,distrsol);

                    tic;
                    distrsol2=distributed_regressionseriale(X_train,Y_train,net,W,0,distributor);
                    time_test=toc;
                    [NMSEtest,NSRtest]=test_reg(X_test,Y_test,net,distrsol2);

                NMSE(ind,:,(jj-1)*n_fold+ii)=[0,batchNMSE,distrNMSE,NMSEtest];
                NSR(ind,:,(jj-1)*n_fold+ii)=[0,batchNSR,distrNSR,NSRtest];
                train_time(ind,:,(jj-1)*n_fold+ii)=[0,time_batch,time_distr/n_nodi,time_test/n_nodi];
            end
        end
        fprintf('simulazione con %i nodi completa\n',n_nodi);
    end
    
    devstNMSE=std(NMSE,0,3);
    devstNSR= std(NSR,0,3);
    
    fprintf('Riepilogo simulazione con 5 nodi:\n---------------------------------------------------------------------------------------------------------------\n');
    fprintf('                                    Media NMSE:     Dev.St.:     Media NSR:     Dev.St.:\n\n');
    fprintf('Dati non distribuiti:               %.4f            %.4f         %.4f\n\n',mean(NMSE(2,2,:),3),std(NMSE(2,2,:)),mean(NSR(2,2,:),3),std(NSR(2,2,:)));
    fprintf('Dati distribuiti con consensus:     %.4f            %.4f         %.4f\n\n',mean(NMSE(2,3,:),3),std(NMSE(2,3,:)),mean(NSR(2,3,:),3),std(NSR(2,3,:)));
    fprintf('Dati distribuiti senza consensus:   %.4f            %.4f         %.4f\n\n',mean(NMSE(2,4,:),3),std(NMSE(2,4,:)),mean(NSR(2,4,:),3),std(NSR(2,4,:)));
    
    errorbar(NMSE(:,1,1),(mean(NMSE(:,2,:),3)),devstNMSE(:,2),'k--','LineWidth',2);
    hold on
    errorbar(NMSE(:,1,1),(mean(NMSE(:,3,:),3)),devstNMSE(:,3),'b','LineWidth',2);
    hold on
    errorbar(NMSE(:,1,1),(mean(NMSE(:,4,:),3)),devstNMSE(:,4),'r','LineWidth',2);
    
    xlim([0 55]);
    
    box on;
    grid on;
    
    legend('centralizzato','distribuito','distribuito senza consensus','Location', 'NorthWest')
end

