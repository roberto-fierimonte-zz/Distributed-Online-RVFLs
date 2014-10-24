function [] = simulaz_regressione_online(X,Y,n_fold,n_run,K,lambda,n_iter,n_nodi,batch)

    for jj=1:n_run
        c = cvpartition(size(X,1),'kfold',n_fold);
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
                    distributor = cvpartition(size(Xtemp,1),'K',n_nodi);

                    [batchsol,K0]=rvflreg_sequenz(K0,Xtemp,Ytemp,batchsol,net);

                    [distrsol,K0dist]=distributed_regressiononlineseriale(K0dist,Xtemp,Ytemp,distrsol,net,W,n_iter,distributor);

                    [distrsol2,K0dist2]=distributed_regressiononlineseriale(K0dist2,Xtemp,Ytemp,distrsol2,net,W,0,distributor);

                    batchNMSE(kk,ii)=test_reg(X_test,Y_test,net,batchsol);
                    distrNMSE(kk,ii)=test_reg(X_test,Y_test,net,distrsol);
                    NMSEtest(kk,ii)=test_reg(X_test,Y_test,net,distrsol2);

                    start=(start+batch);
                else
                    n_online=n_online-1;
                end
            end  
        end
        NMSE(:,1,jj)=mean(batchNMSE,2);
        NMSE(:,2,jj)=mean(distrNMSE,2);
        NMSE(:,3,jj)=mean(NMSEtest,2);
        fprintf('run %i di %i completo\n',jj,n_run);
    end
    devst=std(NMSE,0,3);
    fprintf('Riepilogo simulazione:\n---------------------------------------------------------------------------------------------------------------\n');
    fprintf('                                    Media NMSE:   Dev.St.:\n\n');
    fprintf('Dati non distribuiti:               %.4f        %.4f\n\n',mean(NMSE(n_online,1),3),devst(n_online,1));
    fprintf('Dati distribuiti con consensus:     %.4f        %.4f\n\n',mean(NMSE(n_online,2),3),devst(n_online,2));
    fprintf('Dati distribuiti senza consensus:   %.4f        %.4f\n\n',mean(NMSE(n_online,3),3),devst(n_online,3));
    
    errorbar(1:size(batchNMSE,1),mean(NMSE(:,1),3),devst(:,1),'k--','LineWidth',2);
    hold on
    errorbar(1:size(batchNMSE,1),mean(NMSE(:,2),3),devst(:,2),'b','LineWidth',2);
    hold on
    errorbar(1:size(batchNMSE,1),mean(NMSE(:,3),3),devst(:,3),'r','LineWidth',2);
    
    box on;
    grid on;
    
    legend('centralizzato','distribuito','distribuito senza consensus','Location', 'NorthEast')
end

