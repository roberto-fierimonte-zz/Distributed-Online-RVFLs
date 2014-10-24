function [] = simulaz_classbin_batch(X,Y,n_fold,n_run,K,lambda,n_iter,vett_nodi)

    errore=zeros(size(vett_nodi,2),4,n_run*n_fold);
    for ind=1:size(vett_nodi,2)
        
        n_nodi=vett_nodi(ind);
        
        for jj=1:n_run
            
            c = cvpartition(Y,'kfold',n_fold);
            [coeff,soglie]=genera_rete(K,size(X,2));
            net=struct('soglie',soglie,'coeff',coeff,'dimensione',K,'lambda',lambda);
            generagrafo;

            errore(ind,1)= n_nodi;
            
            for ii = 1:c.NumTestSets

                X_train=X(c.training(ii),:);
                Y_train=Y(c.training(ii),:);
                X_test=X(c.test(ii),:);
                Y_test=Y(c.test(ii),:);

                batchsol=rvflreg(X_train,Y_train,net);
                batcherr=test_classbin(X_test,Y_test,net,batchsol);
                
                if n_nodi == 1
                    distributor = 0;
                else
                    distributor = cvpartition(Y_train,'K',n_nodi);
                end
                    distrsol=distributed_regressionseriale(X_train,Y_train,net,W,n_iter,distributor);
                    distrerr=test_classbin(X_test,Y_test,net,distrsol);

                    distrsol2=distributed_regressionseriale(X_train,Y_train,net,W,0,distributor);
                    errtest=test_classbin(X_test,Y_test,net,distrsol2);
                    
                    %ATCsol=rvfl_ATC_seriale(X_train,Y_train,net,W,zeros(K,n_nodi),10^-5,100000,distributor);
                    %ATCerr=test_classbin(X_test,Y_test,net,ATCsol);

                errore(ind,:,(jj-1)*n_fold+ii)=[0,batcherr,distrerr,errtest];
            end
        end
        fprintf('simulazione con %i nodi completa\n',n_nodi);
    end
    devst=std(errore,1,3);
    fprintf('Riepilogo simulazione con 5 nodi:\n---------------------------------------------------------------------------------------------------------------\n');
    fprintf('                                    Media errore:   Dev.St.:\n\n');
    fprintf('Dati non distribuiti:               %.4f            %.4f\n\n',mean(errore(2,2,:),3),devst(2,2));
    fprintf('Dati distribuiti con consensus:     %.4f            %.4f\n\n',mean(errore(2,3,:),3),devst(2,3));
    fprintf('Dati distribuiti senza consensus:   %.4f            %.4f\n\n',mean(errore(2,4,:),3),devst(2,4));
    %fprintf('Dati non distribuiti, ATC:          %.4f            \n\n',mean(errore(1,5,:),3));
    
    prepare_plot_batch;
end

