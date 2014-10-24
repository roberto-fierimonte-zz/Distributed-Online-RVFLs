function [] = simulaz_classificazione_batch(X,Y,n_fold,n_run,K,lambda,n_iter,vett_nodi)

    errore=zeros(size(vett_nodi,2),4,n_run*n_fold);
    for ind=1:size(vett_nodi,2)
        
        n_nodi=vett_nodi(ind);
        
        for jj=1:n_run
            
            c = cvpartition(Y,'kfold',n_fold);
            [coeff,soglie]=genera_rete(K,size(X,2));
            net=struct('soglie',soglie,'coeff',coeff,'dimensione',K,'lambda',lambda);
            p=0.5;
            generagrafo;

            errore(ind,1)= n_nodi;
            
            for ii = 1:c.NumTestSets

                X_train=X(c.training(ii),:);
                Y_train=Y(c.training(ii),:);
                X_test=X(c.test(ii),:);
                Y_test=Y(c.test(ii),:);

                batchsol=rvflclass(X_train,Y_train,net);
                batcherr=test_class(X_test,Y_test,net,batchsol);
                
                if n_nodi == 1
                    distrsol=batchsol;
                    distrerr=test_class(X_test,Y_test,net,distrsol);
                    
                    distrsol2=batchsol;
                    errtest=test_class(X_test,Y_test,net,distrsol2);
                else
                    distributor = cvpartition(Y_train,'K',n_nodi);

                    distrsol=distributed_classificationseriale(X_train,Y_train,net,W,n_iter,distributor);
                    distrerr=test_class(X_test,Y_test,net,distrsol);

                    distrsol2=distributed_classificationseriale(X_train,Y_train,net,W,0,distributor);
                    errtest=test_class(X_test,Y_test,net,distrsol2);
                end

                errore(ind,:,(jj-1)*n_fold+ii)=[0,batcherr,distrerr,errtest];
            end
        end
        fprintf('simulazione con %i nodi completa\n',n_nodi);
    end
    devst=std(errore,0,3);
    fprintf('Riepilogo simulazione con 5 nodi:\n---------------------------------------------------------------------------------------------------------------\n');
    fprintf('                                    Media errore:   Dev.St.:\n\n');
    fprintf('Dati non distribuiti:               %.4f            %.4f\n\n',mean(errore(2,2,:),3),devst(2,2));
    fprintf('Dati distribuiti con consensus:     %.4f            %.4f\n\n',mean(errore(2,3,:),3),devst(2,3));
    fprintf('Dati distribuiti senza consensus:   %.4f            %.4f\n\n',mean(errore(2,4,:),3),devst(2,4));
    
    prepare_plot_batch;
end

