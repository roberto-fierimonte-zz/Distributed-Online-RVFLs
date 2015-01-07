function [] = simulaz_batch(dataset,n_fold,n_run,K,lambda,n_iter,vett_nodi)
    
%Import dataset
    X=dataset.X; Y=dataset.Y;

%Initialize error, training time and Consensus iterations
    if strcmp(dataset.type,'R')
        NRMSE=zeros(size(vett_nodi,2),4,n_run*n_fold);
        NSR=zeros(size(vett_nodi,2),4,n_run*n_fold);
    else
        error=zeros(size(vett_nodi,2),4,n_run*n_fold); 
    end   
    
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
            
%Generate network topology and RVFL structure            
            generagrafo;
            net=generate_RVFL(K,size(X,2),lambda);
            
            if strcmp(dataset.type,'R')
                NRMSE(ind,1)= n_nodi;
                NSR(ind,1)= n_nodi;
            else
                error(ind,1)= n_nodi; 
            end 
            
            train_time(ind,1)= n_nodi;
            cons_iter(ind,1)=n_nodi;
            
%Initialize training and test sets            
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

%Compute solutions                
                    tic;
                    batch_sol=distributed_rvfl_seriale...
                        (X_train,Y_train,net,1,n_iter,distributor);
                    batch_time=toc;
                
                    tic;
                    [distr_sol,iterations]=distributed_rvfl_seriale...
                        (X_train,Y_train,net,W,n_iter,distributor);
                    distr_time=toc;

                    tic;
                    local_sol=distributed_rvfl_seriale...
                        (X_train,Y_train,net,W,0,distributor);
                    local_time=toc;
                    
%Compute test error                    
                    if strcmp(dataset.type,'BC')
                        batch_err=test_classbin(X_test,Y_test,net,batch_sol);
                        distr_err=test_classbin(X_test,Y_test,net,distr_sol);
                        local_err=test_classbin(X_test,Y_test,net,local_sol);
                    elseif strcmp(dataset.type,'MC')
                        batch_err=test_class(X_test,vec2ind(Y_test')',net,batch_sol);
                        distr_err=test_class(X_test,vec2ind(Y_test')',net,distr_sol);
                        local_err=test_class(X_test,vec2ind(Y_test')',net,local_sol);
                    else
                        [batchNRMSE,batchNSR]=test_reg(X_test,Y_test,net,batch_sol);
                        [distrNRMSE,distrNSR]=test_reg(X_test,Y_test,net,distr_sol);
                        [localNRMSE,localNSR]=test_reg(X_test,Y_test,net,local_sol);
                    end

%Save errors, training time and Consensus iterations   
                if strcmp(dataset.type,'R')
                    NRMSE(ind,:,(jj-1)*n_fold+ii)=[0,batchNRMSE,distrNRMSE,localNRMSE];
                    NSR(ind,:,(jj-1)*n_fold+ii)=[0,batchNSR,distrNSR,localNSR];
                else
                    error(ind,:,(jj-1)*n_fold+ii)=[0,batch_err,distr_err,local_err];
                end
                
                train_time(ind,:,(jj-1)*n_fold+ii)=[0,batch_time,...
                    distr_time/n_nodi,local_time/n_nodi];
                cons_iter(ind,1+(jj-1)*n_fold+ii)=iterations;
            end
        end
        fprintf('simulation with %i nodes completed\n',n_nodi);
    end

%Prepare results for plot 
    if strcmp(dataset.type,'R')
        devstNRMSE=std(NRMSE,1,3);
        devstNSR= std(NSR,1,3);
    else
        devst_err=std(error,1,3);
    end
    
%Show and plot simulation results 
    fprintf('\nSimulation with %d nodes:\n-------------------------------------------\n',...
        vett_nodi(2));
    if strcmp(dataset.type,'R')
        fprintf('                     Avg.NRMSE:   Dev.St.:   Avg.NSR:   Dev.St.:   Training time (s):\n\n');
        fprintf('Centralized RVFL:    %.4f       %.4f     %.4f   %.4f     %.4f\n\n'...
            ,mean(NRMSE(2,2,:),3),devstNRMSE(2,2),mean(NSR(2,2,:),3),...
            devstNSR(2,2),mean(train_time(2,2,:),3));
        fprintf('Consensus RVFL:      %.4f       %.4f     %.4f   %.4f     %.4f\n\n'...
            ,mean(NRMSE(2,3,:),3),devstNRMSE(2,3),mean(NSR(2,3,:),3),...
            devstNSR(2,3),mean(train_time(2,3,:),3));
        fprintf('Local RVFL:          %.4f       %.4f     %.4f   %.4f     %.4f\n\n'...
            ,mean(NRMSE(2,4,:),3),devstNRMSE(2,4),mean(NSR(2,4,:),3),...
            devstNSR(2,4),mean(train_time(2,4,:),3));
        
        prepare_plot_batch_NRMSE;
        prepare_plot_batch_NSR;
    else
        fprintf('                     Avg.Error:   Dev.St.:   Training time (s):\n\n');
        fprintf('Centralized RVFL:    %.4f       %.4f     %.4f\n\n'...
            ,mean(error(2,2,:),3),devst_err(2,2),mean(train_time(2,2,:),3));
        fprintf('Consensus RVFL:      %.4f       %.4f     %.4f\n\n'...
            ,mean(error(2,3,:),3),devst_err(2,3),mean(train_time(2,3,:),3));
        fprintf('Local RVFL:          %.4f       %.4f     %.4f\n\n'...
            ,mean(error(2,4,:),3),devst_err(2,4),mean(train_time(2,4,:),3));

        prepare_plot_batch_err;
    end
    fprintf('Average Consensus iterations: %d\n',round(mean(cons_iter(2,2:end),2)));
    
    prepare_plot_batch_iter;
    prepare_plot_batch_time;
end