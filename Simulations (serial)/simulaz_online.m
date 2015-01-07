function [] = simulaz_online(dataset,n_fold,n_run,K,lambda,n_iter,n_nodi,batch)
    
%Import dataset
    X=dataset.X; Y=dataset.Y;
    X=[X;X]; Y=[Y;Y];
    m=size(Y,2);
    train_time=zeros(4,n_run*n_fold);
    
    for jj=1:n_run
        
        if strcmp(dataset.type,'BC')
            c = cvpartition(Y,'kfold',n_fold);
        else
            c = cvpartition(size(X,1),'kfold',n_fold);
        end
        
%Generate network topology and RVFL structure
        generagrafo;
        net=generate_RVFL(K,size(X,2),lambda);
        
%Initialize training and test sets, initialize iterations count and solutions
        for ii = 1:c.NumTestSets

            X_train=X(c.training(ii),:);
            Y_train=Y(c.training(ii),:); 
            X_test=X(c.test(ii),:);
            Y_test=Y(c.test(ii),:);

            start=1;
            n_online=ceil(size(X_train,1)/batch);
            
            centr_sol=zeros(K,m); 
            distr_sol=zeros(K,m); 
            local_sol=zeros(K,m); 
            lms_sol=zeros(K,m);
            
            K_centr=net.lambda*eye(K);           
            K_dist=repmat(K_centr,1,1,n_nodi); 
            K_local=K_dist;

%Get current chunk
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

%Compute solutions                     
                    tic;
                    [centr_sol,K_centr]=rvfl_rls...
                        (K_centr,Xtemp,Ytemp,centr_sol,net);
                    train_time(1,(jj-1)*n_fold+ii)=train_time(1,(jj-1)*n_fold+ii)...
                        +toc;
                    
                    tic;
                    [distr_sol,K_dist]=distributed_rvfl_rls_seriale...
                        (K_dist,Xtemp,Ytemp,distr_sol,net,W,n_iter,distributor);
                    train_time(2,(jj-1)*n_fold+ii)=train_time(2,(jj-1)*n_fold+ii)...
                        +toc;
                    
                    tic;
                    [local_sol,K_local]=distributed_rvfl_rls_seriale...
                        (K_local,Xtemp,Ytemp,local_sol,net,W,0,distributor);
                    train_time(3,(jj-1)*n_fold+ii)=train_time(3,(jj-1)*n_fold+ii)...
                        +toc;

                    tic;
                    lms_sol=distributed_rvfl_lms_seriale...
                        (Xtemp,Ytemp,lms_sol,net,W,n_iter,distributor,kk);
                    train_time(4,(jj-1)*n_fold+ii)=train_time(4,(jj-1)*n_fold+ii)...
                        +toc;

%Compute test errors                     
                    if strcmp(dataset.type,'BC')
                        centr_err(kk,ii)=test_classbin(X_test,Y_test,net,centr_sol);
                        distr_err(kk,ii)=test_classbin(X_test,Y_test,net,distr_sol);
                        local_err(kk,ii)=test_classbin(X_test,Y_test,net,local_sol);
                        lms_err(kk,ii)=test_classbin(X_test,Y_test,net,lms_sol);
                    elseif strcmp(dataset.type, 'MC')
                        centr_err(kk,ii)=test_class(X_test,vec2ind(Y_test')',net,centr_sol);
                        distr_err(kk,ii)=test_class(X_test,vec2ind(Y_test')',net,distr_sol);
                        local_err(kk,ii)=test_class(X_test,vec2ind(Y_test')',net,local_sol);
                        lms_err(kk,ii)=test_class(X_test,vec2ind(Y_test')',net,lms_sol);
                    else
                        centr_err(kk,ii)=test_reg(X_test,Y_test,net,centr_sol);
                        distr_err(kk,ii)=test_reg(X_test,Y_test,net,distr_sol);
                        local_err(kk,ii)=test_reg(X_test,Y_test,net,local_sol);
                        lms_err(kk,ii)=test_reg(X_test,Y_test,net,lms_sol);
                    end

%Iterate                     
                    start=(start+batch);
                else
                    n_online=n_online-1;
                end
            end  
        end
        
%Save errors         
        error(:,1,jj)=mean(centr_err,2);
        error(:,2,jj)=mean(distr_err,2);
        error(:,3,jj)=mean(local_err,2);
        error(:,4,jj)=mean(lms_err,2);
        fprintf('run %i of %i completed\n',jj,n_run);
    end

%Prepare results for plot     
    if strcmp(dataset.type,'BC')
        baseline=50*ones(1,4,n_run);
    elseif strcmp(dataset.type,'MC')
        baseline=100*(m-1)/m*ones(1,4,n_run);
    else
        baseline=ones(1,4,n_run);
    end
    
    error=[baseline; error];
    devst=std(error,1,3);

%Show simulations results     
    fprintf('Simulation results:\n-------------------------------------------\n');
    fprintf('                           Avg.Error:   St.Dev.:  Total Time (s):\n\n');
    fprintf('Centralized RVFL:          %.4f      %.4f    %.4f\n\n'...
        ,mean(error(n_online+1,1),3),devst(n_online,1),mean(train_time(1,:)));
    fprintf('RLS-Consensus RVFL:        %.4f      %.4f    %.4f\n\n'...
        ,mean(error(n_online+1,2),3),devst(n_online,2),mean(train_time(2,:)));
    fprintf('LMS-Consensus RVFL:        %.4f      %.4f    %.4f\n\n'...
        ,mean(error(n_online+1,4),3),devst(n_online,4),mean(train_time(4,:)));
    fprintf('RLS-Local RVFL:            %.4f      %.4f    %.4f\n\n'...
        ,mean(error(n_online+1,3),3),devst(n_online,3),mean(train_time(3,:)));
 
%Plot error     
    prepare_plot_online;
end