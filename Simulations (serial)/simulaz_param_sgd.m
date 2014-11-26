function [] = simulaz_param_sgd(K,lambda,dataset,Cvec,alfazerovec,n_iter,n_fold)

    X=dataset.X; Y=dataset.Y; m=size(Y,2);
    err=zeros(length(Cvec),length(alfazerovec),n_iter);

    for mm=1:length(Cvec)
        
        C=Cvec(mm);
        
        for jj=1:length(alfazerovec)
            
            alfa_zero=alfazerovec(jj);
            
            for kk=1:n_iter
                
                if strcmp(dataset.type,'BC')
                    c = cvpartition(Y,'kfold',n_fold);
                else
                    c = cvpartition(size(X,1),'kfold',n_fold);
                end
                net=generate_RVFL(K,size(X,2),lambda);
                errtemp=0;

                for ii = 1:c.NumTestSets

                    batch=50;
                    X_train=X(c.training(ii),:);
                    Y_train=Y(c.training(ii),:);
                    X_test=X(c.test(ii),:);
                    Y_test=Y(c.test(ii),:);
                    start=1;
                    n_online=ceil(size(X_train,1)/batch);
                    sol=zeros(K,m);
                    
                    for nn=1:n_online
                        if nn==n_online
                            Xtemp=X_train(start:end,:);
                            Ytemp=Y_train(start:end,:);
                        else
                            Xtemp=X_train(start:(start+batch-1),:);
                            Ytemp=Y_train(start:(start+batch-1),:);
                        end
                        
                        sol=rvfl_sgd(Xtemp,Ytemp,C,alfa_zero,sol,nn,net);
                        
                        start=(start+batch);
                    end
                        if strcmp(dataset.type,'BC')
                            errtemp=errtemp + test_classbin(X_test,Y_test,net,sol);
                        elseif strcmp(dataset.type,'MC')
                            errtemp=errtemp + test_class(X_test,vec2ind(Y_test')',net,sol);
                        else
                            errtemp=errtemp + test_reg(X_test,Y_test,net,sol);
                        end
                end

                err(mm,jj,kk)=errtemp/n_fold;
            end
            
        end
    end
    
    err=mean(err,3);
    surf(Cvec,alfazerovec, err);
    [Cmin,alfazeromin]=find(err == min(err(:)));
    fprintf('Parametri ottimi: C = %i, alfa zero = %e\n', Cmin, alfazeromin);
end

