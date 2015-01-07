function [] = simulaz_param(dataset,lambdavec,Kmax,n_iter,n_fold)

    X=dataset.X; Y=dataset.Y;
    Kvec=50:50:Kmax;
    err=zeros(length(lambdavec),length(Kvec),n_iter);

    for mm=1:length(lambdavec)
        
        lambda=lambdavec(mm);
        
        for jj=1:length(Kvec)
            
            K=Kvec(jj);
            
            for kk=1:n_iter
                
                if strcmp(dataset.type,'BC')
                    c = cvpartition(Y,'kfold',n_fold);
                else
                    c = cvpartition(size(X,1),'kfold',n_fold);
                end
                net=generate_RVFL(K,size(X,2),lambda);
                errtemp=0;

                for ii = 1:c.NumTestSets

                    X_train=X(c.training(ii),:);
                    Y_train=Y(c.training(ii),:);

                    c = cvpartition(size(X_train,1),'kfold',50);
                    sol=distributed_rvfl_seriale(X_train,Y_train,net,ones(50)/50,500,c);
                    
                    if strcmp(dataset.type,'BC')
                        errtemp=errtemp + test_classbin(X_train,Y_train,net,sol);
                    elseif strcmp(dataset.type,'MC')
                        errtemp=errtemp + test_class(X_train,vec2ind(Y_train')',net,sol);
                    else
                        errtemp=errtemp + test_reg(X_train,Y_train,net,sol);
                    end
                end

                err(mm,jj,kk)=errtemp/n_fold;
            end
            
        end
    end
    figure();
    figshift;
    err=mean(err,3);
    surf(Kvec,lambdavec, err);
    [riga,colonna]=find(err == min(err(:)));
    Kmin=Kvec(colonna); lambdamin=lambdavec(riga);
    fprintf('Parametri ottimi: K = %i, lambda = %e\n', Kmin, lambdamin);
end

