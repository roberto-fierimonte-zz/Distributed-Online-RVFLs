function [] = simulaz_classbin_param(X,Y,lambda_vec,Kmax,n_iter,n_fold)

    err=zeros(size(lambda_vec,2),Kmax/100,n_iter);

    for mm=1:size(lambda_vec,2)
        
        lambda=lambda_vec(mm);
        
        for jj=1:Kmax/100
            
            K=100*jj;
            
            for kk=1:n_iter
                
                c = cvpartition(Y,'kfold',n_fold);
                [coeff,soglie]=genera_rete(K,size(X,2));
                net=struct('soglie',soglie,'coeff',coeff,'dimensione',K,'lambda',lambda);
                errtemp=0;

                for ii = 1:c.NumTestSets

                    X_train=X(c.training(ii),:);
                    Y_train=Y(c.training(ii),:);
                    X_test=X(c.test(ii),:);
                    Y_test=Y(c.test(ii),:);

                    sol=rvflreg(X_train,Y_train,net);
                    errtemp=errtemp + test_classbin(X_test,Y_test,net,sol);
                end

                err(mm,jj,kk)=errtemp/n_fold;
            end
            
        end
    end
    
    err=mean(err,3);
    surf(linspace(100,Kmax,Kmax/100),lambda_vec,err);
    [riga,colonna]=find(err == min(err(:)));
    Kmin=colonna*100; lambdamin=lambda_vec(riga);
    fprintf('Parametri ottimi: K = %i, lambda = %e\n', Kmin, lambdamin);
end

