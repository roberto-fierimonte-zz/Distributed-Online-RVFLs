fprintf('Benvenuto,\nAvvio simulazione di apprendimento distribuito...\n\n');
scelta = input('Scegli la modalità della simulazione:\n 1 Addestramento batch usando la k-fold cross-validation\n 2 Addestramento online usando la k-fold cross-validation\n 0 Esci\n:');
if scelta==0
    fprintf('Esco dalla simulazione e pulisco il workspace...\n');
    clear;
    break;
else
    switch scelta
        case 1
            %Questi parametri sono usati all'interno della simulazione
            %batch
            K=input('\nInserisci la dimensione dell espansione funzionale (K): ');
            lambda=input('\nInserisci il valore del parametro di regolarizzazione (lambda): ');
            n_iter=250;
            
            vett_nodi=[1 5 10 15 20 25 30 35 40 45 50];
            
            k=5;
            n=15;
            
            simulaz_classificazione_batch(X,Y,k,n,K,lambda,n_iter,vett_nodi);
        case 2
            %Questi parametri sono usati all'interno della simulazione
            %online
            K=input('\nInserisci la dimensione dell espansione funzionale (K): ');
            lambda=input('\nInserisci il valore del parametro di regolarizzazione (lambda): ');
            n_iter=250;
            n_nodi=5;
            
            k=5;
            n=15;
            on=20;
            
            simulaz_classificazione_online(X,Y,k,n,K,lambda,n_iter,n_nodi,on);
        case 3
            %Questi parametri sono usati all'interno della simulazione per
            %il test dei parametri ottimi
            err=ones(21,10,5);
            lambdavec=logspace(-10,10,21);
            for mm=1:21
                lambda=lambdavec(mm);
                for jj=1:10
                    K=100*jj;
                    for kk=1:5
                        testaparametri_class;
                        err(mm,jj,kk)=errtemp/5;
                    end
                end
            end
            err2=mean(err,3);
            surf(linspace(100,1000,10),lambdavec,err2);
            [riga,colonna]=find(err2 == min(err2(:)));
            Kmin=colonna*100; lambdamin=10^(riga-11);
            fprintf('Parametri ottimi: K = %i, lambda = %e\n', Kmin, lambdamin);
        otherwise
            error('Ancora non sono pronto per questo! :(');
    end
end
clear