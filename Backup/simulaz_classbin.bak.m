fprintf('Benvenuto,\nAvvio simulazione di apprendimento distribuito...\n\n');
scelta = input('Scegli la modalità della simulazione:\n 1 Addestramento batch usando la k-fold cross-validation\n 2 Addestramento online usando la k-fold cross-validation\n 0 Esci\n:');
if scelta==0
    fprintf('Simulazione annullata, esco e pulisco il workspace...\n');
    clear;
    break;
else
    switch scelta
        case 1
            K=input('\nInserisci la dimensione dell espansione funzionale (K): ');
            lambda=input('\nInserisci il valore del parametro di regolarizzazione (lambda): ');
            n_iter=250; %n_iter=input('\nInserisci il numero di iterazioni del consensus (n_iter): ');

            vett_nodi=[1 5 10 15 20 25 30 35 40 45 50];

            k=5; %k=input('Inserisci la dimensione della k-fold cross-validation (k): ');
            n=15; %n=input('Inserisci il numero di run della simulazione (n): ');
            
            simulaz_classbin_batch(X,Y,k,n,K,lambda,n_iter,vett_nodi);
        case 2
            K=input('\nInserisci la dimensione dell espansione funzionale (K): ');
            lambda=input('\nInserisci il valore del parametro di regolarizzazione (lambda): ');
            n_iter=250; %n_iter=input('\nInserisci il numero di iterazioni del consensus (n_iter): ');
            n_nodi=5; %n_nodi=input('\nInserisci il numero di nodi del grado (n_nodi): ');
            
            k=5; %k=input('Inserisci la dimensione della k-fold cross-validation (k): ');
            n=15; %n=input('Inserisci il numero di run della simulazione (n): ');
            on=20; %on=input('Inserisci quanti dati vuoi usare ad ogni iterazione: ');
            
            simulaz_classbin_online(X,Y,k,n,K,lambda,n_iter,n_nodi,on);
        case 3
            err=ones(25,10,5);
            lambdavec=logspace(-6,6,25);
            for mm=1:21
                lambda=lambdavec(mm);
                for jj=1:10
                    K=100*jj;
                    for kk=1:5
                        testaparametri_classbin;
                        err(mm,jj,kk)=errtemp/5;
                    end
                end
            end
            err2=mean(err,3);
            surf(linspace(100,1000,10),lambdavec,err2);
            [riga,colonna]=find(err2 == min(err2(:)));
            Kmin=colonna*100; lambdamin=lambdavec(riga);
            fprintf('Parametri ottimi: K = %i, lambda = %e\n', Kmin, lambdamin);
        otherwise
            error('Ancora non sono pronto per questo! :(');
    end
end
clear;