fprintf('Benvenuto,\nAvvio simulazione di apprendimento distribuito...\n\n');
scelta = input('Scegli la modalità della simulazione:\n 1 Addestramento batch usando la k-fold cross-validation\n 2 Addestramento online usando la k-fold cross-validation\n 0 Esci\n:');
if scelta==0
    fprintf('Simulazione annullata, esco e pulisco il workspace...\n');
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
            
            simulaz_classbin_batch(X,Y,k,n,K,lambda,n_iter,vett_nodi);
        case 2
            %Questi parametri sono usati all'interno della simulazione
            %online
            K=input('\nInserisci la dimensione dell espansione funzionale (K): ');
            lambda=input('\nInserisci il valore del parametro di regolarizzazione (lambda): ');
            n_iter=250;
            n_nodi=5;
            
            k=5;
            n=5;
            on=20;
            
            simulaz_classbin_online(X,Y,k,n,K,lambda,n_iter,n_nodi,on);
        case 3
            %Questi parametri sono usati all'interno della simulazione per
            %il test dei parametri ottimi
            lambdavec=logspace(-10,10,21);
            Kmax=1000;
            
            n_fold=5;
            n_iter=5;

            simulaz_classbin_param(X,Y,lambdavec,Kmax,n_iter,n_fold);
        otherwise
            error('Ancora non sono pronto per questo! :(');
    end
end
clear;