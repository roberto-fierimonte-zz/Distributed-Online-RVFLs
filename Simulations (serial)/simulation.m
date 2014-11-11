fprintf('Benvenuto,\nAvvio simulazione di apprendimento distribuito, dataset %s\n\n',dataset.name);
scelta = input('Scegli la modalità della simulazione:\n 1 Addestramento batch usando la k-fold cross-validation\n 2 Addestramento online usando la k-fold cross-validation\n 0 Esci\n:');
if scelta==0
    fprintf('Simulazione annullata, esco e pulisco il workspace...\n');
    clear;
    break;
else
    switch scelta
        case 1
            if length(dataset.X) > 10^5
                error('Non posso processare un Big Data in batch!');
            else
                %Questi parametri sono usati all'interno della simulazione
                %batch
                K=input('\nInserisci la dimensione dell espansione funzionale (K): ');
                lambda=input('\nInserisci il valore del parametro di regolarizzazione (lambda): ');
                max_iter=500;

                vett_nodi=[1 5 10 15 20 25 30 35 40 45 50];
                %vett_nodi=2:2:14;

                k=5;
                n_run=15;
  
                if strcmp(dataset.type,'R')
                    simulaz_reg_batch(dataset,k,n_run,K,lambda,max_iter,vett_nodi);
                else
                    simulaz_class_batch(dataset,k,n_run,K,lambda,max_iter,vett_nodi);
                end
            end
        case 2
            %Questi parametri sono usati all'interno della simulazione
            %online
            K=input('\nInserisci la dimensione dell espansione funzionale (K): ');
            lambda=input('\nInserisci il valore del parametro di regolarizzazione (lambda): ');
            max_iter=500;
            n_nodi=8;
            
            if length(dataset.X) > 10^5
                fprintf('Questo è un problema di Big Data e quindi non sarà usata la k-fold validation');        
                train_size=input('\nQuanti dati di training vuoi usare? ');
                n_run=5;
                on=800;
                if strcmp(dataset.type,'R')
                    simulaz_reg_bigdata(dataset,train_size,n_run,K,lambda,max_iter,n_nodi,on);
                else
                    simulaz_class_bigdata(dataset,train_size,n_run,K,lambda,max_iter,n_nodi,on);
                end
            else
                k=10;
                n_run=15;
                on=80;
                if strcmp(dataset.type,'R')
                    simulaz_reg_online(dataset,k,n_run,K,lambda,max_iter,n_nodi,on);
                else
                    simulaz_class_online(dataset,k,n_run,K,lambda,max_iter,n_nodi,on);
                end
            end
        case 3
            %Questi parametri sono usati all'interno della simulazione per
            %il test dei parametri ottimi
            lambdavec=2.^[-10:10];
            Kmax=1000;
            
            n_fold=3;
            n_run=5;

            simulaz_param(dataset,lambdavec,Kmax,n_run,n_fold);
        otherwise
            error('Ancora non sono pronto per questo! :(');
    end
end
clear;