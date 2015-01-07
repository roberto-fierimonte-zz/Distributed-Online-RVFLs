fprintf('Welcome,\nstarting distributed learning simulation, dataset %s\n\n'...
    ,dataset.name);
scelta = input('Choose simulation:\n 1 Batch learning via k-fold cross-validation\n 2 Online learning via k-fold cross-validation\n 3 RVFL parameter estimation\n 0 Quit\n:');
if scelta==0
    fprintf('Simulazion aborted, cleaning workspace...\n');
    clear;
    break;
else
    switch scelta
        case 1
            if length(dataset.X) > 10^5
                error('Cant process a big data in batch!');
            else
                %Set parameters for simulation
                K=input('\nSet hidden layer dimension (K): ');
                lambda=input('\nSet regularization parameter (lambda): ');
                max_iter=500;

                %vett_nodi=[1 5 10 15 20 25 30 35 40 45 50];
                vett_nodi=2:2:14;

                k=5;
                n_run=15;

                %Run simulation
                simulaz_batch(dataset,k,n_run,K,lambda,max_iter,vett_nodi);
            end
        case 2
            %Set parameter for simulation
            K=input('\nSet hidden layer dimension (K): ');
            lambda=input('\nSet regularization parameter (lambda): ');
            max_iter=500;
            n_nodi=8;
            
            if length(dataset.X) > 10^5
                fprintf('This dataset is big so it will not be used k-fold validation');        
                train_size=input('\nHow many training data you want to use? ');
                n_run=5;
                on=100;
                
                %Run simulation
                simulaz_bigdata(dataset,train_size,n_run,K,lambda,max_iter,n_nodi,on);
            else
                k=10;
                n_run=5;
                on=20;
                
                %Run simulation
                simulaz_online(dataset,k,n_run,K,lambda,max_iter,n_nodi,on);
            end
        case 3
            %Set parameter for simulation
            lambdavec=2.^[-10:10];
            Kmax=1000;
            
            n_fold=3;
            n_run=5;

            %Run simulation
            simulaz_param(dataset,lambdavec,Kmax,n_run,n_fold);
        otherwise
            error('Im not prepared for this! :(');
    end
end