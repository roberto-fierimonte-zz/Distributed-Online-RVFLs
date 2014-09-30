fprintf('Benvenuto,\nAvvio simulazione di apprendimento distribuito...\n\n');
scelta = input('Scegli la modalità della simulazione:\n 1 Addestramento batch usando la k-fold cross-validation\n 2 Addestramento batch scegliendo la dimensione di training e test set\n 0 Esci\n:');
K=input('\nInserisci la dimensione dell espansione funzionale (K): ');
lambda=input('\nInserisci il valore del parametro di regolarizzazione (lambda): ');
n_iter=input('\nInserisci il numero di iterazioni del consensus (n_iter): ');
n_nodi=input('\nInserisci il numero di nodi del grado (n_nodi): ');
p=input('\Inserisci la probabilità con cui vuoi che un arco sia presente nel grafo (p): ');
fprintf('Pronto!\n');
te=[0,0,0];NMSE=[0,0,0];NSR=[0,0,0];
switch scelta
    case 0
        break;
    case 1
        k=input('Inserisci la dimensione della k-fold cross-validation (k): ');
        n=input('Inserisci il numero di run della simulazione (n): ');
        fprintf('\nEffettuo una %i-fold cross-validation per testare la bontà dell algoritmo\n',k);
        for jj=1:n    
            kfoldbatch1;
        end
        fprintf('Riepilogo simulazione:\n---------------------------------------------------------------------------------------------------------------\n');
        fprintf('                                    Media NMSE:   Media NSR:   Media Training Time:\n\n');
        fprintf('Dati non distribuiti:               %.4f        %.4f      %.4f\n\n',NMSE(1)/(k*n),NSR(1)/(k*n),te(1)/(k*n));
        fprintf('Dati distribuiti con consensus:     %.4f        %.4f      %.4f\n\n',NMSE(2)/(k*n),NSR(2)/(k*n),te(2)/(k*n));
        fprintf('Dati distribuiti senza consensus:   %.4f        %.4f      %.4f\n\n',NMSE(3)/(k*n),NSR(3)/(k*n),te(3)/(k*n));
    case 2
        sceltabatch;
    otherwise
        error('Ancora non sono pronto per questo! :(');
end
clear;