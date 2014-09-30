fprintf('Benvenuto,\nAvvio simulazione di apprendimento distribuito...\n\n');
scelta = input('Scegli la modalità della simulazione:\n 1 Addestramento batch usando la k-fold cross-validation\n 2 Addestramento batch scegliendo la dimensione di training e test set\n 0 Esci\n:');
K=input('\nInserisci la dimensione dell espansione funzionale (K): ');
lambda=input('\nInserisci il valore del parametro di regolarizzazione (lambda): ');
n_iter=input('\nInserisci il numero di iterazioni del consensus (n_iter): ');
n_nodi=input('\nInserisci il numero di nodi del grado (n_nodi): ');
p=input('\Inserisci la probabilità con cui vuoi che un arco sia presente nel grafo (p): ');
fprintf('Pronto!\n');
fprintf('Genero casualmente la topologia del grafo...\n');
generagrafo;
tic;
fprintf('Inizializzo la rete RVFL, K = %i, lambda = %e\n',K,lambda);
fprintf('Genero casualmente i pesi e le soglie dell espansione funzionale...\n');
[coeff,soglie]=genera_rete(K,size(X,2));
net=struct('soglie',soglie,'coeff',coeff,'dimensione',K,'lambda',lambda);
fprintf('Rete inizializzata, trascorsi %.2f secondi\n\n', toc);
t=[0,0,0];errore=[0,0,0];
switch scelta
    case 0
        break;
    case 1
        k=input('Inserisci la dimensione della k-fold cross-validation (k): ');
        n=input('Inserisci il numero di run della simulazione (n): ');
        fprintf('\nEffettuo una %i-fold cross-validation per testare la bontà dell algoritmo\n',k);
        for jj=1:n    
            kfoldclass;
        end
        fprintf('Riepilogo simulazione:\n---------------------------------------------------------------------------------------------------------------\n');
        fprintf('                                    Media errore:   Media Training Time:\n\n');
        fprintf('Dati non distribuiti:               %.4f        %.4f\n\n',errore(1)/(k*n),t(1)/(k*n));
        fprintf('Dati distribuiti con consensus:     %.4f        %.4f\n\n',errore(2)/(k*n),t(2)/(k*n));
        fprintf('Dati distribuiti senza consensus:   %.4f        %.4f\n\n',errore(3)/(k*n),t(3)/(k*n));
    case 2
        sceltabatch;
    otherwise
        error('Ancora non sono pronto per questo! :(');
end
clear;