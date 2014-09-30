fprintf('Benvenuto,\nAvvio simulazione di apprendimento distribuito...\n\n');
scelta = input('Scegli la modalità della simulazione:\n 1 Addestramento batch usando la k-fold cross-validation\n 2 Addestramento batch scegliendo la dimensione di training e test set\n:');
K=input('\nInserisci la dimensione dell espansione funzionale (K): ');
lambda=input('\nInserisci il valore del parametro di regolarizzazione (lambda): ');
n_iter=input('\nInserisci il numero di iterazioni del consensus (n_iter): ');
n_nodi=input('\nInserisci il numero di nodi del grado (n_nodi): ');
%p=input('\Inserisci la probabilità con cui vuoi che un arco sia presente nel grafo (p): ');
fprintf('Pronto!\n');
fprintf('Genero casualmente la topologia del grafo...\n');
generagrafo;
tic;
fprintf('Inizializzo la rete RVFL, K = %i, lambda = %e\n',K,lambda);
fprintf('Genero casualmente i pesi e le soglie dell espansione funzionale...\n');
[coeff,soglie]=genera_rete(K,size(X,2));
net=struct('soglie',soglie,'coeff',coeff,'dimensione',K,'lambda',lambda);
fprintf('Rete inizializzata, trascorsi %.2f secondi\n\n', toc);
NMSE=[0,0,0];
NSR=[0,0,0];
switch scelta
    case 1
        kfoldbatch;
    case 2
        sceltabatch;
    otherwise
        error('Ancora non sono pronto per questo! :(');
end
clear;