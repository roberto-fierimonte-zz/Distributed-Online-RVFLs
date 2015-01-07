%p=0.5;
%p=input('Inserisci la probabilità che un arco sia presente tra 2 nodi del grafo:');
t = RandomTopology(n_nodi, 0.5);
%t = LinearTopology(n_nodi,4);
%t = FullyConnectedTopology(n_nodi);
%t = CyclicLatticeTopology(n_nodi,2);
%t = SmallWorldTopology(n_nodi,6,0.15);
%t = ScaleFreeTopology(n_nodi,1,2);
t = t.initialize();
A=t.W;
degrees=sum(A);
A = A./(max(degrees) +1);
for ii=1:n_nodi
    A(ii,ii)= 1 - degrees(ii)/(max(degrees) + 1);
end
W=A;