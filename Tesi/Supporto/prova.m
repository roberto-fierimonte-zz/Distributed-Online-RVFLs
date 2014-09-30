
N = 10000;
L = 500;

A = randn(N, L);
bias = randn(L, 1);

tic;
for ii = 1:N
    for jj = 1:L
        A(ii, jj) = A(ii, jj) + bias(jj);
    end
end
fprintf('Elapsed time (double for): %.2f secs.\n', toc);

tic;
A = A + repmat(bias', N, 1);
fprintf('Elapsed time (repmat): %.2f secs.\n', toc);


tic;
A = bsxfun(@plus, A, bias');
fprintf('Elapsed time (bsxfun): %.2f secs.\n', toc);