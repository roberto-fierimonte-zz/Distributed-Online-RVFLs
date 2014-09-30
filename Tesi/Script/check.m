%for ii=1:size(beta,1)
%    for jj=1:size(beta,2)
%        if abs(gamma(ii,jj)-mean(beta(ii,:)))>10^-6
%            error('Errore: consenso non raggiunto :(');
%        else
%        end
%    end
%end

beta_avg_real = mean(beta, 2);
assert(all(all((abs(repmat(beta_avg_real, 1, size(gamma, 2)) - gamma) <= 10^-6))), 'Errore: consenso non raggiunto :(');


%for ii = 1:size(gamma, 2)
%    assert(all(abs(beta_avg_real - gamma(:, ii)) <= 10^-6), 'Errore: consenso non raggiunto :(');
%end