function l = FA_likelihood(C, Rdiag, Sigma_yy, N)
% Marginal log-likelihood per sample (+constant) for Factor Analysis model
% l = FA_likelihood(C, Rdiag, Sigma_yy, N)

Sigma = C*C' + diag(Rdiag);
ld = 2*sum(log(diag(chol(Sigma))));
l = -N/2 * (ld + trace(Sigma_yy / Sigma));
