% Memming's implementation of FA

%% Load data
load FAdata

N = size(y, 1); % number of samples

%% Remove mean
y0 = bsxfun(@minus, y, mean(y));

%% Compute sufficient statistic
Sigma_yy = y0' * y0 / N; % empirical covariance of observation

%% Set up parameters for FA
qGuess = 2; % latent dimensionality
assert(size(y, 2) > qGuess, 'p > q');
maxIteration = 5000;
tol = 1e-6;

%% Initialize by PCA
[V, D] = eig(Sigma_yy);
[dval, sidx] = sort(diag(D), 'descend');
C0 = V(sidx(1:qGuess), :)' * diag(dval(1:qGuess));
R0diag = diag(Sigma_yy);

%% Iterate EM-steps
C = C0; Rdiag = R0diag;
marginalLikelihood = nan(maxIteration+1, 1);

tic
marginalLikelihood(1) = FA_likelihood(C, Rdiag, Sigma_yy, N);
for k = 1:maxIteration
    [C, Rdiag] = EM_FA_iteration(C, Rdiag, Sigma_yy);
    marginalLikelihood(k+1) = FA_likelihood(C, Rdiag, Sigma_yy, N);
    if marginalLikelihood(k+1) - marginalLikelihood(k) < tol
	break;
    end
end
toc

%% Plot convergence
figure(1490); clf;
marginalLikelihood = marginalLikelihood(~isnan(marginalLikelihood));
plot(marginalLikelihood)
xlabel('Iteration'); ylabel('Log-likelihood'); title('Convergence');

%% Rotate the factor loadings matrix using the true loadings
if size(trueParams.C, 2) == qGuess
    [Utrue,Strue,Vtrue] = svd(trueParams.C);
    [U,S,V] = svd(C);
    Crot = U*S*Vtrue'; % All we need to do is to replace V with Vtrue
else
    Crot = C;
end

%% Plot weights
figure(5789); clf; hold on
plot(trueParams.C, '--');
plot(Crot, 'LineWidth', 2);

%% Compare to MATLAB's stat toolbox
tic
[Lambda, Psi, T, stats, F] = factoran(y, qGuess);
toc

if size(trueParams.C, 2) == qGuess
    [U,S,V] = svd(Lambda);
    LambdaRot = U*S*Vtrue';
else
    LambdaRot = Lambda;
end
plot(LambdaRot, 'o--');

%% Component plot if there are 2 factors
if qGuess == 2
    figure(9236); hold all;
    bp = zeros(3, 1);
    h = biplot(Crot, 'color', 'b');
    bp(2) = h(1);
    h = biplot(LambdaRot, 'color', 'r');
    bp(3) = h(1);
    h = biplot(trueParams.C, 'color', 0.7 * [1 1 1], 'LineWidth', 2);
    bp(1) = h(1);
    axis equal
    legend(bp, 'True', 'EM', 'factoran', 'Location', 'Best');
end
