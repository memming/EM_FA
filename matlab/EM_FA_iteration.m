function [C, Rdiag] = EM_FA_iteration(C, Rdiag, Sigma_yy)
% EM-iteration for Factor Analysis (Gaussian)
% [C, Rdiag] = EM_FA_iteration(C, Rdiag, Sigma_yy)
%
% Input
%   C: (p x q) factor loadings matrix
%   Rdiag: (p x 1) variance of observation noise
%   Sigma_yy: (p x p) empirical covariance matrix
%
% Output
%   C: updated C
%   Rdiag: updated Rdiag

Rinv = diag(1./Rdiag);
Lambda = inv(eye(size(C,2)) + C'*Rinv*C);
delta = Lambda*(C' * Rinv);
Sd = Sigma_yy * delta';
C =  Sd / (Lambda + delta * Sd);
Rdiag = diag(Sigma_yy - Sd * C');
