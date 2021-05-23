% n = 1000000;
% gamma = 0.001;
% Lvals = zeros(n,1);
% 
% for i = 1:n
%    x = randn(10,1);
%    y = randn(10,1);
%    [~,drhox,~] = rho(x,gamma);
%    [~,drhoy,~] = rho(y,gamma);
%    num = norm(drhox-drhoy);
%    %num = norm(rho(x,gamma)-rho(y,gamma));
%    den = norm(x-y);
%    Lvals(i) = num/den;
% end
% 
% max(Lvals)


% create signals
n = 64;
k = 32;
sigma = 0.03;
alpha = 0.01;
lambda = 0.02;
gamma = 0.001;
[u_star,y,B] = make_undersampled_signal(n,k,sigma);

% find optimal reconstruction
normA = 2; % norm of 1D finite difference operator
normB = normest(B);
L = normB^2 + alpha * normA * (2/gamma) * normA + lambda; % gamma=0.002
mu = normB^2 + lambda;
E_fun = @(u) E_tv(u,y,B,alpha,lambda,gamma);
u0 = zeros(n,1);
tol = 1e-6; 
maxiter = 250000;
u_hat = gd_convex(E_fun,u0,L,mu,tol,maxiter);

[~,DuR,DuuR] = TV(u_hat,gamma);
tildePhi = -DuR';
tildeH = B'*B + alpha*DuuR + lambda*eye(n);
norm(inv(tildeH))

[UB,SB,VB] = svd(B'*B);
sminB = min(diag(SB));
sminB
1/(sminB + lambda)