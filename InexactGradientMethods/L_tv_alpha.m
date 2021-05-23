function [val,grad] = L_tv_alpha(u_star,y,B,alpha,lambda,gamma,delta1,delta2)

% Compute val=L(u_hat(lambda)) and grad=\nabla L(u_hat(lambda)) with
% TV+Tikhonov regularisation using approximate (lower-level / linear system) solutions

n = numel(u_star);

% Lipschitz and convexity constants
normA = 2; % norm of 1D finite difference operator
normB = normest(B);
L = normB^2 + alpha * normA * (2/gamma) * normA + lambda;
mu = normB^2 + lambda;

% solve lower level problem to delta1 accuracy
E_fun = @(u) E_tv(u,y,B,alpha,lambda,gamma);
u0 = zeros(n,1);
tol = delta1^2;
maxiter = 250000;
u_hat = gd_convex(E_fun,u0,L,mu,tol,maxiter);

% upper level objective
val = 0.5 * norm(u_hat-u_star)^2;

% upper level gradient
[~,DuR,DuuR] = TV(u_hat,gamma);
tildePhi = -DuR';
tildeH = B'*B + alpha*DuuR + lambda*eye(n);
tildeb = u_hat - u_star;
[tildew,flag] = cgs(tildeH, tildeb, delta2, 200); 
grad = tildePhi * tildew;

end
