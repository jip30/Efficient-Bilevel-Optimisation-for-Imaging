function norm_e = e_tv_bound(u_star,y,B,alpha,lambda,gamma,delta1,delta2)

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

% upper level gradient
[~,DuR,DuuR] = TV(u_hat,gamma);
tildeH = B'*B + alpha*DuuR + lambda*eye(n);
tildePhi = -DuR';

% estimate norm(H^-1)
[~,SB,~] = svd(B'*B);
sminB = min(diag(SB));
normHinv = 1/(sminB + lambda);

% Lipschitz constant of Phi and M
MPhi = 8 * delta1 / gamma;
MH = 16 * delta1 / gamma^2;

% estimate norm(tildew)
tildeb = u_hat - u_star;
[tildew,flag] = cgs(tildeH, tildeb, delta2, 200);
normtildew = norm(tildew);

% estimate norm(tildePhi)
%normtildePhi = 2*sqrt(n);

norm_e = normHinv * (MPhi*delta1 + norm(tildePhi)) * (delta1 + delta2 + MH*delta1*normtildew) + MPhi*delta1*normtildew;

%norm_e = normHinv * (8*delta1/gamma + norm(tildePhi)) * (delta1 + delta2 + 8*normtildew/gamma) + 8*delta1*normtildew/gamma;
%norm_e = normHinv * (8*delta1/gamma + 2*sqrt(n)) * (delta1 + delta2 + 16*delta1^2*normtildew/gamma^2) + 8*delta1*normtildew/gamma;

end
