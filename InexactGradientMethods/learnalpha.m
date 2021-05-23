function [alpha_vals, L_vals, delta_vals, norm_gradL_2_vals, u_best, lower_iters, linear_iters] = learnalpha(u_star, y, B, lambda, gamma, alpha0, acc_k_fun, delta0, tol, maxiter, tau0, alpha_tau, beta1_tau, beta2_tau, beta1_delta, beta2_delta)

% INPUT
% u_star (nx1): ground truth signal
% y (kx1): undersampled noisy signal
% B (kxn): forward operator matrix, e.g. undersampling
% lambda (>0): the Tikhonov regularisation parameter
% gamma (>0): the TV smoothing parameter
% alpha0 (>0): the initial parameter
% acc_k_fun (function of k): a function from N_0 -> R+ that specifies the gradient accuracy for iteration k (k -> eps_k)
% delta0 (2x1): the initial values for delta1 and delta2
% tol (>0): the stopping criterion, stop when norm(grad(L))^2 < tol
% maxiter: maximum number of iterations to solve upper level problem
% tau0 (>0): the initial step-size
% alpha_tau (0<alpha_tau<1): the alpha value for the Armijo condition in step-size backtracking
% beta1_tau (0<beta1_tau<1): the beta1 value in step-size backtracking
% beta2_tau (>1): the beta1 value in step-size backtracking
% beta1_delta (0<beta1_delta<1): the beta1 value in delta backtracking
% beta2_tau (>1): the beta1 value in delta backtracking

% dimension of problem
n = size(u_star,1);

% upper level objective function
L = @(u) (1/2) * norm(u - u_star)^2;

% preallocate arrays to store outputs 
alpha_vals = zeros(maxiter+1,1);
L_vals = zeros(maxiter+1,1);
delta_vals = zeros(maxiter+1,2);
norm_gradL_2_vals = zeros(maxiter+1,1);
lower_iters = 0;
linear_iters = 0;

% compute Lipschitz and strong convexity constants of lower level problem
normB = normest(B);
M = @(alpha) normB^2 + 8*alpha/gamma + lambda; % depends on alpha
mu = normB^2 + lambda;

% compute C_H, M_Phi, M_H
[~,SBB,~] = svd(B'*B);
C_H = 1/(min(diag(SBB))+lambda);
M_Phi = 8/gamma;
M_H = 16/gamma^2;

%%%%% THE ALGORITHM %%%%%

alpha = alpha0;
delta = delta0;
tau = tau0;
u0 = zeros(n,1);
k = 0;
norm_gradL_alpha_2 = tol + 1;

while k <= maxiter && norm_gradL_alpha_2 > tol
    
    % required accuracy for kth iteration
    acc_k = acc_k_fun(k);
    
    % fictional initial value for error in the gradient
    e_tilde = acc_k + 1;
    
    % find delta such that required accuracy is reached
    while e_tilde > acc_k
    
        % solve lower level problem to accuracy delta1, initialised at u0
        E_fun = @(u) E_tv(u, y, B, alpha, lambda, gamma);
        [u_tilde, ~, ~, lower_iters_k] = gd_convex(E_fun, u0, M(alpha), mu, delta(1)^2, 250000); 

        % compute tildePhi, tildeH, tildeb
        [~, DuR, DuuR] = TV(u_tilde, gamma);
        tildePhi = -DuR';
        tildeH = B'*B + alpha*DuuR + lambda*eye(n);
        tildeb = u_tilde - u_star;
        [tildew, ~, ~, linear_iters_k] = cgs(tildeH, tildeb, delta(2)/norm(tildeb), 200); % cgs uses relative residual ||Ax-b||/||b|| as stopping criterion

        % estimate the error in the gradient
        e_tilde = acc_bound(C_H, M_Phi, M_H, delta, tildePhi, tildew);
        
        % count number of iterations
        lower_iters = lower_iters + lower_iters_k;
        linear_iters = linear_iters + linear_iters_k;
        
        if e_tilde > acc_k        
            delta = beta1_delta .* delta; % make delta smaller so that accuracies are better
            u0 = u_tilde; % use current reconstruction as initialisation for lower level problem
        end

    end
    
    
    
    % compute the objective function and its gradient
    L_alpha = L(u_tilde);
    gradL_alpha = tildePhi * tildew;
    norm_gradL_alpha_2 = norm(gradL_alpha)^2;
    
    % store values in arrays
    alpha_vals(k+1) = alpha;
    L_vals(k+1) = L_alpha;
    delta_vals(k+1,:) = delta;
    norm_gradL_2_vals(k+1) = norm_gradL_alpha_2;
    
    if norm_gradL_alpha_2 < tol
        
        fprintf('Stopping criterion attained after %.f iterations. \n', k')
        
        % trim size of output arrays
        alpha_vals = alpha_vals(1:k+1);
        L_vals = L_vals(1:k+1);
        delta_vals = delta_vals(1:k+1,:);
        norm_gradL_2_vals = norm_gradL_2_vals(1:k+1);
        
        % obtain the optimal reconstruction (to high accuracy)
        u_best = gd_convex(E_fun, u0, M(alpha), mu, 1e-10, 250000);
        
        return
    
    else    

        % make GD update
        alpha = alpha - tau * gradL_alpha;

        % move to next iteration, with larger delta and current reconstruction as initialisation
        k = k+1;
        delta = beta2_delta .* delta;
        u0 = u_tilde;
        
    end

end

% obtain the optimal reconstruction (to high accuracy)
u_best = gd_convex(E_fun, u0, M(alpha), mu, 1e-10, 250000);

end




%         % find step size using two-way backtracking
% 
%         % while Armijo condition not satisfied, reduce tau
%         alpha_prop = alpha - tau * gradL_alpha;
%         u_tilde_prop = gd_convex(E_fun, u_tilde, M(alpha_prop), mu, delta(1)^2, 250000);
%         
%         while L(u_tilde_prop) - L_alpha > alpha_tau * tau * norm_gradL_alpha_2
%             
%             tau = beta1_tau * tau
%             alpha_prop = alpha - tau * gradL_alpha;
%             u_tilde_prop = gd_convex(E_fun, u_tilde, M(alpha_prop), mu, delta(1)^2, 250000);
%             
%         end
% 
%         % try a larger tau
%         tau = beta2_tau * tau;
%         u_tilde_prop = gd_convex(E_fun, u0, M(alpha - tau*gradL_alpha), mu, delta(1)^2, 250000);
% 
%         % if Armijo condition no longer satisfied, revert back to previously found tau
%         if L(u_tilde_prop) - L_alpha > alpha_tau * tau * norm(gradL_alpha)^2
%             tau = tau / beta2_tau;
%         end

