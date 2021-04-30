function [u_hat, E_u_hat, uhist, Ehist, acchist] = solve_lower(y, u0, B, alpha, gamma, epsilon, maxiter, tol)
    % Solves the lower level problem, i.e reconstructs the signal from given data
    % INPUT:
        % y : the noisy/inpainted image, a kx1 vector
        % u0 : initialisation, an nx1 vector
        % B : the inpainting matrix of size kxn
        % alpha : non-negative regularisation parameter
        % gamma : positive parameter of Huber function
        % epsilon : non-negative convex penalty parameter
        % maxiter: maximum number of iterations of GD
        % tol: tolernace for GD
    % OUTPUT:
        % u_hat : the value at which the obj function is minimised
        % E_u_hat : the value of the obj function at u_hat
        % uhist : the history of the u at each iteration
        % Ehist : the history of the objective function at each iteration
        % acchist : the history of the accuracy at at each iteration
    
    % lower level Lipschitz and strong convexity constants
    n = size(u0);
    norm_A = 2;    
    norm_B = normest(B);
    L = norm_B^2 + alpha * norm_A * (2/gamma) * norm_A + epsilon;
    mu = norm_B^2 + epsilon;
    
    % set fixed step size
    tau = 2/(L+mu);
    
    % reconstruct image by GD
    E_fun = @(u) E(u, y, B, alpha, gamma, epsilon);
    [u_hat, E_u_hat, uhist, Ehist, acchist] = gd_convex(E_fun, u0, tau, maxiter, tol, mu);
    
end
