function [val, grad] = E(u, y, B, alpha, gamma, epsilon)

% Evaluates the lower level objective function and its gradient
    % INPUT:
        % u : an arbitrary signal, an nx1 vector
        % y : the observed signal, an nx1 vector
        % B : the forward operator
        % alpha : non-negative regularisation parameter
        % gamma : positive parameter of Huber loss function
        % epsilon : non-negative convex penalty parameter
    % OUTPUT:
        % E : a real number, E(u_y)
        % grad_E : an n vector, \nabla E(u_{y_i})
    
    % make finite difference operator matrix
    n = size(u,1);
    e = ones(n,1);
        
    A = spdiags([-e e],0:1,n,n);
    A(n,n)=0;
    
    % pre-compute matrix multiplications (used twice)
    Au = A * u;
    Bu = B * u;
    
    rho_absAu = rho(abs(Au), gamma);
    [~, deriv_rho_Au] = rho(Au, gamma);
    
    val = 1/2 * norm(Bu - y)^2 + alpha * sum(rho_absAu) + 0.5 * epsilon * norm(u)^2;   
    grad = B' * (Bu - y) + alpha * A' * deriv_rho_Au + epsilon * u;
           
end