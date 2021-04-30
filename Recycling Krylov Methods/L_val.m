function value = L_val(U_star, Y, B, alpha, gamma, epsilon, lower_maxiter, lower_tol)

    % Function which computes the vallue of the upper level objective function.
    % Only computes the value of the objective function, not its gradient.
    % INPUT:
        % U_star : the ground truths, an nxN matrix
        % Y : the set of observed signals
        % B : the forward operator
        % alpha : non-negative regularisation parameter
        % gamma : positive parameter of Huber function
        % epsilon : non-negative convex penalty parameter
        % lower_maxiter: max number of iterations for lower level GD
        % lower_tol: tolerance for for lower level GD

    % OUTPUT:
        % L : the loss function L 
        % grad : the gradient of L wrt alpha   
        
    n = size(U_star, 1); 
    N = size(U_star, 2); 
       
    value = zeros(N,1);
 
    u0 = zeros(n,1);
    
    for i = 1:N
        
        u_hat = solve_lower(Y(:,i), u0, B, alpha, gamma, epsilon, lower_maxiter, lower_tol);
        u_star = U_star(:,i);
        
        % compute L
        value(i) = 1/2 * norm(u_hat - u_star)^2;
        
    end 
    
    value = mean(value);
    
end

