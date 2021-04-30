function [val, grad] = L_gmres(U_star, Y, B, alpha, gamma, epsilon, lower_maxiter, lower_tol, gmresmaxiter)
    
    % Function which evaluates the upper level objective function and its
    % gradient. Gradient is solved using standard GMRES
    % INPUT:
        % U_star : the ground truths, an nxN matrix
        % Y : the set of observed signals
        % B : the forward operator
        % alpha : non-negative regularisation parameter
        % gamma : positive parameter of Huber function
        % epsilon : non-negative convex penalty parameter
        % lower_maxiter: max number of iterations for lower level GD
        % lower_tol: tolerance for for lower level GD
        % gmresmaxiter:  max number of iterations for Recycled GMRES

    % OUTPUT:
        % L : the loss function L 
        % grad : the gradient of L wrt alpha
 
        
    n = size(U_star, 1); 
    N = size(U_star, 2); 
    A = diag([-1 * ones(1,n-1) 0]) + diag(ones(1,n-1), 1); %spdiag
    
    val = zeros(N,1);
    grad = zeros(N,1);
    
    u0 = zeros(n,1);
    w0 = zeros(n,1);
    
    for i = 1:N
        
        u_hat = solve_lower(Y(:,i), u0, B, alpha, gamma, epsilon, lower_maxiter, lower_tol);
        u_star = U_star(:,i);
        
        % compute L
        val(i) = 1/2 * norm(u_hat - u_star)^2;
        
        if nargout > 1
            %compute grad_L
            [~, deriv_rho_Au_hat, double_rho] = rho(A*u_hat,gamma);
            DuuE = B' * B + alpha * A'* diag(double_rho) * A + epsilon * eye(n);
            DauE = A' * deriv_rho_Au_hat;
            w = u_hat - u_star;

            w = gmres_basic( DuuE, w, w0, gmresmaxiter);
            grad(i) = -w' * DauE;
        end 
         
      
    end
    
    val = mean(val);
    grad = mean(grad);
    
end

