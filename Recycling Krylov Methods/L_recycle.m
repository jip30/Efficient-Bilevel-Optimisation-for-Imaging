function [val, grad, V_recycle, H_recycle] = L_recycle(U_star, Y, B, alpha, gamma, epsilon, lower_maxiter, lower_tol, gmresmaxiter,V_old,H_old,Ek)

    % Function which evaluates the upper level objective function and its
    % gradient. Gradient is solved using Recycled GMRES
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
        % V_old: old Arnoldi basis from previous iteration
        % H_old: old Hessenberg matrix from previous iteration
        % Ek: determines which vectors will be recycled.
    % OUTPUT:
        % L : the loss function L 
        % grad : the gradient of L wrt alpha
        % V_recycle: the newly generated Arnoldi basis
        % H_recycle: The newly generate Hesseneberg matrix
            
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
          

            [w,V_recycle, H_recycle, ~] = gmres_recycle( DuuE, w, w0, V_old, H_old, Ek, gmresmaxiter);
            grad(i) = -w' * DauE;
        end 
        
      
    end
    
    val = mean(val);
    grad = mean(grad);
    
end

