function [alpha_hat, L_hat, alphahist, Lhist, acchist,t,gradhist] = solve_upper_basic(U_star, Y, B, alpha0, gamma, epsilon, lower_maxiter, lower_tol, gmresmaxiter, beta, maxiter, tol)
    
    % Solves the upper level problem, i.e optimises alpha. The linear
    % system is solved using standard GMRES
    % INPUT:
        % U_star : the set of ground truths, nxN matrix
        % Y : the set of observed signals, nxN matrix
        % B : the blurring operator
        % alpha0 : initialisation for regularisation parameter
        % gamma : positive parameter of Huber function
        % epsilon : non-negative convex penalty parameter
        % lower_maxiter : max iterations for the lower level GD
        % lower_tol : tolerance for the lower level GD
        % gmresmaxiter : max iterations for solving linear system by GMRES
        % cgstol : tolerance for solving linear system by CG
        % maxiter : max iterations for the (upper level) GD
        % beta: backtracking parameter
        % tol : tolerance for the (upper level) GD
    % OUTPUT:
        % alpha_hat: optimal alpha (the value at which the obj function is minimised)
        % L_hat: the value of the obj function at alpha_hat
        % alphahist: vector containing the value of alpha at each iteration
        % Lhist: the history of the objective function at each iteration
        % acchist : the history of the accuracy at at each iteration
        % t: vector containing the time taken for each iteration
        % gradhist: the history of the gradient of the objective function at each iteration
           
    
    L_fun1 = @(alpha) L_gmres(U_star, Y, B, alpha, gamma, epsilon, lower_maxiter, lower_tol, gmresmaxiter);
    L_fun2 = @(alpha) L_val(U_star, Y, B, alpha, gamma, epsilon, lower_maxiter, lower_tol);
    
    [alpha_hat, L_hat, alphahist, Lhist, acchist,t,gradhist] = gd_non_convex_basic(L_fun1,L_fun2, alpha0, beta, maxiter, tol);
end