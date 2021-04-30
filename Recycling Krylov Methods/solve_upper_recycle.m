function [alpha_hat, L_hat, alphahist, Lhist, acchist,V_recycle,H_recycle,t,gradhist] = solve_upper_recycle(U_star, Y, B, alpha0, gamma, epsilon, lower_maxiter, lower_tol, gmresmaxiter, beta, maxiter, tol,k)
    
    % Solves the upper level problem, i.e optimises alpha. The linear
    % system is solved using Recycled GMRES
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
        % maxiter : max iterations for the (upper level) GD
        % beta: backtracking parameter
        % tol : tolerance for the (upper level) GD
        % k : the number of subspace vectors to be recycled. 
    % OUTPUT:
        % alpha_hat: optimal alpha (the value at which the obj function is minimised)
        % L_hat: the value of the obj function at alpha_hat
        % alphahist: vector containing the value of alpha at each iteration
        % Lhist: the history of the objective function at each iteration
        % acchist : the history of the accuracy at at each iteration
        % V_recycle : the Arnoldi basis from the last iteration
        % H_recycle : the Hessenberg matrix from the last iteration
        % t: vector containing the time taken for each iteration
        % gradhist: the history of the gradient of the objective function at each iteration
           
    
    
        
    %Initialising
    n = size(U_star,1);
    V_recycle = zeros(n,gmresmaxiter+1);
    H_recycle = eye(gmresmaxiter+1,gmresmaxiter);
    H_recycle(end,end) = 0;
    
  
    em = selection(gmresmaxiter,gmresmaxiter);
    Hm = H_recycle(1:end-1, 1:end);
    Mat =  Hm + ((H_recycle(end, end))^2)*(Hm'\em)*em'; 
    [P_full,~] = eig(Mat);
    Pk = P_full(:,1:k);
    
        
    L_fun1 = @(alpha, V_recycle, H_recycle) L_recycle(U_star, Y, B, alpha, gamma, epsilon, lower_maxiter, lower_tol,gmresmaxiter,V_recycle,H_recycle,Pk );
    L_fun2 = @(alpha) L_val(U_star, Y, B, alpha, gamma, epsilon, lower_maxiter, lower_tol);
    [alpha_hat, L_hat, alphahist, Lhist, acchist,V_recycle,H_recycle,t,gradhist] = gd_non_convex_recycle(L_fun1, L_fun2, alpha0, beta, maxiter, tol,V_recycle, H_recycle);
    
end