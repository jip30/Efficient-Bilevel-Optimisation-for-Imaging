function [x, fval, xhist, fhist, acchist] = gd_convex(fun, x0, tau, maxiter, tol, mu)

    % Gradient descent with fixed step and a-posteriori stopping criterion
    % for a convex objective function.
    % INPUT:
        % fun : the objective function (taking just one argument) with two outputs [f, grad_f]
        % x0 : the initial value
        % tau : the step size
        % maxiter : maximum number of iterations 
        % tol : >0, required accuracy - stop when ||grad_fun(x)||^2 / mu^2 < tol 
        % mu : the strong convexity constant (used in the stopping criterion)
    % OUTPUT:
        % x : the value at which the obj function is minimised
        % fval : the value of the obj function at x
        % fhist : the history of the objective function at each iteration
        % acchist : the history of the accuracy at at each iteration
    
    % preallocate convergence history
    xhist = zeros(numel(x0),maxiter); 
    fhist = zeros(1,maxiter); 
    acchist = zeros(1,maxiter); 
    
    k = 1; 
    x = x0;
    acc = tol+1; % initialise with some fictional value greater than tol
    
    while k<=maxiter && acc>tol
        
        % store current x
        xhist(:,k) = x;
        
        % compute objective function and gradient at x
        [fun_x, grad_fun_x] = fun(x);
        fhist(k) = fun_x;
        
        % compute accuracy
        acc = norm(grad_fun_x)^2 / mu^2;
        acchist(k) = acc;
        
        % GD update
        x = x - tau * grad_fun_x;
        
        % next iteration
        k = k+1;
        
    end
    
    % final value of the objective function
    fval = fun_x;
    
    % truncate histories
    k = k-1;
    xhist = xhist(:,1:k);
    fhist = fhist(1:k);
    acchist = acchist(1:k);
    
end