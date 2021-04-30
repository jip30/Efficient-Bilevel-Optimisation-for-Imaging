function [x, fval, xhist, fhist, acchist,t,gradhist] = gd_non_convex_basic(fun1,fun2, x0, beta, maxiter, tol)

    % Gradient descent with adaptive step size (backtracking line
    % search)for the standard GMRES algorithm
    % INPUT:
        % fun1 : the objective function 
        % fun2 : the objective function, which only evaluates L and not its
        % gradient
        % x0 : the initial value
        % tau : the step size
        % beta : a constant in (0,1), but typically in (0.1,0.8)
        % maxiter : maximum number of iterations 
        % tol : >0, required accuracy - stop when ||grad_fun(x)||^2 < tol 
        
    % OUTPUT:
        % x : the value at which the obj function is minimised
        % fval : the value of the obj function at x
        % fhist : the history of the objective function at each iteration
        % acchist : the history of the accuracy at at each iteration
        % t: vector containing the time taken for each iteration
        % gradhist: the history of the gradient of the objective function at each iteration
    
    
    % preallocate convergence history
    xhist = zeros(numel(x0),maxiter); 
    fhist = zeros(1,maxiter); 
    gradhist = zeros(1,maxiter); 
    acchist = zeros(1,maxiter); 
    
    k = 1; 
    x = x0;
    acc = tol+1; % initialise with some fictional value greater than tol
    t = zeros(maxiter,1); % put int output
    
    
    while k<=maxiter && acc>tol
        
        % store current x
        xhist(:,k) = x;
        
        % compute objective function and gradient at x
        startLoop = tic;
        [fun_x, grad_fun_x] = fun1(x);
        t(k) = toc(startLoop);
        fhist(k) = fun_x;
        gradhist(k) = grad_fun_x;
        
        % compute accuracy
        acc = norm(grad_fun_x)^2;
        acchist(k) = acc;
        
        fprintf('Iteration: %d \n ', k) 
        fprintf('Current Alpha: %f \n ',x)
        fprintf('Current L: %f \n ', fun_x)
       
        
        % adaptive step size by backtracking line search
        % see Armijo's condition (Backtracking GD) in 
        % https://www.mn.uio.no/math/english/research/groups/statistics-data-science/events/conferences/halden-workshop-2019/presentations/backtrackinggdveryshortversion.pdf
        tau = 0.1;
        count_iter = 0; %max 10 (or 5) iterations
        while fun2(x - tau * grad_fun_x) - fun_x > -(tau/2) * acc && count_iter <10
            tau = beta * tau; 
            count_iter = count_iter +1;
        end

        
        x = x - tau * grad_fun_x;       
        k = k+1;

    end
    
    % final value of the objective function
    fval = fun_x;
    
    % truncate histories
    k = k-1;
    xhist = xhist(:,1:k);
    fhist = fhist(1:k);
    acchist = acchist(1:k);
    gradhist = gradhist(1:k);
    t = t(1:k);
    
end
