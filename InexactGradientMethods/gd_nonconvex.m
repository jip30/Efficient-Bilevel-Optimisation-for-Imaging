function [x,xhist,fhist,acchist] = gd_nonconvex(fun,x0,beta,tol,maxiter)

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
    acc = norm(grad_fun_x)^2
    acchist(k) = acc;

    % adaptive step size by backtracking line search
    % see Armijo's condition (Backtracking GD) in 
    % https://www.mn.uio.no/math/english/research/groups/statistics-data-science/events/conferences/halden-workshop-2019/presentations/backtrackinggdveryshortversion.pdf
    tau = 1;
    while fun(x - tau * grad_fun_x) - fun_x > -(tau/2) * acc
        tau = beta * tau;
    end
    x = x - tau * grad_fun_x
    
    k = k+1

end

% truncate histories
k = k-1;
xhist = xhist(:,1:k);
fhist = fhist(1:k);
acchist = acchist(1:k);

end
