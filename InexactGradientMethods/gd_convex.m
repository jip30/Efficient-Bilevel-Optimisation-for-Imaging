function [x, xhist, fhist, k] = gd_convex(fun, x0, L, mu, tol, maxiter)

% GD for L-smooth, mu-convex objective function fun. Initialise at x0.
% Run for maxiter iterations or until ||grad_fun(x)||^2 < tol * mu^2

% preallocate histories
xhist = zeros(numel(x0),maxiter+1); 
fhist = zeros(1,maxiter+1); 

% fixed step size
tau = 2/(L+mu);

% 
k = 1; 
x = x0;
xhist(:,k) = x;
% compute objective function and gradient at x
[fun_x, grad_fun_x] = fun(x);
fhist(k) = fun_x;
% compute accuracy
acc = norm(grad_fun_x)^2 / mu^2;

while k<=maxiter && acc>tol
    
    k = k+1;
    
    % GD update
    x = x - tau * grad_fun_x;

    % store current x
    xhist(:,k) = x;
    
    % compute objective function and gradient at x
    [fun_x, grad_fun_x] = fun(x);
    fhist(k) = fun_x;
    
    % compute accuracy
    acc = norm(grad_fun_x)^2 / mu^2;

end

% truncate histories
%k = k-1;
xhist = xhist(:,1:k);
fhist = fhist(1:k);

end

