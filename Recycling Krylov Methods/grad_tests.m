%%%%%%% TEST IMPLEMENTATION OF LOWER LEVEL GRADIENT %%%%%%%%

%rng(123);

% make signal
n = 10;
sigma = 0.02;
k = 0.8 * n;
B = make_B(n, k); 
N = 1;
[U_star, Y] = make_signals(n, sigma, B, N);

% lower level parameters
gamma = 0.002;
epsilon = 0.01;
alpha = 0.01;

% will take a step of size h in direction e_i
h_vals = 10.^(-14:1:1);
i = randi(n);

% vectors to store the results
lim_grad = zeros(numel(h_vals), 1);
fun_grad = zeros(numel(h_vals), 1);
error = zeros(numel(h_vals), 1);

for j = 1:numel(h_vals)
    
    h = h_vals(j);
    d = zeros(n,1);
    d(i) = h;
    
    [val, grad] = E(U_star, Y, B, alpha, gamma, epsilon);
    val_plus = E(U_star + d, Y, B, alpha, gamma, epsilon);
    
    lim_grad(j) = (val_plus - val) / h;
    fun_grad(j) = grad(i);
    error(j) = lim_grad(j) - fun_grad(j);
    
end


% loglog(h_vals, abs(error))
% title('1D lower level gradient testing');
% xlabel('h');
% ylabel('Error');


%%%%%%% TEST IMPLEMENTATION OF UPPER LEVEL GRADIENT %%%%%%%%

% make signal
n = 10;
sigma = 0.02;
k = 0.8 * n;
B = make_B(n, k); 
N = 1;
[U_star, Y] = make_signals(n, sigma, B, N);

% lower level parameters
gamma = 0.01;
epsilon = 0.2;
alpha = 0.5;

% lower level GD
u0 = zeros(n,1);
lower_maxiter = 20000;
lower_tol = 1e-20;


% conjugate gradient
cgsmaxiter = 20;
cgstol = 1e-10;

% will take a step of size h
h_vals = 10.^(-14:1:1);

% vectors to store the results
lim_grad = zeros(numel(h_vals), 1);
fun_grad = zeros(numel(h_vals), 1);
error = zeros(numel(h_vals), 1);

for j = 1:numel(h_vals)
   
    h = h_vals(j);
    
    [val, grad] = L(U_star, Y, B, alpha, gamma, epsilon, lower_maxiter, lower_tol, cgsmaxiter, cgstol);
    val_plus = L(U_star, Y, B, alpha + h, gamma, epsilon, lower_maxiter, lower_tol, cgsmaxiter, cgstol);
    
    lim_grad(j) = (val_plus - val) / h;
    fun_grad(j) = grad;
    error(j) = lim_grad(j) - fun_grad(j);
    
end

figure;
loglog(h_vals, abs(error))
title('1D upper level gradient testing');
xlabel('h');
ylabel('Error');
