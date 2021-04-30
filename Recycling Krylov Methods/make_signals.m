function [U_star, Y] = make_signals(n, sigma, B, N)

% Creates the training set of N ground truth/ noisy signals of size n
% INPUT:
    % n : length of signal
    % sigma : standard deviation of the Gaussian noise
    % B : the blurring operator
% OUTPUT:
    % U_star : nxN matrix of ground truth signals(each column is a ground truth signal)
    % Y : nxN matrix of blurred, noisy signals (each column is an observed signal)
    
    U_star = zeros(n,N);
    Y = zeros(n,N);
    
    for j = 1:N
        u_star = generate_signal_1D(n,1);
        y = u_star + sigma * randn(n,1);
        y = B* y; 
        
        U_star(:,j) = u_star;
        Y(:,j) = y;
        
    end
    
end
    
    