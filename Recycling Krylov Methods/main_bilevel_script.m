%% Script which conducts the experimentation for the Recycling Krylov Methods strand
% Run this to run te entire bilevel procedure

format long
rng(103)

%Defining Plotting Colours
colour1 = [0, 0.4470, 0.7410]; %blue
colour2 = [0.4660, 0.6740, 0.1880]; %green
colour3 = [0.9290 0.6940, 0.1250]; %yellow
colour4 = [0.6350, 0.0780, 0.1840]; %burgundy
colour5 = [0.8500, 0.3250, 0.0980]; %orange
%% Defining the parameters

% Parameters for the training data
n = 256;        %dimension of the problem
sigma = 0.03;   %the standard deviation for added Gaussian noise
N = 1;          %Number of training data. Change this to repeat the experimentation in the report 


% Lower level functional parameters
gamma = 0.002; %Huber Loss function parameter
epsilon = 0.01; %Convex penalty coefficient

% Lower level GD parameters
lower_maxiter = 1000; 
lower_tol = 1e-10; 

% upper level GD
alpha0 = 1; %Initial guess for alpha
upper_maxiter = 10000;
upper_tol = 1e-9; 
beta = 0.1; %backtracking parameter

% GMRES variables
gmres_iter = 20;
k = 3;
gmresmaxiter = gmres_iter - k;


%Defining the forward operator and initial guess for GD
B = blur1D(n,10,0.2); 
u0 = zeros(n,1); 


%% Making the Training Data
[U_star, Y] = make_signals(n, sigma, B, N);

%% Optimisation with GMRES Basic
start_gmres = tic; 
[alpha_hat_g, L_hat_g, alphahist_g, Lhist_g, acchist_g,t_g,gradhist_g] = solve_upper_basic(U_star, Y, B, alpha0, gamma, epsilon, lower_maxiter, lower_tol, gmres_iter, beta, upper_maxiter, upper_tol);
total_gmres = toc(start_gmres); 

fprintf('\nMethod: GMRES \n')
fprintf('Optimal Alpha: %f \n ',alpha_hat_g)
fprintf('Total time: %f \n\n ',total_gmres)

%% Optimisation with GMRES Recycle
start_recycle = tic; 
[alpha_hat_r, L_hat_r, alphahist_r, Lhist_r, acchist_r,V_recycle_r,H_recycle_r,t_r,gradhist_r] = solve_upper_recycle(U_star, Y, B, alpha0, gamma, epsilon, lower_maxiter, lower_tol, gmresmaxiter, beta, upper_maxiter, upper_tol,k);
total_recycle = toc(start_recycle); 

fprintf('\nMethod: Recycled GMRES \n')
fprintf('Optimal Alpha: %f \n ',alpha_hat_r)
fprintf('Total time: %f \n\n ',total_recycle)

kg = size(t_g,1);
kr = size(t_r,1);

fprintf('\nGMRES time: %f \n',total_gmres)
fprintf('Recycling time: %f \n',total_recycle)
fprintf('GMRES Iterations: %d \n', kg)
fprintf('Recycling Iterations: %d \n', kr)
fprintf('GMRES Alpha: %f \n',alpha_hat_g)
fprintf('Recycling Alpha: %f \n',alpha_hat_r)
fprintf('GMRES L: %f \n',Lhist_g(end))
fprintf('Recycling L: %f \n',Lhist_r(end))

%% Displaying the Results

% Graph showing the time per iteration for each method
figure;
plot(t_g);
xlabel('Iteration')
ylabel('Time (s)')
title('Number of Iterations vs Time Taken for N=', N)
hold on
plot(t_r)
legend('Standard','Recycling')

% Plot the values of alpha and L over each iteration
figure;
title('Convergence history for N=',N)
xlabel('Iteration')
yyaxis left
plot(Lhist_g,'Color',colour1,'LineWidth', 1);
hold on
plot(Lhist_r,'-','Color',colour2,'LineWidth', 1)
ylabel('Objective function, L')
yyaxis right
plot(alphahist_g,'-','Color',colour3,'LineWidth', 1);
plot(alphahist_r,'-','Color',colour4,'LineWidth', 1);
ylabel('Alpha')
legend('L: Standard','L: Recycling', 'Alpha: Standard', 'Alpha: Recycling')


% Recycled GMRES Reconstruction

U_recon_r = zeros(n,N);
for i = 1:N
    U_recon_r(:,i) = solve_lower(Y(:,i), u0, B, alpha_hat_r, gamma, epsilon, lower_maxiter, lower_tol);
end
figure;
plot(U_star(:,1),'LineWidth', 1)
hold on 
plot(Y(:,1),'LineWidth', 1)
plot(U_recon_r(:,1),'LineWidth', 1)
title('Reconstruction vs Groundtruth')
hold off
legend('Groundtruth','Data','Reconstruction')
axis tight
