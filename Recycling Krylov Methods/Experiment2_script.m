%% Testing the Optimal Alpha
% Code for Experiment 2 - tetsing the optimal alpha on a signal which is
% not in the training data

%Initialising + parameters
n = 256;
sigma = 0.03;
gamma = 0.002; 
epsilon = 0.01;
B = blur1D(n,10,0.2); 
u0 = zeros(n,1);  
lower_maxiter = 1000; 
lower_tol = 1e-10;


%Defining Plotting Colours
colour1 = [0, 0.4470, 0.7410]; %blue
colour2 = [0.4660, 0.6740, 0.1880]; %green
colour3 = [0.9290 0.6940, 0.1250]; %yellow
colour4 = [0.6350, 0.0780, 0.1840]; %burgundy
colour5 = [0.8500, 0.3250, 0.0980]; %orange

%% Training Data

%Creating the training data
rng(103)
[U_star, Y] = make_signals(n, sigma, B, 20);

% Visualing the Ground Truth Training Data
figure;
subplot(5,4,1); plot(U_star(:,1),'-','Color',colour1,'LineWidth', 1); ylim([-2.2 2.2]); xlim([1 n]); title('i=1')
subplot(5,4,2); plot(U_star(:,2),'-','Color',colour1,'LineWidth', 1); ylim([-2.2 2.2]); xlim([1 n]); title('i=2')
subplot(5,4,3); plot(U_star(:,3),'-','Color',colour1,'LineWidth', 1); ylim([-2.2 2.2]); xlim([1 n]); title('i=3')
subplot(5,4,4); plot(U_star(:,4),'-','Color',colour1,'LineWidth', 1); ylim([-2.2 2.2]); xlim([1 n]); title('i=4')
subplot(5,4,5); plot(U_star(:,5),'-','Color',colour1,'LineWidth', 1); ylim([-2.2 2.2]); xlim([1 n]); title('i=5')
subplot(5,4,6); plot(U_star(:,6),'-','Color',colour1,'LineWidth', 1); ylim([-2.2 2.2]); xlim([1 n]); title('i=6')
subplot(5,4,7); plot(U_star(:,7),'-','Color',colour1,'LineWidth', 1); ylim([-2.2 2.2]); xlim([1 n]); title('i=7')
subplot(5,4,8); plot(U_star(:,8),'-','Color',colour1,'LineWidth', 1); ylim([-2.2 2.2]); xlim([1 n]); title('i=8')
subplot(5,4,9); plot(U_star(:,9),'-','Color',colour1,'LineWidth', 1); ylim([-2.2 2.2]); xlim([1 n]); title('i=9')
subplot(5,4,10); plot(U_star(:,10),'-','Color',colour1,'LineWidth', 1); ylim([-2.2 2.2]); xlim([1 n]); title('i=10')
subplot(5,4,11); plot(U_star(:,11),'-','Color',colour1,'LineWidth', 1); ylim([-2.2 2.2]); xlim([1 n]); title('i=11')
subplot(5,4,12); plot(U_star(:,12),'-','Color',colour1,'LineWidth', 1); ylim([-2.2 2.2]); xlim([1 n]); title('i=12')
subplot(5,4,13); plot(U_star(:,13),'-','Color',colour1,'LineWidth', 1); ylim([-2.2 2.2]); xlim([1 n]); title('i=14')
subplot(5,4,14); plot(U_star(:,14),'-','Color',colour1,'LineWidth', 1); ylim([-2.2 2.2]); xlim([1 n]); title('i=14')
subplot(5,4,15); plot(U_star(:,15),'-','Color',colour1,'LineWidth', 1); ylim([-2.2 2.2]); xlim([1 n]); title('i=15')
subplot(5,4,16); plot(U_star(:,16),'-','Color',colour1,'LineWidth', 1); ylim([-2.2 2.2]); xlim([1 n]); title('i=16')
subplot(5,4,17); plot(U_star(:,17),'-','Color',colour1,'LineWidth', 1); ylim([-2.2 2.2]); xlim([1 n]); title('i=17')
subplot(5,4,18); plot(U_star(:,18),'-','Color',colour1,'LineWidth', 1); ylim([-2.2 2.2]); xlim([1 n]); title('i=18')
subplot(5,4,19); plot(U_star(:,19),'-','Color',colour1,'LineWidth', 1); ylim([-2.2 2.2]); xlim([1 n]); title('i=19')
subplot(5,4,20); plot(U_star(:,20),'-','Color',colour1,'LineWidth', 1); ylim([-2.2 2.2]); xlim([1 n]); title('i=20')



%% Reconstruction of a signal not in the training data
rng(25)

%Generating new signal
[U_test_gt, Y_test] = make_signals(n, sigma, B, 1);

%Visualising the new signal 
figure; plot(U_test_gt,'-','Color',colour1,'LineWidth', 1); ylim([-2.2 2.2]); xlim([1 n]); title('New signal not in training data')

%Optimal alpha from N=20
alpha_20 = 0.290147; 

%Reconstruction
U_recon_test20 = solve_lower(Y_test, u0, B, alpha_20, gamma, epsilon, lower_maxiter, lower_tol);

%Visualising
figure;
plot(U_test_gt,'LineWidth', 1)
hold on 
plot(Y_test,'LineWidth', 1)
plot(U_recon_test20,'LineWidth', 1)
title('Reconstruction vs Groundtruth for Optimal Alpha')
hold off
legend('Groundtruth','Data','Reconstruction')
axis tight

