rng('default');

% create signals
n = 64;
k = 32; % 50 gives interesting case
sigma = 0.03;
[u_star,y,B] = make_undersampled_signal(n,k,sigma);

% model parameters
lambda = 1e-3;
gamma = 1e-3;

% algorithm parameters
alpha0 = 1;
acc_k_fun = @(k) max(2*10.^(14 - 0.25.*k), 10.^6);
delta0 = [100; 0.1];
tol = 1e-4;
maxiter = 50;
tau0 = 1/25;
alpha_tau = 1e-3;
beta1_tau = 0.7;
beta2_tau = 1.5;
beta1_delta = 0.8;
beta2_delta = 2;


% learn alpha
[alpha_vals, L_vals, delta_vals, norm_gradL_2_vals, u_best, lower_iters, linear_iters] = learnalpha(u_star, y, B, lambda, gamma, alpha0, acc_k_fun, delta0, tol, maxiter, tau0, alpha_tau, beta1_tau, beta2_tau, beta1_delta, beta2_delta);

alpha_best = alpha_vals(end);
L_best = L_vals(end);
iters = numel(alpha_vals);

% CREATE FIGURES

figure
t = tiledlayout(1, 2);
t.TileSpacing = 'compact';
%t.Padding = 'compact';
tTitle = title(t,'Exponentially decreasing');
set(tTitle, 'FontName', 'CMU Serif')
set(tTitle, 'FontSize', 18, 'FontWeight', 'bold')

% create plot of the error sequences
nexttile
semilogy(0:iters-1, acc_k_fun(0:iters-1).*10^(-12),'LineWidth', 1.5, 'Color', [0 0.4470 0.7410]);
hXLabel = xlabel('Iteration, $k$','Interpreter','latex');
hYLabel = ylabel('$\mathcal{E}_k$','Interpreter','latex');
% Adjust line properties (functional)
% Adjust line properties (aesthetics)
% Adjust font
set(gca, 'FontName', 'CMU Serif')
set([hXLabel, hYLabel], 'FontName', 'CMU Serif')
set([hXLabel, hYLabel], 'FontSize', 16)
%Adjust axes properties
xlim([0 iters-1])
set(gca, 'Box', 'off', 'TickDir', 'out', 'TickLength', [.02 .02], ...
    'XMinorTick', 'on', 'YMinorTick', 'on', 'YGrid', 'on', ...
    'XColor', [.3 .3 .3], 'LineWidth', 1)


% Create plot of convergence history
nexttile
hold on
yyaxis left
alpha_hist = line(0:iters-1, alpha_vals);
hXLabel = xlabel('Iteration');
hY1Label = ylabel('Parameter');
yyaxis right
L_hist = line(0:iters-1, L_vals);
hY2Label = ylabel('Objective function');
% Adjust line properties (functional)
set(alpha_hist, 'Color', [0 0.4470 0.7410])
% Adjust line properties (aesthetics)
set(alpha_hist, 'LineWidth', 1.5)
set(L_hist, 'LineWidth', 1.5)
% Adjust font
set(gca, 'FontName', 'CMU Serif')
set([hXLabel, hY1Label, hY2Label], 'FontName', 'CMU Serif')
set([hXLabel, hY1Label, hY2Label], 'FontSize', 16)
%Adjust axes properties
xlim([0 iters-1])
set(gca, 'Box', 'off', 'TickDir', 'out', 'TickLength', [.02 .02], ...
    'XMinorTick', 'on', 'YMinorTick', 'on', 'YGrid', 'on', ...
    'XColor', [.3 .3 .3], 'LineWidth', 1)

%filename = 'figs/error_sequences/error_seq_exp_decrease.pdf';
%exportgraphics(t,filename)

sprintf('Learnt parameter = %.3f', alpha_best)


%%%

% eperiment 1 - large constant error
% alpha0 = 1;
% acc_k_fun = @(k) 2*10^14;
% delta0 = [100; 0.1];
% tol = 1e-4;
% maxiter = 50;
% tau0 = 1/25;
% alpha_tau = 1e-3;
% beta1_tau = 0.7;
% beta2_tau = 1.5;
% beta1_delta = 0.8;
% beta2_delta = 2;
% semilogy(0:iters-1, 2*10^(2)*ones(iters,1), 'LineWidth', 1.5, 'Color', [0 0.4470 0.7410]);

% eperiment 2 - small constant error
% alpha0 = 1;
% acc_k_fun = @(k) 10^10;
% delta0 = [100; 0.1];
% tol = 1e-4;
% maxiter = 50;
% tau0 = 1/25;
% alpha_tau = 1e-3;
% beta1_tau = 0.7;
% beta2_tau = 1.5;
% beta1_delta = 0.8;
% beta2_delta = 2;
% semilogy(0:iters-1, 10^(-2)*ones(iters,1), 'LineWidth', 1.5, 'Color', [0 0.4470 0.7410]);

% expoentnial decrease
% alpha0 = 1;
% acc_k_fun = @(k) max(2*10.^(14 - 0.25.*k), 10.^6);
% delta0 = [100; 0.1];
% tol = 1e-4;
% maxiter = 250;
% tau0 = 1/25;
% alpha_tau = 1e-3;
% beta1_tau = 0.7;
% beta2_tau = 1.5;
% beta1_delta = 0.8;
% beta2_delta = 2;
% semilogy(0:iters-1, acc_k_fun(0:iters-1).*10^(-12),'LineWidth', 1.5, 'Color', [0 0.4470 0.7410])

% vanishing non-summable
% alpha0 = 1;
% acc_k_fun = @(k) 2*10^14 ./ (k+1);
% delta0 = [100; 0.1];
% tol = 1e-4;
% maxiter = 50;
% tau0 = 1/25;
% alpha_tau = 1e-3;
% beta1_tau = 0.7;
% beta2_tau = 1.5;
% beta1_delta = 0.8;
% beta2_delta = 2;
% semilogy(0:iters-1, acc_k_fun(0:iters-1).*10^(-12),'LineWidth', 1.5, 'Color', [0 0.4470 0.7410])


%%% PLOT OPTIMAL RECONSTRUCTION CODE

% hold on
% truth = line(1:n, u_star);
% data = line(find(~all(B==0)), y);
% recon = line(1:n, u_best);
% % Adjust line properties (functional)
% set(truth, 'Color', [0 0.4470 0.7410])
% set(data, 'LineStyle', 'none', 'Marker', '.', 'Color', [.3 .3 .3])
% set(recon, 'LineStyle', '--', 'Color', 'r')
% % Adjust line properties (aesthetics)
% set(truth, 'LineWidth', 1.5)
% set(data, 'LineWidth', 0.5, 'Marker', 'o', 'MarkerSize', 4, 'MarkerEdgeColor', [.2 .2 .2], 'MarkerFaceColor' , [.7 .7 .7])
% set(recon, 'LineWidth', 1.5)
% % % Add labels
% % %hTitle = title(sprintf('$\\alpha = %.3f$',alpha_best),'interpreter','latex');
% % Add legend
% hLegend = legend([truth, data, recon], ...
%     'Truth', 'Data', 'Reconstruction', 'Location', 'NorthEast');
% % Adjust font
% set(gca, 'FontName', 'CMU Serif')
% %set(hTitle, 'FontName', 'CMU Serif')
% set([hLegend, gca], 'FontSize', 12)
% %set(hTitle, 'FontSize', 18, 'FontWeight', 'bold')
% ylim([-1.1 2.1])
% xlim([1 n])
% % Adjust axes properties
% set(gca, 'Box', 'off', 'TickDir', 'out', 'TickLength', [.02 .02], ...
%     'XMinorTick', 'on', 'YMinorTick', 'on', 'YGrid', 'on', ...
%     'XColor', [.3 .3 .3], 'YColor', [.3 .3 .3], 'YTick', -2:1:3, ...
%     'LineWidth', 1)

lower_iters
linear_iters
