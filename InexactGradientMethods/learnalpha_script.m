rng('default');

% create signals
n = 64;
k = 32; % 50 gives interesting case
sigma = 0.03;
[u_star,y,B] = make_undersampled_signal(n,k,sigma);

% model parameters
lambda = 1e-2;
gamma = 1e-2;

% algorithm parameters
alpha0 = 0.0001;
acc_k_fun = @(k) 10^8;
delta0 = [10; 0.1];
tol = 1e-6;
maxiter = 0;
tau0 = 1/25;
alpha_tau = 1e-3;
beta1_tau = 0.7;
beta2_tau = 1.5;
beta1_delta = 0.8;
beta2_delta = 2;


[alpha_vals, L_vals, delta_vals, norm_gradL_2_vals, u_best, lower_iters, linear_iters] = learnalpha(u_star, y, B, lambda, gamma, alpha0, acc_k_fun, delta0, tol, maxiter, tau0, alpha_tau, beta1_tau, beta2_tau, beta1_delta, beta2_delta);

alpha_best = alpha_vals(end);
L_best = L_vals(end);
iters = numel(alpha_vals);

% CREATE FIGURES

% Create plot of optimal reconstruction
figure
hold on
truth = line(1:n, u_star);
data = line(find(~all(B==0)), y);
recon = line(1:n, u_best);
% Adjust line properties (functional)
set(truth, 'Color', [0 0.4470 0.7410])
set(data, 'LineStyle', 'none', 'Marker', '.', 'Color', [.3 .3 .3])
set(recon, 'LineStyle', '--', 'Color', 'r')
% Adjust line properties (aesthetics)
set(truth, 'LineWidth', 1.5)
set(data, 'LineWidth', 0.5, 'Marker', 'o', 'MarkerSize', 4, 'MarkerEdgeColor', [.2 .2 .2], 'MarkerFaceColor' , [.7 .7 .7])
set(recon, 'LineWidth', 1.5)
% % Add labels
hTitle = title(sprintf('$\\alpha = %.4f$',alpha_best),'interpreter','latex');
% Add legend
hLegend = legend([truth, data, recon], ...
    'Truth', 'Data', 'Reconstruction', 'Location', 'NorthEast');
% Adjust font
set(gca, 'FontName', 'CMU Serif')
set(hTitle, 'FontName', 'CMU Serif')
set([hLegend, gca], 'FontSize', 13)
set(hTitle, 'FontSize', 18, 'FontWeight', 'bold')
ylim([-1.1 2.1])
xlim([1 n])
% Adjust axes properties
set(gca, 'Box', 'off', 'TickDir', 'out', 'TickLength', [.02 .02], ...
    'XMinorTick', 'on', 'YMinorTick', 'on', 'YGrid', 'on', ...
    'XColor', [.3 .3 .3], 'YColor', [.3 .3 .3], 'YTick', -2:1:3, ...
    'LineWidth', 1)
filename = 'figs/optimal_recon_3.pdf';
exportgraphics(gcf,filename)

% % Create plot of convergence history
% nexttile
% hold on
% yyaxis left
% alpha_hist = line(0:iters-1, alpha_vals);
% hXLabel = xlabel('Iteration');
% hY1Label = ylabel('Parameter');
% yyaxis right
% L_hist = line(0:iters-1, L_vals);
% hY2Label = ylabel('Objective function');
% % Adjust line properties (functional)
% set(alpha_hist, 'Color', [0 0.4470 0.7410])
% % Adjust line properties (aesthetics)
% set(alpha_hist, 'LineWidth', 1.5)
% set(L_hist, 'LineWidth', 1.5)
% % Adjust font
% set(gca, 'FontName', 'CMU Serif')
% set([hXLabel, hY1Label, hY2Label], 'FontName', 'CMU Serif')
% set([hXLabel, hY1Label, hY2Label], 'FontSize', 16)
% %Adjust axes properties
% xlim([0 iters-1])
% set(gca, 'Box', 'off', 'TickDir', 'out', 'TickLength', [.02 .02], ...
%     'XMinorTick', 'on', 'YMinorTick', 'on', 'YGrid', 'on', ...
%     'XColor', [.3 .3 .3], 'LineWidth', 1)
% % filename = 'figs/error-seqs/convergence_history.pdf';
% % exportgraphics(gcf,filename)
% 
% % Create plot of delta values
% nexttile
% hold on
% yyaxis left
% delta1_hist = line(0:iters-1, delta_vals(:,1));
% hXLabel = xlabel('Iteration');
% hY1Label = ylabel(sprintf('$\\delta_1$'),'interpreter','latex');
% yyaxis right
% delta2_hist = line(0:iters-1, delta_vals(:,2));
% hY2Label = ylabel(sprintf('$\\delta_2$'),'interpreter','latex');
% % Adjust line properties (functional)
% set(delta1_hist, 'Color', [0 0.4470 0.7410])
% % Adjust line properties (aesthetics)
% set(delta1_hist, 'LineWidth', 1.5)
% set(delta2_hist, 'LineWidth', 1.5)
% set(delta1_hist, 'LineStyle', '-')
% set(delta2_hist, 'LineStyle', '--')
% % Adjust font
% set(gca, 'FontName', 'CMU Serif')
% set([hXLabel, hY1Label, hY2Label], 'FontName', 'CMU Serif')
% set([hXLabel, hY1Label, hY2Label], 'FontSize', 16)
% %Adjust axes properties
% xlim([0 iters-1])
% set(gca, 'Box', 'off', 'TickDir', 'out', 'TickLength', [.02 .02], ...
%     'XMinorTick', 'on', 'YMinorTick', 'on', 'YGrid', 'on', ...
%     'XColor', [.3 .3 .3], 'LineWidth', 1)
% % filename = 'figs/delta_history.pdf';
% % exportgraphics(gcf,filename)
% 
% filename = 'figs/tiled.pdf';
% exportgraphics(t,filename)



% EXPERIMENT 3 - ADAPTING DELTA1 and AND DELTA2 DIFFERENTLY

% alpha0 = 1;
% acc_k_fun = @(k) 10^11;
% delta0 = [0.1; 0.1];
% tol = 1e-6;
% maxiter = 5000;
% tau0 = 1/20;
% alpha_tau = 1e-3;
% beta1_tau = 0.7;
% beta2_tau = 1.5;
% beta1_delta = [0.1; 0.9]; 
% beta2_delta = [10; 1.1]; 

