rng('default');

%%%% LEARN ALPHA %%%%

% create signals
n = 64;
k = 20; % 50 gives interesting case
sigma = 0.03;
[u_star,y,B] = make_undersampled_signal(n,k,sigma);

% % fixed parameters of the variational model
% lambda = 1e-3;
% gamma = 1e-3;
% 
% % accuracies
% delta1 = 1e-2;
% delta2 = 1e-2;
% 
% % upper level problem
% alpha0 = 0.5; % initialisation
% beta = 0.3; % backtracking parameter
% tol = 1e-4;
% maxiter = 20; 
% 
% % learn alpha
% [alpha_star,alphahist,Lhist,acchist,u_hat] = learnalpha(u_star,y,B,lambda,gamma,delta1,delta2,alpha0,beta,tol,maxiter);

%%%% CREATE PLOT OF GROUND TRUTH, DATA, AND RECONSTRUCTION %%%%

% % fixed parameters of the variational model
% lambda = 1e-3;
% gamma = 1e-3;
% 
% % upper level problem
% alpha0 = 0.5; % initialisation
% beta = 0.3; % backtracking parameter
% tol = 1e-4;
% maxiter = 3; 
% 
% power_vals = [-1];
% 
% for i = 1:numel(power_vals)
%     for j = 1:numel(power_vals)
%         
%         power1 = power_vals(i);
%         power2 = power_vals(j);
%         
%         power1 = -1;
%         power2 = -4;
%         
%         delta1 = 10^power1;
%         delta2 = 10^power2;
%         
%         % learn alpha
%         [alpha_star,alphahist,Lhist,acchist,u_hat] = learnalpha(u_star,y,B,lambda,gamma,delta1,delta2,alpha0,beta,tol,maxiter);
%         
%         % Create basic plot
%         figure
%         hold on
%         truth = line(1:n, u_star);
%         data = line(find(~all(B==0)), y);
%         recon = line(1:n, u_hat);
%         % Adjust line properties (functional)
%         set(truth, 'Color', [0 0 .5])
%         set(data, 'LineStyle', 'none', 'Marker', '.', 'Color', [.3 .3 .3])
%         set(recon, 'LineStyle', '--', 'Color', 'r')
%         % Adjust line properties (aesthetics)
%         set(truth, 'LineWidth', 1.5)
%         set(data, 'LineWidth', 0.5, 'Marker', 'o', 'MarkerSize', 4, 'MarkerEdgeColor', [.2 .2 .2], 'MarkerFaceColor' , [.7 .7 .7])
%         set(recon, 'LineWidth', 1.5)
%         % Add labels
%         hTitle = title(sprintf('$\\delta_1 = 10^{%.f}$, $\\delta_2 = 10^{%.f}$',power1,power2),'interpreter','latex');
%         hXLabel = xlabel('Pixel');
%         hYLabel = ylabel('Signal intensity');
%         % Add legend
%         hLegend = legend([truth, data, recon], ...
%             'Truth', 'Data', 'Reconstruction', 'Location', 'NorthEast');
%         % Adjust font
%         set(gca, 'FontName', 'CMU Serif')
%         set([hTitle, hXLabel, hYLabel], 'FontName', 'CMU Serif')
%         set([hLegend, gca], 'FontSize', 8)
%         set([hXLabel, hYLabel], 'FontSize', 10)
%         set(hTitle, 'FontSize', 12, 'FontWeight' , 'bold')
%         ylim([-1.1 2.1])
%         xlim([1 n])
%         % Adjust axes properties
%         set(gca, 'Box', 'off', 'TickDir', 'out', 'TickLength', [.02 .02], ...
%             'XMinorTick', 'on', 'YMinorTick', 'on', 'YGrid', 'on', ...
%             'XColor', [.3 .3 .3], 'YColor', [.3 .3 .3], 'YTick', -2:1:3, ...
%             'LineWidth', 1)
%         %saveas(gcf,'myfigure.pdf')
%         filename = sprintf('figs/fixed_delta/recon_1e%.f_1e%.f.pdf',power1,power2);
%         exportgraphics(gcf,filename)
%         
%     end
% end



% % Create basic plot
% figure
% hold on
% truth = line(1:n, u_star);
% data = line(find(~all(B==0)), y);
% recon = line(1:n, u_hat);
% % Adjust line properties (functional)
% set(truth, 'Color', [0 0 .5])
% set(data, 'LineStyle', 'none', 'Marker', '.', 'Color', [.3 .3 .3])
% set(recon, 'LineStyle', '--', 'Color', 'r')
% % Adjust line properties (aesthetics)
% set(truth, 'LineWidth', 1.5)
% set(data, 'LineWidth', 0.5, 'Marker', 'o', 'MarkerSize', 4, 'MarkerEdgeColor', [.2 .2 .2], 'MarkerFaceColor' , [.7 .7 .7])
% set(recon, 'LineWidth', 1.5)
% % Add labels
% hTitle = title('$\delta_1 = 10^{-9}$, $\delta_2 = 10^{-6}$','interpreter','latex');
% hXLabel = xlabel('Pixel');
% hYLabel = ylabel('Signal intensity');
% % Add legend
% hLegend = legend([truth, data, recon], ...
%     'Truth', 'Data', 'Reconstruction', 'Location', 'NorthEast');
% % Adjust font
% set(gca, 'FontName', 'CMU Serif')
% set([hTitle, hXLabel, hYLabel], 'FontName', 'CMU Serif')
% set([hLegend, gca], 'FontSize', 8)
% set([hXLabel, hYLabel], 'FontSize', 10)
% set(hTitle, 'FontSize', 12, 'FontWeight' , 'bold')
% ylim([-1.1 2.1])
% xlim([1 n])
% % Adjust axes properties
% set(gca, 'Box', 'off', 'TickDir', 'out', 'TickLength', [.02 .02], ...
%     'XMinorTick', 'on', 'YMinorTick', 'on', 'YGrid', 'on', ...
%     'XColor', [.3 .3 .3], 'YColor', [.3 .3 .3], 'YTick', -2:1:3, ...
%     'LineWidth', 1)
% %saveas(gcf,'myfigure.pdf')
% exportgraphics(gcf,'myfigure.pdf')


%%%% EXPERIMENTAL VERIFICATION OF THE BOUND

% create signals
n = 64; % small dimension since exact computation involves inversion
k = 20; 
sigma = 0.03;
[u_star,y,B] = make_undersampled_signal(n,k,sigma);

% fixed parameters of the variational model
lambda = 1e-3;
gamma = 1e-3;

% value of alpha at which to test gradient
alpha = 0.01;

% values of delta - will use delta=delta_1=delta_2
delta_vals = 10.^(-9:1:-1);
grad_errors = zeros(size(delta_vals));
e_bounds = zeros(size(delta_vals));

for i = 1:numel(delta_vals)
    
    delta = delta_vals(i);
    [val_exact,grad_exact] = L_tv_alpha_exact(u_star,y,B,alpha,lambda,gamma);
    [val_approx,grad_approx] = L_tv_alpha(u_star,y,B,alpha,lambda,gamma,delta,delta);
    grad_errors(i) = norm(grad_exact - grad_approx);
    
    norm_e = e_tv_bound(u_star,y,B,alpha,lambda,gamma,delta,delta);
    e_bounds(i) = norm_e;
    
end

loglog(delta_vals, grad_errors, 'r', delta_vals, e_bounds , 'b')



