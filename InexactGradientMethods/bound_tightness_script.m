rng('default');

%%%% SETUP %%%%

% create signals
n = 32;
k = 32; % 50 gives interesting case
sigma = 0.03;
[u_star,y,B] = make_undersampled_signal(n,k,sigma);

% fixed parameters
alpha = 0.01; % FIXED FOR PURPOSES OF THIS EXPERIMENTS
lambda = 1e-1;
gamma = 1e-1;

% values of delta (delta_1 = delta_2)

delta_vals = 10.^(-1:-1:-8);
num_delta_vals = numel(delta_vals);

% dimension of problem
n = size(u_star,1);

% upper level objective function
L = @(u) (1/2) * norm(u - u_star)^2;

% preallocate arrays to store outputs 
e_bound_vals = zeros(num_delta_vals,1);
e_exact_vals = zeros(num_delta_vals,1);

% compute Lipschitz and strong convexity constants of lower level problem
normB = normest(B);
M = normB^2 + 8*alpha/gamma + lambda; % depends on alpha
mu = normB^2 + lambda;

% compute C_H, M_Phi, M_H
[~,SBB,~] = svd(B'*B);
C_H = 1/(min(diag(SBB))+lambda);
M_Phi = 8/gamma;
M_H = 16/gamma^2;


%%%% EXPERIMENT %%%%

for i = 1:num_delta_vals
    
    delta = [delta_vals(i); delta_vals(i)];
    
    % COMPUTE ESTIMATE OF ERROR

    % solve lower level problem to accuracy delta1, initialised at u0
    E_fun = @(u) E_tv(u, y, B, alpha, lambda, gamma);
    u_tilde = gd_convex(E_fun, zeros(n,1), M, mu, delta(1)^2, 250000); 

    % compute tildePhi, tildeH, tildeb, tildew
    [~, DuR, DuuR] = TV(u_tilde, gamma);
    tildePhi = -DuR';
    tildeH = B'*B + alpha*DuuR + lambda*eye(n);
    tildeb = u_tilde - u_star;
    [tildew, ~] = cgs(tildeH, tildeb, delta(2)/norm(tildeb), 200); % cgs uses relative residual ||Ax-b||/||b|| as stopping criterion

    % compute approximate gradient
    grad_approx = tildePhi * tildew;

    % estimate the error in the gradient
    e_bound = acc_bound(C_H, M_Phi, M_H, delta, tildePhi, tildew);

    % COMPUTE ACTUAL ERROR

    % get accurate lower level solution
    u_hat = gd_convex(E_fun, zeros(n,1), M, mu, 1e-12, 250000);

    % compute Phi, H, b, w (by inverting exactly
    [~, DuR, DuuR] = TV(u_tilde, gamma);
    Phi = -DuR';
    H = B'*B + alpha*DuuR + lambda*eye(n);
    b = u_tilde - u_star;
    w = H\b;

    % compute actual gradient
    grad_exact = Phi * w;

    % compute actual error in gradient
    e_exact = norm(grad_exact - grad_approx);
    
    % STORE RESULTS
    
    e_bound_vals(i) = e_bound;
    e_exact_vals(i) = e_exact;
    
end


%%%% CREATE FIGURE

figure
loglog(delta_vals, e_exact_vals,'LineWidth', 1.5, 'Color', [0 0.4470 0.7410])
hold on
loglog(delta_vals, e_bound_vals,'LineWidth', 1.5, 'Color', 'r')
hold off
% Add labels
hXLabel = xlabel(sprintf('$\\delta=\\delta_1=\\delta_2$'),'interpreter','latex');
hYLabel = ylabel('Norm of gradient error');
hTitle = title(sprintf('$\\alpha = %.2f$', alpha),'interpreter','latex');
% Add legend
hLegend = legend('Exact', 'Estimate/bound', 'Location', 'NorthWest');
% Adjust font
set(gca, 'FontName', 'CMU Serif')
set(hTitle, 'FontName', 'CMU Serif')
set([hLegend, gca], 'FontSize', 16)
set([hXLabel, hYLabel], 'FontName', 'CMU Serif')
set([hXLabel, hYLabel], 'FontSize', 16)
set(hTitle, 'FontSize', 18, 'FontWeight', 'bold')
% Adjust axes properties
set(gca, 'Box', 'off', 'TickDir', 'out', 'TickLength', [.02 .02], ...
    'XMinorTick', 'off', 'YMinorTick', 'off', 'YGrid', 'off', ...
    'XColor', [.3 .3 .3], 'YColor', [.3 .3 .3], ...
    'LineWidth', 1)
filename = 'figs/bound_tightness/bound_tight_alpha1.pdf';
exportgraphics(gcf,filename)






