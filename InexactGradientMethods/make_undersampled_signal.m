function [u_star,y,B] = make_undersampled_signal(n,k,sigma)

% Creates a 1D signal of length n, and its undersampled noisy counterpart. 
% u_star is the ground truth, of length n.
% y=B*u_star+xi where B is kxn undersampling matrix, and xi~N(0,sigma^2).

% make undersampling matrix B
r = sort(randsample(n,k));
B = eye(n);
B = B(r,:);

% make ground truth signal
u_star = zeros(n,1);
jumps = randsample(2:2:6,1); % choose number of jumps, even number between 2 and 6
sub = sort(randsample(1:n,jumps)); % choose position of jumps
for j = 1:(length(sub)-1) % create piecewise constant signal with random amplitudes between jump points
  h = randsample(-2:1:2,1); % random amplitude between -2 and 2
  left = sub(j);
  right = sub(j+1);
  u_star(left:right) = h;
end

% make undersampled noisy signal
y = B*u_star + sigma*randn(k,1);

end

