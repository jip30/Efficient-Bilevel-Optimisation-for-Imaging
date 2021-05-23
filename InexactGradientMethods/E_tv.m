function [val,grad] = E_tv(u,y,B,alpha,lambda,gamma)

% Evaluate val=E(u;alpha) and grad=\nabla E(u;alpha) with TV+Tikhonov.

[R,DuR,~] = TV(u,gamma);

val = 0.5*norm(B*u-y)^2 + alpha*R + 0.5*lambda*norm(u)^2; 
grad = B'*(B*u-y) + alpha*DuR + lambda*u;

end

