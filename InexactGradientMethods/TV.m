function [R,DuR,DuuR] = TV(u,gamma)

% make finite difference operator matrix
n = size(u,1);
e = ones(n,1);
A = spdiags([-e e],0:1,n,n);
A(n,n)=0;

Au = A * u;
rabsAu = rho(abs(Au),gamma);
[~, drAu,ddrAu] = rho(Au,gamma);

R = sum(rabsAu);
DuR = A' * drAu;
DuuR = A' * diag(ddrAu) * A;

end

