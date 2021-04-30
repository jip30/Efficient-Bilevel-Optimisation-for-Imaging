function [x, V, H, Convergence] = gmres_recycle(A, b, x0, V_old, h, Ek, maxiter)
%Function which implements the recycling GMRES algorithm. 

%%Input:
% A,b: the matrix and data vector in Ax =b
% x0: The initial estimate
% V_old: the recycled arnoldi vectors from the previous iteration
% h: the Hessenberg matrix from the previous iteration
% Ek: matrix which determines which vectors to recycle
% maxiter: maximun number of iterations

%%Output:
% x: the estimate for the vector x
% V: the new arnoldi basis
% H: the new Hessenberg matrix
% Convergence: vector containing the convergence information


m = size(x0,1); 
n = maxiter;


%Selecting recycled vectors
Vk = V_old(:, 1:end-1)*Ek; 

Hk = h*Ek; %selected Hessenberg matrix

%Obtain the QR decomposition of Hk. 
[Q,R] = qr(Hk,0);

k = size(Ek, 2);

U = Vk*inv(R);
%U = Vk/R;
C = V_old*Q;
P = C*C';

%Initial guess x0 (can choose this to be the vector x from regular gmres)
r0 = b - A*x0; %initial residual

%Initialise first Arnoldi Vector:
v1 = r0/ norm(r0);

V = zeros(m,1); %Initialising the matrix of basis vectors for the subspace Kj
V(:,1) = v1;
v = v1;
H = zeros(2,1); %Initialising the Hessenberg matrix
Convergence = zeros(n,1);
%Arnoldi Process:
for j = 1:n
      %fprintf('j =  %d \n ', j)
      %Computing the Arnoldi vectors (constructing our orthonormal basis)
      %w = ((eye(m) - P)*A)*v; 
      %Computationally Cheaper than before
      F = A*v;
      G = C'*F;
      J = C*G;
      w = F - J;
          
      for i = 1:j
          H(i,j) = dot(w,V(:,i)); %Component of Hessenberg Matrix
          w = w - H(i,j)*V(:,i);
      end   
      H(j+1,j) = norm(w);
      if H(j+1,j) < 10e-10
          n = j;
          fprintf('Breaking Recycling')
          break; 
      else
        v = w/ H(j+1,j);      
        V(:,j+1) = v; 
      end
      
      B = C'*A*V(:, 1: end-1);

      %Solve the system for y and z. 
      e1 = zeros(j+1, 1);
      e1(1) = 1;
      beta = norm((eye(m) - P)*r0); 
      g = beta*e1;
      
      %Edit initialise to be full size, choose what to invert

      y = H\g; %Solving for y
      z = C'*r0 - B*y; %Solving for z

      %Computing x
      s = U*z;
      t = V(:, 1: end-1)*y;
      x = x0 + s + t; 
      
      %Convergence Information
      r = norm(b - A*x);
      Convergence(j) = r;
end
end 