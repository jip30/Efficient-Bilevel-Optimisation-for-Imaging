function [x, V, h, Convergence] = gmres_basic( A, b, x0, maxiter)
%Generalized Minimum Residual Method (GMRES). To solve the system Ax = b for x. 

% Want to return the solution and the basis vectors (v_1 .... v_m)
%Inputs: 
    %A, b, x0 (initial guess),maxiter: maximum number of iterations

%Outputs: 
    % x: the solution
    % V: the matrix containing the Arnoldi basis vectors for the Krylov subspace
    % h: the Hessenberg matrix 
    %Convergence: vector with the norms of the residuals

%The approximation is given by: x = x0 +Vn*y. x is mx1, V is mxn, y is nx1.
%m is the dimension of A, n is the size of the Krylov space (the number of iterations). 
  m = length(A);
  n = maxiter;
  
 
 %Defining the initial residual, beta and v1 (Arnoldi vector)
  r0 = b - A * x0;
  beta = norm(r0);
  v1 = r0/beta; %first arnoldi vectors
  
  V = zeros(m,1); %Initialising the matrix of basis vectors
  V(:,1) = v1;
  
  h = zeros(2,1); %Initialising the Hessenberg matrix H_bar
   
  w = 0;
  v = v1;  %Initialising the first Arnoldi vector
 
  for j = 1:n 
      %Computing the Arnoldi vectors (constructing our orthonormal basis)
      w = A*v;
      for i = 1:j
          h(i,j) = dot(w,V(:,i)); %Component of Hessenberg Matrix
          w = w - h(i,j)*V(:,i);
      end   
      h(j+1,j) = norm(w);
      if h(j+1,j) < 10e-16
          n = j;
          fprintf('Breaking')
          break; 
      else
        v = w/ h(j+1,j);      
        V(:,j+1) = v; 
      end
      
      
      e1 = zeros(j+1, 1);
      e1(1) = 1;
      g = beta*e1;
  
      y = h\g; 
      x= x0 + V(:, 1: end-1)*y;
      
      %Convergence Information
      r = norm(b - A*x);
      Convergence(j) = r;

  end


end

