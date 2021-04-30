function A = blur1D(N,band,sigma)
%Operator which applies Gaussian blur to a 1D signal. 

z = [exp(-((0:band-1).^2)/(2*sigma^2)),zeros(1,N-band)];
A = toeplitz(z);
A = sparse(A);
A = (1/sqrt((2*pi*sigma^2)))*A;
