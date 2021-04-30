function Ek = selection(m, varargin)

%Generates the selection matrix. Creates a matrix of canonical vectors,
%with index at whichever number vector you would like to choose. Eg.
%e1,e5,e9 == varagin =  1,5,9.
%Inputs:
%m: the length of the data
%varagin: what position the 1 will be in in the different vectors

Ek = zeros(m,1);

varg = sort(cell2mat(varargin));


for i =1:nargin-1
    Ek(varg(i),i) = 1; 
end