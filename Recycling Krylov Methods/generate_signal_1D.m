function [signals, k_vals] = generate_signal_1D(n, total)

%Function which creates a number of 1D signals with jumps. 
    %Input: 
        % n : The length of your signal.
        % total : The number of signals you want.

    %Outputs:
        % signals: matrix which contain the individual signals in its columns.
        % k_vals: the number of jumps in each signal.

%The jumps are at random locations. The amplitude between jumps is also
%random. 


%Initialising 
signals = zeros(n,total); %Matrix which will contain the output signals in its columns
x = zeros(n, 1); %One output signal
Amp = -2:1:2; %These are our amplitudes. Want to choose randomly from these. 
R = 1:n; %Array of size n with integer values from 1 to n. 
eve = 2:2:10; %Array of even numbers in the the range 1 to 10 (needed to specificy the amount of jumps). (used to be n, but there were too many jumps)

for i = 1:total
    k = randsample(eve,1); %Chooses at random one even integer. This will equal the amount of jumps. 
    sub = sort(randsample(R,k));
    for j =1:(length(sub) -1)
        h = randsample(Amp,1);
        left = sub(j);
        right = sub(j+1);
        x(left: right) = h;
    end
    signals(:,i) = x;
    k_vals(i) = k;
 end

end