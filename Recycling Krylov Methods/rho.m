function [rho, deriv_rho, second_deriv_rho] = rho(x, gamma)
    % Computes the values of the Huber loss function, and its first and
    % second derivative. 
    % INPUT:
        % x : an array of values
        % gamma : positive Huber parameter
    % OUTPUT:
        % rho : the function rho evaluated at the points x
        % deriv_rho : the first derivative of rho evaluated at the points x
        % second_deriv_rho: the second derivative of rho evaluated at the points x

    rho = (abs(x)<=gamma) .* (- abs(x).^3 / (3*gamma^2) + x.^2/gamma) + (abs(x)>gamma) .* (abs(x) - gamma/3);
    deriv_rho = (abs(x)<=gamma) .* (-x.^2 .* sign(x) / (gamma^2) + 2 * x/gamma) + (abs(x)>gamma) .* (sign(x));
    second_deriv_rho = (abs(x)<=gamma) .* ( - (2 * x .* sign(x)) / gamma^2 + (2 / gamma)) + (abs(x)>gamma) .* (0);
    
end

