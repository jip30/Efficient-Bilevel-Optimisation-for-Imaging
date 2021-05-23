function [rho, drho, ddrho] = rho(x,gamma)

rho = (abs(x)<=gamma) .* (- abs(x).^3 / (3*gamma^2) + x.^2/gamma) + (abs(x)>gamma) .* (abs(x) - gamma/3);
drho = (abs(x)<=gamma) .* (-x.^2 .* sign(x) / (gamma^2) + 2 * x/gamma) + (abs(x)>gamma) .* (sign(x));
ddrho = (abs(x)<=gamma) .* ( - (2 * x .* sign(x)) / gamma^2 + (2 / gamma)) + (abs(x)>gamma) .* (0);

end

