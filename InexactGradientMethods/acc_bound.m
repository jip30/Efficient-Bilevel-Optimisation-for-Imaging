function e_tilde = acc_bound(C_H, M_Phi, M_H, delta, tildePhi, tildew)

term1 = C_H * (M_Phi * delta(1) + norm(tildePhi)) * (delta(1) + delta(2) + M_H * delta(1) * norm(tildew));
term2 = M_Phi * delta(1) * norm(tildew);

e_tilde =  term1 + term2; 


end

