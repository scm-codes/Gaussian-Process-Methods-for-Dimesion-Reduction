function val = rbf_kernel(x,xp,beta1,beta2,beta3)
% rbf kernel 

    val =  beta1*exp(-beta2/2*norm(x-xp)^2);
    if x == xp
        val = val + beta3;
    end
    
end 