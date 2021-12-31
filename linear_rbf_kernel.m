function val = linear_rbf_kernel(x,xp,alpha1,alpha2,alpha3,alpha4)
% linear + rbf kernel 

    val =  alpha1*exp(-alpha2/2*norm(x-xp)^2) + alpha3*x*xp';
    if x == xp
        val = val + alpha4;
    end
    
end 