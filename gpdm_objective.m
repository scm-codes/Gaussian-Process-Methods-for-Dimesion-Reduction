function cost = gpdm_objective(z,Y,Xdim,Ydim,W,alphalen,betalen,...
                                    kernel_X,kernel_Y)
% 

% unpack z
X = reshape(z(1:end-(alphalen+betalen),1),Xdim);
alpha = z(end-(alphalen+betalen)+1: end-betalen);
beta = z(end-betalen+1:end);

%
Xout = X(2:end,:);
time_steps = Xdim(1);
lnt_dim = Xdim(2);
obs_dim = Ydim(2);

K_Y = zeros(time_steps);
K_X = zeros(time_steps-1);

for ii = 1:time_steps
    for jj = 1:time_steps
        K_Y(ii,jj) = kernel_Y(X(ii,:),X(jj,:),beta(1),beta(2),beta(3));
        if ii ~= time_steps && jj ~= time_steps
            K_X(ii,jj) = kernel_X(X(ii,:),X(jj,:),alpha(1),alpha(2),alpha(3),alpha(4));
        end
    end
end

% K_Y = K_Y + beta(3)*eye(time_steps);
det_term = obs_dim/2*logdet(K_Y);
trace_term = 1/2*trace(K_Y\Y*W^2*Y');
log_likelihood = det_term + trace_term; 

% K_X = K_X + alpha(4)*eye(time_steps-1);
det_term = lnt_dim/2*logdet(K_X);
trace_term = 1/2*trace(K_X\Xout*Xout');
log_prior =  det_term + trace_term + sum(log(alpha)) + sum(log(beta));

cost = (log_prior+log_likelihood);

end 