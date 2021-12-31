function cost = gplvm_objective(z,Y,Xdim,Ydim,W,betalen,kernel_Y)
% 

% unpack z
X = reshape(z(1:end-betalen,1),Xdim);
beta = z(end-betalen+1:end);

%
time_steps = Xdim(1);
obs_dim = Ydim(2);
K_Y = zeros(time_steps);

for ii = 1:time_steps
    for jj = 1:time_steps
        K_Y(ii,jj) = kernel_Y(X(ii,:),X(jj,:),beta(1),beta(2),beta(3));
    end
end

% K_Y = K_Y + beta(3)*eye(time_steps);
det_term = obs_dim/2*logdet(K_Y);
trace_term = 1/2*trace(K_Y\Y*Y');
log_likelihood = det_term + trace_term + sum(log(beta)); 

cost = (log_likelihood);

end 