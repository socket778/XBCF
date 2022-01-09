library(XBCF)

n = 500
nt = 500
x = as.matrix(rnorm(n+nt, 0, 5), n+nt,1)
tau = -cos(x)
A = rbinom(n+nt, 1, 0*(abs(x)>5) + 0.5*(abs(x)<=5))
y1 = cos(x) + A*tau
y0 = cos(x)
y = A*y1 + (1-A)*y0 + rnorm(n+nt, 0, 0.2)

# propensity score?
# pihat = NULL
sink("/dev/null")
fitz = nnet::nnet(A ~.,data = as.matrix(x, n+nt, 1), size = 3,rang = 0.1, maxit = 1000, abstol = 1.0e-8, decay = 5e-2)
sink() # close the stream
pihat = fitz$fitted.values


ytrain = as.matrix(y[1:n]); ytest = as.matrix(y[(n+1):(n+nt)])
ztrain = as.matrix(A[1:n]); ztest = as.matrix(A[(n+1):(n+nt)])
# pihat_tr = pihat[1:n]; pihat_te = pihat[(n+1):(n+nt)]
xtrain = as.matrix(x[1:n,]); xtest = as.matrix(x[(n+1):(n+nt),])



# run XBCF
t1 = proc.time()
xbcf.fit = XBCF(as.matrix(ytrain), as.matrix(ztrain), xtrain, xtrain, 
                pihat = NULL, pcat_con = 0,  pcat_mod = 0,
                num_sweeps = 2, n_trees_mod = 5, burnin = 0)
tau_gp = xbcf.fit$sigma1_draws[xbcf.fit$model_params$num_trees_trt, xbcf.fit$model_params$num_sweeps]^2/xbcf.fit$model_params$num_trees_trt
pred.gp = predictGP(xbcf.fit, as.matrix(ytrain), as.matrix(ztrain), xtrain, xtrain, xtest, xtest, 
                    pihat_tr = NULL, pihat_te = NULL, tau = tau_gp, verbose = FALSE)
# pred = predict.XBCF(xbcf.fit, xt, xt, pihat = pihat)
t1 = proc.time() - t1

pred = predict.XBCF(xbcf.fit, xtest, xtest, pihat = NULL)
tauhats.pred <- rowMeans(pred$taudraws)
tauhats.gp <- rowMeans(pred.gp$taudraws)

# true tau?
tauhat = y1[(n+1):(n+nt)] - y0[(n+1):(n+nt)]
cat('True ATE:, ', round(mean(tauhat), 3), ', GP tau: ', round(mean(tauhats.gp), 3), 
    ', XBCF tau: ', round(mean(tauhats.pred), 3))

gp.upper <- apply(pred.gp$taudraws, 1, quantile, 0.975, na.rm = TRUE)
gp.lower <- apply(pred.gp$taudraws, 1, quantile, 0.025, na.rm = TRUE)


# plot(xtest, ytest, col = ztest+1)
# readline()
plot(xtest, tau[(n+1):(n+nt)], ylim = range(c(tau, tauhats.pred, tauhats.gp)))
points(xtest, tauhats.gp, col = 2)
points(xtest, tauhats.pred, col = 4) # the same ????
# readline()

points(xtest, gp.upper, col = 3)
points(xtest, gp.lower, col = 3)
# readline()

# plot(xtest, gp.upper-gp.lower)
