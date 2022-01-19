library(XBCF)

n = 500
nt = 200
x = as.matrix(rnorm(n+nt, 0, 5), n+nt,1)
# tau = -sin(0.3*x)
tau = 0.1*x
A = rbinom(n+nt, 1, 0*(x>5) + 0.5*(abs(x)<=5) + 1*(x< -5))
# A = rbinom(n+nt, 1, 0*(x< -5) + 0.5*(abs(x)<=5) + 1*(x>5))
y1 = cos(0.2*x) + tau
y0 = cos(0.2*x)
Ey = A*y1 + (1-A)*y0
sig = 0.25*sd(Ey)
y = Ey + sig*rnorm(n+nt)

# propensity score?
# pihat = NULL
# make sure pihat_tr is consistent
sink("/dev/null")
fitz = nnet::nnet(A[1:n] ~.,data = as.matrix(x[1:n], n, 1), size = 3,rang = 0.1, maxit = 1000, abstol = 1.0e-8, decay = 5e-2)
sink() # close the stream
pihat_tr = fitz$fitted.values
pihat_te = predict(fitz, as.matrix(x[(n+1):(n+nt)], nt, 1))

ytrain = as.matrix(y[1:n]); ytest = as.matrix(y[(n+1):(n+nt)])
ztrain = as.matrix(A[1:n]); ztest = as.matrix(A[(n+1):(n+nt)])
# pihat_tr = pihat[1:n]; pihat_te = pihat[(n+1):(n+nt)]
xtrain = as.matrix(x[1:n,]); xtest = as.matrix(x[(n+1):(n+nt),])
tautr = tau[1:n]; taute = tau[(n+1):(n+nt)]

# test on train
ytest = ytrain; xtest = xtrain; ztest = ztrain; taute = tautr; pihat_te = pihat_tr
# run XBCF
t1 = proc.time()
burnin = 10; num_sweeps = 100; num_trees_trt = 10; num_trees_pr = 10
# burnin = 20; num_sweeps = 100; num_trees_trt = 10; num_trees_pr = 10
xbcf.fit = XBCF(as.matrix(ytrain), as.matrix(ztrain), xtrain, xtrain, 
                pihat = pihat_tr, pcat_con = 0,  pcat_mod = 0,
                num_sweeps = num_sweeps, n_trees_mod = num_trees_trt, n_trees_con = num_trees_pr, burnin = burnin)
tau_gp = mean(xbcf.fit$sigma1_draws)^2/ (xbcf.fit$model_params$num_trees_trt + xbcf.fit$model_params$num_trees_pr) 
pred.gp = predictGP(xbcf.fit, as.matrix(ytrain), as.matrix(ztrain), xtrain, xtrain, xtest, xtest, 
                    pihat_tr = pihat_tr, pihat_te = pihat_tr, theta = 1, tau = tau_gp, verbose = FALSE)
# pred = predict.XBCF(xbcf.fit, xt, xt, pihat = pihat)
t1 = proc.time() - t1

pred = predict.XBCF(xbcf.fit, xtest, xtest, pihat = NULL)
tauhats.pred <- rowMeans(pred$taudraws)
tauhats.gp <- rowMeans(pred.gp$tau.adjusted)

# true tau?
cat('True ATE:, ', round(mean(taute), 3), ', GP tau: ', round(mean(tauhats.gp), 3), 
    ', XBCF tau: ', round(mean(tauhats.pred), 3), '\n')

gp.upper <- apply(pred.gp$tau.adjusted, 1, quantile, 0.975, na.rm = TRUE)
gp.lower <- apply(pred.gp$tau.adjusted, 1, quantile, 0.025, na.rm = TRUE)
xbcf.upper <- apply(pred$taudraws, 1, quantile, 0.975, na.rm = TRUE)
xbcf.lower <- apply(pred$taudraws, 1, quantile, 0.025, na.rm = TRUE)

# evaluate coverage
cat('Coverage:', '\n')
cat('GP = ', round(mean((gp.upper >= taute) & (gp.lower <= taute)), 3), '\n')
cat('XBCF = ', round(mean((xbcf.upper >= taute) & (xbcf.lower <= taute)), 3), '\n')

par(mfrow=c(1,2))
plot(xtest, y0[1:n], col = 1, cex = 0.5, ylim = range(y))
points(xtest, rowMeans(pred.gp$mu.adjusted) , col = 3, cex = 0.5)
points(xtest, rowMeans(pred$mudraws), col = 4, cex = 0.5)

plot(xtest, y1[1:n], col = 1, cex = 0.5, ylim = range(y))
points(xtest, rowMeans(pred.gp$mu.adjusted) + rowMeans(pred.gp$tau.adjusted), col = 3, cex = 0.5)
points(xtest, rowMeans(pred$mudraws) + rowMeans(pred$taudraws), col = 4, cex = 0.5)
legend('topleft', cex = 0.5, pch = 1, col = c(1, 3, 4), legend = c('y1', 'gp','xbcf'))


par(mfrow=c(1,1))
plot(xtest, y1[1:n] - y0[1:n], col = ztest + 1, cex = 0.5, ylim = range(rowMeans(pred.gp$tau.adjusted), y1- y0))
points(xtest, rowMeans(pred$taudraws), col = 4, cex = 0.5)
points(xtest, rowMeans(pred.gp$tau.adjusted), col = 6, cex = 0.5)
points(xtest, gp.upper, col = 3, cex = 0.5)
points(xtest, gp.lower, col = 3, cex = 0.5)
legend('topleft', cex = 0.5, pch = 1, col = c(1, 4, 6, 5), 
       legend = c('y1 - y0', 'tau.xbcf','tau1 - tau0', 'gp C.I'))


plot(xtest, y[1:n], col = ztest + 1, cex = 0.5)
points(xtest, rowMeans(pred.gp$mu.adjusted) + ztest*rowMeans(pred.gp$tau.adjusted), col = 3, cex = 0.5)
points(xtest, rowMeans(pred.gp$mu.adjusted), col = 4, cex = 0.5)
legend('topright', cex = 0.5, pch = 1, col = c(1, 2, 3, 4), 
       legend = c('y[ztest==0]', 'y[ztest==1]', 'tau + mu', 'mu'))
# par(mfrow=c(1,1))
# plot(xtest, y[1:n], cex = 0.5, col = ztest + 1, ylim = range(rowMeans(pred.gp$mu0), rowMeans(pred.gp$mu1)))
# points(xtest, rowMeans(pred.gp$mu0) , cex = 0.5, col = 3) #+ ztest * rowMeans(pred.gp$tau.old)
# points(xtest, rowMeans(pred.gp$mu1), cex = 0.5, col = 4)
# # points(xtest, rowMeans(pred.gp$mu.adjusted)+ ztest*rowMeans(pred.gp$tau.old), cex = 0.5, col = 3)
# legend('topright', cex = 0.5, pch = 1, col = c(1, 2, 3, 4), legend = c('y[ztest==0]', 'y[ztest==1]', 'mu0', 'mu1' ))
# 
# tau.adjust = rowMeans(pred.gp$tau.adjusted) + 
#   ztest * (rowMeans(pred.gp$mu.adjusted - pred.gp$mu1)) + 
#   (1-ztest) * rowMeans(pred.gp$mu0 - pred.gp$mu.adjusted)
# 
# plot(xtest, y1[1:n] - y0[1:n], col = ztest + 1)
# points(xtest, tau.adjust, col = 3)
# points(xtest, rowMeans(pred.gp$tau.old), col = 4)
# 
# # plot mu0 against y - tau_fit on control
# # plot mu1 against y- tau_fit on treated
# # see how they are affected by the poorly estimated mean and how the wrong info accumulate
# plot(xtest, rowMeans(pred.gp$mu0), col = 3)
# points(xtest, y0[1:n],col = 1+ztest )
# points(xtest, rowMeans(pred.gp$mu1), col = 4)
# points(xtest, rowMeans(pred.gp$mu.adjusted), col =5)
# # 
# 
# # par(mfrow=c(1,2))
# plot(xtest, rowMeans(pred.gp$mu1) + rowMeans(pred.gp$tau.old), col = 3, ylim = range(y))
# points(xtest, y1[1:n] , col = 1 + ztest)
# #
# plot(xtest, y[1:n] - ztest * rowMeans(pred.gp$tau.old) - rowMeans(pred.gp$mu.adjusted)/2, col = 1 + ztest, ylim = range(pred.gp$mu0, pred.gp$mu1))
# points(xtest, rowMeans(pred.gp$mu0) - rowMeans(pred.gp$mu.adjusted), col = 3)
# points(xtest, rowMeans(pred.gp$mu1)- rowMeans(pred.gp$mu.adjusted), col = 4)
# legend
# 
