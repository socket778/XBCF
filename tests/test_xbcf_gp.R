library(XBCF)

n = 500
nt = 500
A = rbinom((n+nt), 1, 0.5)
# small non-overlap
mu1 = 1; mu2 = 2; p = 0.5
# substantial non-overlap
mu1 = 1; m3 = 3; p = 0.6

x1 = rnorm(n + nt, mean = A*mu1 + (1-A)*0, 1)
x2 = rnorm(n + nt, mean = A*mu2 + (1-A)*2, 1)
x3 = rbinom(n + nt, 1, p = A*p + (1-A)*0.4)
x = cbind(x1, x2, x3)
# model1 
y1 = rnorm(n + nt, 1 - 2*x1 + x2 - 1.2*x3 + 2*1, 1)
y0 = rnorm(n + nt, 1 - 2*x1 + x2 - 1.2*x3 + 2*0, 1)
y = A*y1 + (1-A)*y0
# model2
# t = 1; c = 0
# y1 = rnorm(n + nt, -3-2.5*x1 + 2*x1^2*t + exp(1.4-x2*t) + x2*x3 - 1.2*x3 - 2*x3*t + 2*t, 1)
# y0 = rnorm(n + nt, -3-2.5*x1 + 2*x1^2*c + exp(1.4-x2*c) + x2*x3 - 1.2*x3 - 2*x3*c + 2*c, 1)
# y = A*y1 + (1-A)*y0
# propensity score?
# pihat = NULL
sink("/dev/null")
fitz = nnet::nnet(A ~.,data = x, size = 3,rang = 0.1, maxit = 1000, abstol = 1.0e-8, decay = 5e-2)
sink() # close the stream
pihat = fitz$fitted.values

ytrain = y[1:n]; ytest = y[(n+1):(n+nt)]
ztrain = A[1:n]; ztest = A[(n+1):(n+nt)]
pihat_tr = pihat[1:n]; pihat_te = pihat[(n+1):(n+nt)]
xtrain = x[1:n,]; xtest = x[(n+1):(n+nt),]



# run XBCF
t1 = proc.time()
xbcf.fit = XBCF(as.matrix(ytrain), as.matrix(ztrain), xtrain, xtrain, pihat = pihat_tr, pcat_con = 1,  pcat_mod = 1)
pred.gp = predictGP(xbcf.fit, xtest, xtest, xtrain, as.matrix(ytrain), as.matrix(ztrain), 
                    tau = 1, pihat = pihat_te, verbose = FALSE)
# pred = predict.XBCF(xbcf.fit, xt, xt, pihat = pihat)
t1 = proc.time() - t1

pred = predict.XBCF(xbcf.fit, xtest, xtest, pihat = pihat_te)
tauhats.pred <- rowMeans(pred$taudraws)
tauhats.gp <- rowMeans(pred.gp$taudraws)

# true tau?
tau = y1[(n+1):(n+nt)] - y0[(n+1):(n+nt)]
cat('True ATE:, ', round(mean(tau), 3), ', GP tau: ', round(mean(tauhats.gp), 3), 
    ', XBCF tau: ', round(mean(tauhats.pred), 3))
