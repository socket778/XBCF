##########################################
## XBCF-GP comparison demo
##########################################
# setwd('~/Dropbox (ASU)/xbart_gp/causal/')
# setwd('~/coverage/demo/')
n <- 500
simnum<-100

# exp set up --------------------------------------------------------------
## run simulation ##
# setwd(wd)
library(dbarts)
library(XBART)
library(XBCF)
# source('functions.R')
set.seed(simnum)


# DGP --------------------------------------------------------------------

# a 1-dim dgp that has non-overlap area
n = 1000
x = seq(-10, 10, length.out=n)
mu = sin(x)
tau = 0.25*x
pi = 0.08*x + 0.5
pi[pi > 1] = 1
pi[pi < 0] = 0
# plot(x,pi)
# pi = rep(0.5, n)
# pi[(x<4.5)&(x>-4.5)] = 0
# pi[(x>8)|(x< -8)]=0


z = rbinom(n, 1, pi)
f = mu + tau*z
y = f + 0.2*sd(f)*rnorm(n)

# overlap area
v1 = max(x[which(pi==0)])
v2 = min(x[which(pi==1)])


# fitz = XBART.multinomial(y = z, num_class = 2, X = x, p_cateogrical = 0)
# predz = predict(fitz, as.matrix(x))
# ps = predz$prob

sink("/dev/null")
fitz = nnet::nnet(z ~ .,data = cbind(z, x), size = 3,rang = 0.1, maxit = 1000, abstol = 1.0e-8, decay = 5e-2)
sink() # close the stream
ps = fitz$fitted.values

#######################
## 1. XBCF ##
#######################
xbcf.fit = XBCF(y, z, as.matrix(x), as.matrix(x), n_trees_mod = 20, num_sweeps=100,
                pihat = ps, pcat_con = 0,  pcat_mod = 0, Nmin = 20)

ce_xbcf = list()
xbcf.tau = xbcf.fit$tauhats.adjusted
ce_xbcf$ite = rowMeans(xbcf.tau)
ce_xbcf$itu = apply(xbcf.tau, 1, quantile, 0.975, na.rm = TRUE)
ce_xbcf$itl = apply(xbcf.tau, 1, quantile, 0.025, na.rm = TRUE)

#######################
## 1. XBCF-GP ##
#######################
tau_gp = mean(xbcf.fit$sigma1_draws)^2/ (xbcf.fit$model_params$num_trees_trt)
xbcf.gp = predictGP(xbcf.fit, y, z, as.matrix(x),as.matrix(x), as.matrix(x), as.matrix(x),
                    pihat_tr = ps, pihat_te = ps, theta = 0.1, tau = tau_gp, verbose = FALSE)
ce_xbcf_gp = list()
xbcf.gp.tau <- xbcf.gp$tau.adjusted
ce_xbcf_gp$ite = rowMeans(xbcf.gp.tau)
ce_xbcf_gp$itu = apply(xbcf.gp.tau, 1, quantile, 0.975, na.rm = TRUE)
ce_xbcf_gp$itl = apply(xbcf.gp.tau, 1, quantile, 0.025, na.rm = TRUE)

# Demo plot ---------------------------------------------------------------
cex_size = 1.2
lab_size = 2
tick_size = 2
line_size = 1.5

# xbcf
plot(x, tau, col = z + 1, ylim = range(tau, ce_xbcf$itu, ce_xbcf$itl),
     ylab = 'Treatment Effect', xlab = '', cex.lab = lab_size, cex.axis = tick_size, cex = line_size, pch = 20)
abline(v = v1, col = 1, lty = 3, cex = 2)
abline(v = v2, col = 1, lty = 3, cex = 2)
lines(x,ce_xbcf$ite, col = 4, lwd=2, cex = line_size)
lines(x,ce_xbcf$itu, col = 3, lwd=2, cex = line_size)
lines(x,ce_xbcf$itl, col = 3, lwd=2, cex = line_size)
legend('topleft', legend = c('Treated', 'Control', 'XBCF', '95% CI'), col = c(2, 1, 4, 3),
       lty = c(NA, NA, 1, 1), pch = c(20, 20, NA, NA), cex = cex_size, font_size)

# xbcf_gp
plot(x, tau, col = z + 1, ylim = range(tau, ce_xbcf_gp$ite, ce_xbcf_gp$itl), 
     ylab = 'Treatment Effect', xlab = '' , cex.lab = lab_size, cex.axis = tick_size, cex = line_size, pch = 20)
abline(v = v1, col = 1, lty = 3, cex = 2)
abline(v = v2, col = 1, lty = 3, cex = 2)
lines(x,ce_xbcf_gp$ite , col = 4, lwd=2, cex = line_size)
lines(x,ce_xbcf_gp$itu, col = 3, lwd=2, cex = line_size)
lines(x,ce_xbcf_gp$itl, col = 3, lwd=2, cex = line_size)
legend('topleft', legend = c('Treated', 'Control', 'XBCF-GP', '95% CI'), col = c(2, 1, 4, 3),
       lty = c(NA, NA, 1, 1), pch = c(20, 20, NA, NA), cex = cex_size)


