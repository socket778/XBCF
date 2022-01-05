# simple demonstration of XBCF with default parameters
library(XBCF)
library(dbarts)

#### 1. DATA GENERATION PROCESS
n = 5000 # number of observations
# set seed here
# set.seed(1)

# generate dcovariates
x1 = rnorm(n)
x2 = rbinom(n,1,0.2)
x3 = sample(1:3,n,replace=TRUE,prob = c(0.1,0.6,0.3))
x4 = rnorm(n)
x5 = rbinom(n,1,0.7)
x = cbind(x1,x2,x3,x4,x5)

# define treatment effects
tau = 2 + 0.5*x[,4]*(2*x[,5]-1)

## define prognostic function (RIC)
mu = function(x){
  lev = c(-0.5,0.75,0)
  result = 1 + x[,1]*(2*x[,2] - 2*(1-x[,2])) + lev[x3]
  return(result)
}

# compute propensity scores and treatment assignment
pi = pnorm(-0.5 + mu(x) - x[,2] + 0.*x[,4],0,3)
#hist(pi,100)
z = rbinom(n,1,pi)

# generate outcome variable
Ey = mu(x) + tau*z
sig = 0.25*sd(Ey)
y = Ey + sig*rnorm(n)

# If you didn't know pi, you would estimate it here
pihat = pi

# matrix prep
x <- data.frame(x)
x[,3] <- as.factor(x[,3])
x <- makeModelMatrixFromDataFrame(data.frame(x))
x <- cbind(x[,1],x[,6],x[,-c(1,6)])

# add pihat to the prognostic term matrix
# x1 <- cbind(pihat,x)

# trim categorical values
x = x[, 1:2]

#### 2. XBCF

# run XBCF
t1 = proc.time()
xbcf.fit = XBCF(as.matrix(y), as.matrix(z), x, x, pihat = as.matrix(pihat), pcat_con = 0,  pcat_mod = 0)
pred.gp = predictGP(xbcf.fit, x, x, x, as.matrix(y), as.matrix(z), tau = 1, pihat = pihat, verbose = FALSE)
pred = predict.XBCF(xbcf.fit, x, x, pihat = pihat)
t1 = proc.time() - t1

# get treatment individual-level estimates
tauhats <- getTaus(xbcf.fit)
tauhats.pred <- rowMeans(pred$taudraws)
tauhats.gp <- rowMeans(pred.gp$taudraws)

# main model parameters can be retrieved below
#print(xbcf.fit$model_params)

# compare results to inference
plot(tau, tauhats.pred); abline(0,1);
points(tau, tauhats.gp, col = 'red')
print(paste0("xbcf RMSE: ", sqrt(mean((tauhats - tau)^2))))
print(paste0("xbcf runtime: ", round(as.list(t1)$elapsed,2)," seconds"))
