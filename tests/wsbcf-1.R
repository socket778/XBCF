# simple demonstration of XBCF with default parameters
library(XBCF)
library(dbarts)

#### 1. DGP
n = 500 # number of observations
# set seed here
# set.seed(1)

# generate covariates
x1 = rnorm(n)
x2 = sample(1:2,n,replace=TRUE)
x3 = sample(1:3,n,replace=TRUE,prob = c(0.3,0.4,0.3))
x4 = rnorm(n)
x5 = rnorm(n)

x = data.frame(x1,x4,x5,x2,as.factor(x3))

# define treatment effects
tau = 1 + 2*x[,2]*x[,4]

## define prognostic function (RIC)
mu = function(x){
  lev = c(2,-1,-4)
  result = -6 + lev[x[,5]] + 6*abs(x[,3] - 1)
  return(result)
}

# compute propensity scores and treatment assignment
pi = 0.8*pnorm(3*mu(x)/sd(mu(x))-0.5*x[,1],0,1) + 0.05 + 0.1*runif(n)
#hist(pi,100)
z = rbinom(n,1,pi)

# generate outcome variable
mu_true = mu(x)
Ey = mu(x) + tau * z
sig = 0.5 * sd(Ey)
y = Ey + sig * rnorm(n)

# If you didn't know pi, you would estimate it here
pihat = pi

# matrix prep
x <- makeModelMatrixFromDataFrame(data.frame(x))

# add pihat to the prognostic term matrix
x1 <- cbind(pihat,x)


#### 2. XBCF
xbcf.fit <- XBCF(y, x1, x, z, p_categorical_pr = 4,  p_categorical_trt = 4)
xbcf.tauhats <- getTaus(xbcf.fit) # get treatment individual-level estimates

# main model parameters can be retrieved below
#print(xbcf.fit$model_params)

# compare results to true values
plot(tau, xbcf.tauhats); abline(0,1)
xbcf.rmse <- sqrt(mean((xbcf.tauhats - tau)^2))

# compute coverage CATE
xbcf.tau <- xbcf.fit$tauhats.adjusted
lbs <- as.numeric(apply(xbcf.tau,1,quantile,0.025))
ubs <- as.numeric(apply(xbcf.tau,1,quantile,0.975))
xbcf.cate.cover <- mean(ubs > tau & lbs < tau)

# print
print(paste0("xbcf RMSE: ", xbcf.rmse))
print(paste0("xbcf CATE coverage: ", xbcf.cate.cover))


#### 3. warmstart-BCF
fit <- wsbcf(y, x, x, z, xbcf_fit = xbcf.fit, pihat = pihat, pcat_con = 4, pcat_mod = 4)
ws.tauhats <- getTaus(fit)

# compare results to true values
plot(tau, ws.tauhats); abline(0,1)
ws.rmse <- sqrt(mean((ws.tauhats - tau)^2))

# compute coverage CATE
ws.tau <- fit$tau_draws
lbs <- as.numeric(apply(ws.tau,1,quantile,0.025))
ubs <- as.numeric(apply(ws.tau,1,quantile,0.975))
ws.cate.cover <- mean(ubs > tau & lbs < tau)

# print
print(paste0("wsbcf RMSE: ", ws.rmse))
print(paste0("wsbcf CATE coverage: ", ws.cate.cover))
