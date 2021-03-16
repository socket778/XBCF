# simple demonstration of XBCF with default parameters
library(wsBCF)
library(dbarts)

#### 1. DATA GENERATION PROCESS
n = 500 # number of observations
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
x1 <- cbind(pihat,x)


#### 2. XBCF

# run XBCF
xbcf.fit = XBCF(y, x1, x, z, p_categorical_pr = 5,  p_categorical_trt = 5)

# get treatment individual-level estimates
xbcf.tauhats <- getTaus(xbcf.fit)

# main model parameters can be retrieved below
#print(xbcf.fit$model_params)

# compare results to inference
plot(tau, xbcf.tauhats); abline(0,1)
print(paste0("xbcf RMSE: ", sqrt(mean((xbcf.tauhats - tau)^2))))

#### 2. warmstart-BCF

fit <- wsbcf(y, x, x, z, xbcf_fit = xbcf.fit, pihat = pihat, pcat_con = 5, pcat_mod = 5)

ws.tauhats <- getPointEstimates(fit)

# compare results to inference
plot(tau, ws.tauhats); abline(0,1)
print(paste0("wsbcf RMSE: ", sqrt(mean((ws.tauhats - tau)^2))))
