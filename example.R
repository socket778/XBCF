library(XBCF)
# library(nnet)

#### 1. data generation ####

## generate a matrix of covariates
n = 500
x1 = rnorm(n)
x2 = sample(1:2,n,replace=TRUE)
x3 = sample(1:3,n,replace=TRUE,prob = c(0.3,0.4,0.3))
x4 = rnorm(n)
x5 = rnorm(n)
x = cbind(x1,x4,x5,x2,as.factor(x3)) # three continuous vars, one binary var, one categorical ordinal var

## define prognostic function
linear = TRUE # choose if linear
mu = function(x){
  lev = c(2,-1,-4)
  if (linear) {result = 1 + lev[x[,5]] + x[,1]*x[,3]
  } else {result = -6 + lev[x[,5]] + 6*abs(x[,3] - 1)}
  return(result)
}

## generate treatment effects
homogeneous = FALSE # choose if homogeneous
if(homogeneous) {tau = rep(3,n)
} else {tau =  1 + 2*x[,2]*x[,4]}

## define the propensity score function
pi = 0.8*pnorm(3*mu(x)/sd(mu(x))-0.5*x[,1],0,1) + 0.05 + 0.1*runif(n)

## generate treatment assignment scheme
z = rbinom(n,1,pi)

## generate response variable
mu = mu(x)
Ey = mu + tau*z
sig = 0.5*sd(Ey)
y = Ey + sig*rnorm(n)


#### 2. treatment effect estimation ####

## compute propensity score estimates
# fitz = nnet(z ~ .,data = x,size = 3,rang = 0.1, maxit = 1000,abstol = 1.0e-8, decay = 5e-2)
# pihat = fitz$fitted.values

## specify the number of categorical variables in x
p_cat <- 2

## run xbcf
fit_xbcf = XBCF(y, z, x, x, pi, pcat_con = p_cat)

## obtain the treatment effect estimates from the fit
tauhats <- getTaus(fit_xbcf)
muhats <- getMus(fit_xbcf)

## plot the estimates
plot(tau,tauhats)
abline(0,1,col='red')

## compute rmse
rmse <- function(a,b) {
  return(sqrt(mean((a - b)^2)))
}
rmse(tau,tauhats)



## predict function check: predict with the same inputs
#x1 <- cbind(pi,x)
#obj <- predict.XBCF(fit_xbcf, x, x, pi, burnin = 20)
#mus <- rowMeans(obj$mudraws)
#taus <- rowMeans(obj$taudraws)
taus <- predictTaus(fit_xbcf, x)
mus <- predictMus(fit_xbcf, x, pi)

# should be along the line
plot(taus,tauhats)
abline(0,1,col='red')
plot(mus,muhats)
abline(0,1,col='red')
