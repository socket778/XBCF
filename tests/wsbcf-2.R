library(dbarts)
library(XBCF)


########### DATA GENERATION ###########
n = 5000
p = 50
k = 5

f = matrix(rnorm(k*n),k,n)
A = matrix(rnorm(k*p),p,k)

x = A%*%f

x = scale(t(x))

p_con = p/2   # half of features is continuous
p_cat = p/2   # half of features is categorical

thresh = rnorm(p_cat)

x[,1:p_cat+p_con] = sapply(1:p_cat+p_con,function(j) x[,j]>thresh[j-p_con])

betas = rep(0,p)
betas[sample(p,10)] <- rnorm(10)

alphas = rep(0,p)
alphas[sample(p,10)] <- rnorm(10)

tau = function(x){
    s = x%*%alphas/sqrt(10)
    return(as.numeric(cut(s,10,labels=FALSE)))
}

mu = function(x){(x%*%betas/sqrt(10))^2}

pis = 0.05 + 0.9*(2*pnorm(mu(x)) - 1)
hist(pis,50)

z = rbinom(n,1,pis)

h = 0.1*diff(range(mu(x)))/diff(range(tau(x)))
kappa = 0.6*sd(mu(x))
y = mu(x) + h*tau(x)*z + kappa*rnorm(n)


# store true parameters
tau.true <- h*tau(x)
ate.true <- mean(tau.true)

# preprocess the matrix
data.mod = makeModelMatrixFromDataFrame(as.data.frame(x))

# number of categorical features
pcat = ncol(data.mod) - 25


########### WARM-START BCF CATE ESTIMATION ###########
fit <- wsbcf(y, data.mod, data.mod, z, pcat_con = pcat, pcat_mod = pcat)
ws.tauhats <- getTaus(fit)

# RMSE
ws.rmse <- sqrt(mean((ws.tauhats - tau.true)^2))

# CATE coverage
ws.tau <- fit$tau_draws
lbs <- as.numeric(apply(ws.tau,1,quantile,0.025))
ubs <- as.numeric(apply(ws.tau,1,quantile,0.975))
ws.cate.cover <- mean(ubs > tau.true & lbs < tau.true)

# print
print(paste0("wsbcf RMSE: ", ws.rmse))
print(paste0("wsbcf CATE coverage: ", ws.cate.cover))

# plot
plot(tau.true,ws.tauhats)
abline(0,1)
