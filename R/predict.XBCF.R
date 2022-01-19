#' Get post-burnin draws from trained model
#'
#' @param model A trained XBCF model.
#' @param x_con An input matrix for the prognostic term of size n by p1. Column order matters: continuos features should all bgo before of categorical.
#' @param x_mod An input matrix for the treatment term of size n by p2 (default x_mod = x_con). Column order matters: continuos features should all go before categorical.
#' @param pihat An array of propensity score estimates (default is NULL). In the default case propensity scores are estimated using nnet function.
#' @param burnin The number of burn-in iterations to discard from prediction (the default value is taken from the trained model).
#'
#' @return A list with two matrices. Each matrix corresponds to a set of draws of predicted values; rows are datapoints, columns are iterations.
#' @export
predict.XBCF <- function(model, x_con, x_mod=x_con, pihat=NULL, burnin=NULL) {

    if(!("matrix" %in% class(x_con))) {
        cat("Msg: input x_con is not a matrix, try to convert type.\n")
        x_con = as.matrix(x_con)
    }
    if(!("matrix" %in% class(x_mod))) {
        cat("Msg: input x_mod is not a matrix, try to convert type.\n")
        x_mod = as.matrix(x_mod)
    }

    if(ncol(x_con) != model$input_var_count$x_con) {
        stop(paste0('Check dimensions of input matrices. The model was trained on
        x_con with ', model$input_var_count$x_con,
        ' columns; trying to predict on x_con with ', ncol(x_con),' columns.'))
    }
    if(ncol(x_mod) != model$input_var_count$x_mod) {
        stop(paste0('Check dimensions of input matrices. The model was trained on
        x_con with ', model$input_var_count$x_mod,
        ' columns; trying to predict on x_con with ', ncol(x_mod),' columns.'))
    }

    if(is.null(pihat)) {
        if(is.null(model$fitz)){
            stop('No model to fit pihat')
        }else{
            pihat = as.matrix(stats::predict(model$fitz, x_con))
        }
    }
    if(!("matrix" %in% class(pihat))) {
        cat("Msg: input pihat is not a matrix, try to convert type.\n")
        pihat = as.matrix(pihat)
    }

    if(ncol(pihat) != 1) {
        stop(paste0('Propensity score input must be a 1-column matrix or NULL (default).
        A matrix with ', ncol(pihat), ' columns was provided instead.'))
    }

    x_con <- cbind(pihat, x_con)

    obj1 = .Call(`_XBCF_xbcf_predict`, x_con, model$model_list$tree_pnt_pr)
    obj2 = .Call(`_XBCF_xbcf_predict`, x_mod, model$model_list$tree_pnt_trt)

    sweeps <- ncol(model$tauhats)
    if(is.null(burnin)) {
        burnin <- model$model_params$burnin
    }

    if(burnin >= sweeps) {
        stop(paste0('burnin (',burnin,') cannot exceed or match the total number of sweeps (',sweeps,')'))
    }

    mus <- matrix(NA, nrow(x_con), sweeps - burnin)
    taus <- matrix(NA, nrow(x_mod), sweeps - burnin)
    seq <- (burnin+1):sweeps

    for (i in seq) {
        taus[, i - burnin] = obj2$predicted_values[,i] * model$sdy * (model$b1_draws[nrow(model$b1_draws), i] - model$b0_draws[nrow(model$b0_draws), i])
        mus[, i - burnin] = obj1$predicted_values[,i] * model$sdy * (model$a_draws[nrow(model$a_draws), i]) + model$meany
    }

    obj <- list(mudraws=mus, taudraws=taus)

    return(obj)
}

#' Get post-burnin draws from trained model (treatment term only)
#'
#' @param model A trained XBCF model.
#' @param x_mod An input matrix for the treatment term of size n by p2. Column order matters: continuos features should all go before categorical.
#' @param burnin The number of burn-in iterations to discard from prediction (the default value is taken from the trained model).
#'
#' @return A matrix with a set of draws of predicted treatment effect estimates; rows are datapoints, columns are iterations.
#' @export
predictTauDraws <- function(model, x_mod, burnin = NULL) {

    if(!("matrix" %in% class(x_mod))) {
        cat("Msg: input x_mod is not a matrix -- converting type.\n")
        x_mod = as.matrix(x_mod)
    }

    if(ncol(x_mod) != model$input_var_count$x_mod) {
        stop(paste0('Check dimensions of input matrices. The model was trained on
        x_con with ', model$input_var_count$x_mod,
        ' columns; trying to predict on x_con with ', ncol(x_mod),' columns.'))
    }

    obj = .Call(`_XBCF_xbcf_predict`, x_mod, model$model_list$tree_pnt_trt)

    sweeps <- model$model_params$num_sweeps
    if(is.null(burnin)) {
        burnin <- model$model_params$burnin
    }

    if(burnin >= sweeps){
        stop(paste0('burnin (',burnin,') cannot exceed or match the total number of sweeps (',sweeps,')'))
    }

    tauhat.draws <- matrix(NA, nrow(x_mod), sweeps - burnin)
    seq <- (burnin+1):sweeps

    for (i in seq) {
        tauhat.draws[, i - burnin] = obj$predicted_values[,i] * model$sdy * (model$b1_draws[nrow(model$b1_draws), i] - model$b0_draws[nrow(model$b0_draws), i])
    }

    return(tauhat.draws)
}

#' Get point-estimates of treatment effect
#'
#' @param model A trained XBCF model.
#' @param x_mod An input matrix for the treatment term of size n by p2. Column order matters: continuos features should all go before categorical.
#' @param burnin The number of burn-in iterations to discard from prediction (the default value is taken from the trained model).
#'
#' @return An array with point-estimates of treatment effect per datapoint in the given matrix.
#' @export
predictTaus <- function(model, x_mod, burnin = NULL) {

    if(!("matrix" %in% class(x_mod))) {
        cat("Msg: input x_mod is not a matrix -- converting type.\n")
        x_mod = as.matrix(x_mod)
    }

    if(ncol(x_mod) != model$input_var_count$x_mod) {
        stop(paste0('Check dimensions of input matrices. The model was trained on
        x_con with ', model$input_var_count$x_mod,
        ' columns; trying to predict on x_con with ', ncol(x_mod),' columns.'))
    }

    obj = .Call(`_XBCF_xbcf_predict`, x_mod, model$model_list$tree_pnt_trt)

    sweeps <- model$model_params$num_sweeps
    if(is.null(burnin)) {
        burnin <- model$model_params$burnin
    }

    if(burnin >= sweeps) {
        stop(paste0('burnin (',burnin,') cannot exceed or match the total number of sweeps (',sweeps,')'))
    }

    tauhat.draws <- matrix(NA, nrow(x_mod), sweeps - burnin)
    seq <- (burnin+1):sweeps

    for (i in seq) {
        tauhat.draws[, i - burnin] = obj$predicted_values[,i] * model$sdy * (model$b1_draws[nrow(model$b1_draws), i] - model$b0_draws[nrow(model$b0_draws), i])
    }

    tauhats <- rowMeans(tauhat.draws)

    return(tauhats)
}

#' Get post-burnin draws from trained model (prognostic term only)
#'
#' @param model A trained XBCF model.
#' @param x_con An input matrix for the treatment term of size n by p1. Column order matters: continuos features should all go before categorical.
#' @param pihat An array of propensity score estimates (default is NULL). In the default case propensity scores are estimated using nnet function.
#' @param burnin The number of burn-in iterations to discard from prediction (the default value is taken from the trained model).
#'
#' @return A matrix with a set of draws of predicted prognostic effect estimates; rows are datapoints, columns are iterations.
#' @export
predictMuDraws <- function(model, x_con, pihat=NULL, burnin = NULL) {

    if(!("matrix" %in% class(x_con))) {
        cat("Msg: input x_con is not a matrix -- converting type.\n")
        x_con = as.matrix(x_con)
    }

    if(ncol(x_con) != model$input_var_count$x_con) {
        stop(paste0('Check dimensions of input matrices. The model was trained on
        x_con with ', model$input_var_count$x_con,
        ' columns; trying to predict on x_con with ', ncol(x_con),' columns.'))
    }

    if(is.null(pihat)) {
        sink("/dev/null") # silence output
        fitz = nnet::nnet(z~.,data = x_con, size = 3,rang = 0.1, maxit = 1000, abstol = 1.0e-8, decay = 5e-2)
        sink() # close the stream
        pihat = fitz$fitted.values
    }

    if(!("matrix" %in% class(pihat))) {
        cat("Msg: input pihat is not a matrix -- converting type.\n")
        pihat = as.matrix(pihat)
    }

    if(ncol(pihat) != 1) {
        stop(paste0('Propensity score input must be a 1-column matrix or NULL (default).
        A matrix with ', ncol(pihat), ' columns was provided instead.'))
    }

    x_con <- cbind(pihat, x_con)

    obj = .Call(`_XBCF_xbcf_predict`, x_con, model$model_list$tree_pnt_pr)

    sweeps <- model$model_params$num_sweeps
    if(is.null(burnin)) {
        burnin <- model$model_params$burnin
    }

    if(burnin >= sweeps){
        stop(paste0('burnin (',burnin,') cannot exceed or match the total number of sweeps (',sweeps,')'))
    }

    muhat.draws <- matrix(NA, nrow(x_con), sweeps - burnin)
    seq <- (burnin+1):sweeps

    for (i in seq) {
        muhat.draws[, i - burnin]= obj1$predicted_values[,i] * model$sdy * (model$a_draws[nrow(model$a_draws), i]) + model$meany
    }

    return(muhat.draws)
}

#' Get point-estimates of prognostic effect
#'
#' @param model A trained XBCF model.
#' @param x_con An input matrix for the treatment term of size n by p1. Column order matters: continuos features should all go before categorical.
#' @param pihat An array of propensity score estimates (default is NULL). In the default case propensity scores are estimated using nnet function.
#' @param burnin The number of burn-in iterations to discard from prediction (the default value is taken from the trained model).
#'
#' @return An array with point-estimates of prognostic effect per datapoint in the given matrix.
#' @export
predictMus <- function(model, x_con, pihat = NULL, burnin = NULL) {

    if(!("matrix" %in% class(x_con))) {
        cat("Msg: input x_con is not a matrix -- converting type.\n")
        x_con = as.matrix(x_con)
    }

    if(ncol(x_con) != model$input_var_count$x_con) {
        stop(paste0('Check dimensions of input matrices. The model was trained on
        x_con with ', model$input_var_count$x_con,
        ' columns; trying to predict on x_con with ', ncol(x_con),' columns.'))
    }

    if(is.null(pihat)) {
        sink("/dev/null") # silence output
        fitz = nnet::nnet(z~.,data = x_con, size = 3,rang = 0.1, maxit = 1000, abstol = 1.0e-8, decay = 5e-2)
        sink() # close the stream
        pihat = fitz$fitted.values
    }

    if(!("matrix" %in% class(pihat))) {
        cat("Msg: input pihat is not a matrix -- converting type.\n")
        pihat = as.matrix(pihat)
    }

    if(ncol(pihat) != 1) {
        stop(paste0('Propensity score input must be a 1-column matrix or NULL (default).
        A matrix with ', ncol(pihat), ' columns was provided instead.'))
    }

    x_con <- cbind(pihat, x_con)

    obj = .Call(`_XBCF_xbcf_predict`, x_con, model$model_list$tree_pnt_pr)

    sweeps <- model$model_params$num_sweeps
    if(is.null(burnin)) {
        burnin <- model$model_params$burnin
    }

    if(burnin >= sweeps) {
        stop(paste0('burnin (',burnin,') cannot exceed or match the total number of sweeps (',sweeps,')'))
    }

    muhat.draws <- matrix(NA, nrow(x_con), sweeps - burnin)
    seq <- (burnin+1):sweeps

    for (i in seq) {
        muhat.draws[, i - burnin] = obj$predicted_values[,i] * model$sdy * (model$a_draws[nrow(model$a_draws), i]) + model$meany
    }

    muhats <- rowMeans(muhat.draws)
    return(muhats)
}

#' Get post-burnin draws from trained model with gaussian process on treatment forest (to extrapolate)
#'
#' @param model A trained XBCF model.
#' @param x_con An input matrix for the prognostic term of size nt by p1 of the testing set. Column order matters: continuos features should all bgo before of categorical.
#' @param x_mod An input matrix for the treatment term of size nt by p2 (default x_mod = x_con) of the testing set. Column order matters: continuos features should all go before categorical.
#' @param xtrain_mod An input matrix for the treatment term of size nt by p2 (default x_mod = x_con) of the training set. Column order matters: continuos features should all go before categorical.
#' @param y An array of outcome variables of length n (expected to be continuos).
#' @param z A binary array of treatment assignments of length n.
#' @param pihat An array of propensity score estimates (default is NULL). In the default case propensity scores are estimated using nnet function.
#' @param burnin The number of burn-in iterations to discard from prediction (the default value is taken from the trained model).
#'
#' @return A list with two matrices. Each matrix corresponds to a set of draws of predicted values; rows are datapoints, columns are iterations.
#' @export
predictGP <- function(model, y, z, xtrain_con, xtrain_mod = xtrain_con, x_con, x_mod=x_con,
                    pihat_tr = NULL, pihat_te = NULL, theta = 1, tau = NULL, burnin=NULL, verbose = FALSE, 
                    parallel = TRUE, set_random_seed = FALSE, random_seed = 0) {

    if(!("matrix" %in% class(x_con))) {
        cat("Msg: input x_con is not a matrix, try to convert type.\n")
        x_con = as.matrix(x_con)
    }
    if(!("matrix" %in% class(x_mod))) {
        cat("Msg: input x_mod is not a matrix, try to convert type.\n")
        x_mod = as.matrix(x_mod)
    }

    if(!("matrix" %in% class(xtrain_con))) {
        cat("Msg: input x_con is not a matrix, try to convert type.\n")
        xtrain_con = as.matrix(xtrain_con)
    }
    if(!("matrix" %in% class(xtrain_mod))) {
        cat("Msg: input x_mod is not a matrix, try to convert type.\n")
        xtrain_mod = as.matrix(xtrain_mod)
    }
    if(!("matrix" %in% class(z))){
        cat("Msg: input z is not a matrix, try to convert type.\n")
        z = as.matrix(z)
    }
    if(!("matrix" %in% class(y))){
        cat("Msg: input y is not a matrix, try to convert type.\n")
        y = as.matrix(y)
    }

    if(ncol(x_con) != model$input_var_count$x_con) {
        stop(paste0('Check dimensions of input matrices. The model was trained on
        x_con with ', model$input_var_count$x_con,
        ' columns; trying to predict on x_con with ', ncol(x_con),' columns.'))
    }
    if(ncol(x_mod) != model$input_var_count$x_mod) {
        stop(paste0('Check dimensions of input matrices. The model was trained on
        x_con with ', model$input_var_count$x_mod,
        ' columns; trying to predict on x_con with ', ncol(x_mod),' columns.'))
    }

    if(is.null(pihat_tr)) {
        sink("/dev/null") # silence output
        fitz = nnet::nnet(z~.,data = xtrain_con, size = 3,rang = 0.1, maxit = 1000, abstol = 1.0e-8, decay = 5e-2)
        sink() # close the stream
        pihat_tr = as.matrix(fitz$fitted.values)
    }
    if(is.null(pihat_te)){
        if (!exists("fitz")){
            sink("/dev/null") # silence output
            fitz = nnet::nnet(z~.,data = xtrain_con, size = 3,rang = 0.1, maxit = 1000, abstol = 1.0e-8, decay = 5e-2)
            sink() # close the stream
        }
        pihat_te = as.matrix(stats::predict(fitz, x_con))
    }
    if(!("matrix" %in% class(pihat_tr))) {
        cat("Msg: input pihat is not a matrix, try to convert type.\n")
        pihat_tr = as.matrix(pihat_tr)
    }
    if(!("matrix" %in% class(pihat_te))) {
        cat("Msg: input pihat is not a matrix, try to convert type.\n")
        pihat_te = as.matrix(pihat_te)
    }

    if(ncol(pihat_tr) != 1) {
        stop(paste0('Propensity score input must be a 1-column matrix or NULL (default).
        A matrix with ', ncol(pihat_tr), ' columns was provided instead.'))
    }
    if(ncol(pihat_te) != 1) {
        stop(paste0('Propensity score input must be a 1-column matrix or NULL (default).
        A matrix with ', ncol(pihat_te), ' columns was provided instead.'))
    }

    if (is.null(tau)){
        cat("Set tau = var(y)/num_trees")
        tau = var(y) / model$model_params$num_trees_trt
    }

    xtrain_con <- cbind(model$pihat, xtrain_con)
    x_con <- cbind(pihat_te, x_con)

    y = y - model$meany
    if(model$sdy == 0) {
        stop('y is a constant variable; sdy = 0')
    } else {
        y = y / model$sdy
    }
    mutr = .Call(`_XBCF_xbcf_predict`, xtrain_con, model$model_list$tree_pnt_pr)
    tautr = .Call(`_XBCF_xbcf_predict`, xtrain_mod, model$model_list$tree_pnt_trt)
    objmu = .Call(`_XBCF_xbcf_predict`, x_con, model$model_list$tree_pnt_pr)
    objtau = .Call(`_XBCF_xbcf_predict`, x_mod, model$model_list$tree_pnt_trt)

    objmu.gp = .Call(`_XBCF_predict_gp`, 0, y, z, xtrain_con, x_con, model$model_list$tree_pnt_pr, 
                tautr$predicted_values, model$pihat, pihat_te, 
                model$sigma0_draws, model$sigma1_draws, 
                model$a_draws, model$b0_draws, model$b1_draws,
                theta, tau, model$model_params$p_categorical_trt,
                verbose, parallel, set_random_seed, random_seed)

    objtau.gp = .Call(`_XBCF_predict_gp`, 1, y, z, xtrain_mod, x_mod, model$model_list$tree_pnt_trt, 
                mutr$predicted_values, model$pihat, pihat_te, 
                model$sigma0_draws, model$sigma1_draws, 
                model$a_draws, model$b0_draws, model$b1_draws,
                theta, tau, model$model_params$p_categorical_trt,
                verbose, parallel, set_random_seed, random_seed)

    sweeps <- ncol(model$tauhats)
    if(is.null(burnin)) {
        burnin <- model$model_params$burnin
    }

    if(burnin >= sweeps) {
        stop(paste0('burnin (',burnin,') cannot exceed or match the total number of sweeps (',sweeps,')'))
    }

    mu.adjusted <- matrix(NA, nrow(x_con), sweeps - burnin)
    tau.adjusted <- matrix(NA, nrow(x_mod), sweeps - burnin)
    tau.old <- matrix(NA, nrow(x_mod), sweeps - burnin)
    seq <- (burnin+1):sweeps

    mu0.adjusted <- matrix(NA, nrow(x_con), sweeps - burnin)
    mu1.adjusted <- matrix(NA, nrow(x_con), sweeps - burnin)  
    tau0.adjusted <- matrix(NA, nrow(x_con), sweeps - burnin)
    tau1.adjusted <- matrix(NA, nrow(x_con), sweeps - burnin) 
    tau.4gp <- matrix(NA, nrow(x_con), sweeps - burnin)    

    for (i in seq) {
        # tau.gp[, i - burnin] = objtau.gp$predicted_values[,i] * model$sdy * (model$b1_draws[nrow(model$b1_draws), i] - model$b0_draws[nrow(model$b0_draws), i])
        mu.adjusted[, i - burnin] = objmu$predicted_values[,i] * model$sdy * (model$a_draws[nrow(model$a_draws), i]) + model$meany
        mu0.adjusted[, i - burnin] = objmu.gp$y0[,i] * model$sdy * (model$a_draws[nrow(model$a_draws), i]) + model$meany
        mu1.adjusted[, i - burnin] = objmu.gp$y1[,i] * model$sdy * (model$a_draws[nrow(model$a_draws), i]) + model$meany
        tau0.adjusted[, i - burnin] = objtau.gp$y0[,i] * model$sdy * (model$b1_draws[nrow(model$b1_draws), i] - model$b0_draws[nrow(model$b0_draws), i])
        tau1.adjusted[, i - burnin] = objtau.gp$y1[,i] * model$sdy * (model$b1_draws[nrow(model$b1_draws), i] - model$b0_draws[nrow(model$b0_draws), i])
        tau.adjusted[, i - burnin] = objtau$predicted_values[,i] * model$sdy * (model$b1_draws[nrow(model$b1_draws), i] - model$b0_draws[nrow(model$b0_draws), i])
        tau.4gp[, i - burnin] = tau.adjusted[, i-burnin] + mu1.adjusted[,i-burnin] - mu0.adjusted[, i-burnin] + tau1.adjusted[,i-burnin]- tau0.adjusted[,i-burnin]
    }

    obj <- list(mu.adjusted=mu.adjusted, tau.adjusted=tau.adjusted, tau.4gp = tau.4gp, 
                mu0 = mu0.adjusted, mu1 = mu1.adjusted, tau0 = tau0.adjusted, tau1 = tau1.adjusted)

    return(obj)
}