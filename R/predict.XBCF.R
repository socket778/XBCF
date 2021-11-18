predict.XBCF <- function(model, x_con, x_mod=x_con, pihat=NULL, burnin=NULL) {

    if(!("matrix" %in% class(x_con))){
        cat("Msg: input x_con is not a matrix, try to convert type.\n")
        x_con = as.matrix(x_con)
    }
    if(!("matrix" %in% class(x_mod))){
        cat("Msg: input x_mod is not a matrix, try to convert type.\n")
        x_mod = as.matrix(x_mod)
    }

    # TODO: add a check for matrix dimensions (may need to be somewhat sophisticated)

    if(is.null(pihat)) {
        sink("/dev/null") # silence output
        fitz = nnet::nnet(z~.,data = x_con, size = 3,rang = 0.1, maxit = 1000, abstol = 1.0e-8, decay = 5e-2)
        sink() # close the stream
        pihat = fitz$fitted.values
    }
    if(!("matrix" %in% class(pihat))){
        cat("Msg: input pihat is not a matrix, try to convert type.\n")
        pihat = as.matrix(pihat)
    }

    x_con <- cbind(pihat, x_con)

    obj1 = .Call(`_XBCF_predict`, x_con, model$model_list$tree_pnt_pr)  # model$tree_pnt
    obj2 = .Call(`_XBCF_predict`, x_mod, model$model_list$tree_pnt_trt)  # model$tree_pnt


    sweeps <- ncol(model$tauhats)
    if(is.null(burnin)) {
        burnin <- model$model_params$burnin
    }

    if(burnin >= sweeps){
        stop(paste0('burnin (',burnin,') cannot exceed or match the total number of sweeps (',sweeps,')'))
    }

    mus <- matrix(NA, nrow(x_con), sweeps - burnin)
    taus <- matrix(NA, nrow(x_mod), sweeps - burnin)
    seq <- (burnin+1):sweeps

    for (i in seq) {
        taus[, i - burnin] = obj2$predicted_values[,i] * model$sdy * (model$b_draws[i,2] - model$b_draws[i,1])
        mus[, i - burnin] = obj1$predicted_values[,i] * model$sdy * (model$a_draws[i]) + model$meany
    }

    obj <- list(mudraws=mus, taudraws=taus)

    return(obj)
}

# predict function returning draws of treatment estimates
predictTauDraws <- function(model, x_mod, burnin = NULL) {

    if(!("matrix" %in% class(x_mod))){
        cat("Msg: input x_mod is not a matrix -- converting type.\n")
        x_mod = as.matrix(x_mod)
    }

    obj = .Call(`_XBCF_predict`, x_mod, model$model_list$tree_pnt_trt)  # model$tree_pnt

    # TODO: add a check for matrix dimensions (may need to be somewhat sophisticated)

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
        tauhat.draws[, i - burnin] = obj$predicted_values[,i] * model$sdy * (model$b_draws[i,2] - model$b_draws[i,1])
    }

    return(tauhat.draws)
}

# predict function returning treatment point-estimates (average over draws)
predictTaus <- function(model, x_mod, burnin = NULL) {

    if(!("matrix" %in% class(x_mod))){
        cat("Msg: input x_mod is not a matrix -- converting type.\n")
        x_mod = as.matrix(x_mod)
    }

    obj = .Call(`_XBCF_predict`, x_mod, model$model_list$tree_pnt_trt)  # model$tree_pnt

    # TODO: add a check for matrix dimensions (may need to be somewhat sophisticated)

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
        tauhat.draws[, i - burnin] = obj$predicted_values[,i] * model$sdy * (model$b_draws[i,2] - model$b_draws[i,1])
    }

    tauhats <- rowMeans(tauhat.draws)

    return(tauhats)
}

# predict function returning draws of prognostic estimates
predictMuDraws <- function(model, x_con, pihat=NULL, burnin = NULL) {

    if(!("matrix" %in% class(x_con))){
        cat("Msg: input x_con is not a matrix -- converting type.\n")
        x_con = as.matrix(x_con)
    }

    if(is.null(pihat)) {
        sink("/dev/null") # silence output
        fitz = nnet::nnet(z~.,data = x_con, size = 3,rang = 0.1, maxit = 1000, abstol = 1.0e-8, decay = 5e-2)
        sink() # close the stream
        pihat = fitz$fitted.values
    }

    if(!("matrix" %in% class(pihat))){
        cat("Msg: input pihat is not a matrix -- converting type.\n")
        pihat = as.matrix(pihat)
    }

    x_con <- cbind(pihat, x_con)

    obj = .Call(`_XBCF_predict`, x_con, model$model_list$tree_pnt_pr)  # model$tree_pnt

    # TODO: add a check for matrix dimensions (may need to be somewhat sophisticated)

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
        muhat.draws[, i - burnin] = obj$predicted_values[,i] * model$sdy * (model$a_draws[i]) + model$meany
    }

    return(muhat.draws)
}

# predict function returning prognostic point-estimates (average over draws)
predictMus <- function(model, x_con, pihat = NULL, burnin = NULL) {

    if(!("matrix" %in% class(x_con))){
        cat("Msg: input x_con is not a matrix -- converting type.\n")
        x_con = as.matrix(x_con)
    }

    if(is.null(pihat)) {
        sink("/dev/null") # silence output
        fitz = nnet::nnet(z~.,data = x_con, size = 3,rang = 0.1, maxit = 1000, abstol = 1.0e-8, decay = 5e-2)
        sink() # close the stream
        pihat = fitz$fitted.values
    }

    if(!("matrix" %in% class(pihat))){
        cat("Msg: input pihat is not a matrix -- converting type.\n")
        pihat = as.matrix(pihat)
    }

    x_con <- cbind(pihat, x_con)

    obj = .Call(`_XBCF_predict`, x_con, model$model_list$tree_pnt_pr)  # model$tree_pnt

    # TODO: add a check for matrix dimensions (may need to be somewhat sophisticated)

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
        muhat.draws[, i - burnin] = obj$predicted_values[,i] * model$sdy * (model$a_draws[i]) + model$meany
    }

    muhats <- rowMeans(muhat.draws)
    return(muhats)
}