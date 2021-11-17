predict.XBCF <- function(model, X, X_tau, burnin) {
    obj = .Call(`_XBCF_xbcf_predict`, X, X_tau,
                model$model_list$tree_pnt_pr, model$model_list$tree_pnt_trt)  # model$tree_pnt

    # TODO: add a check for matrix dimensions (may need to be somewhat sophisticated)

    sweeps <- ncol(model$tauhats)

    if(burnin >= sweeps){
        stop(paste0('burnin (',burnin,') cannot exceed or match the total number of sweeps (',sweeps,')'))
    }

    taus <- matrix(NA, nrow(X), sweeps - burnin)
    mus <- matrix(NA, nrow(X), sweeps - burnin)
    seq <- (burnin+1):sweeps

    for (i in seq) {
        taus[, i - burnin] = obj$tauhats[,i] * model$sdy * (model$b_draws[i,2] - model$b_draws[i,1])
        mus[, i - burnin] = obj$muhats[,i] * model$sdy * (model$a_draws[i]) + model$meany
    }

    obj$tauhats <- taus
    obj$muhats <- mus

    return(obj)
}

# predict function returning draws of treatment estimates
predictTauDraws <- function(model, X, X_tau, burnin = NULL) {
    obj = .Call(`_XBCF_xbcf_predict`, X, X_tau,
                model$model_list$tree_pnt_pr, model$model_list$tree_pnt_trt)  # model$tree_pnt

    # TODO: add a check for matrix dimensions (may need to be somewhat sophisticated)

    sweeps <- model$model_params$num_sweeps
    if(is.null(burnin)) {
        burnin <- model$model_params$burnin
    }

    if(burnin >= sweeps){
        stop(paste0('burnin (',burnin,') cannot exceed or match the total number of sweeps (',sweeps,')'))
    }

    tauhat.draws <- matrix(NA, nrow(X), sweeps - burnin)
    seq <- (burnin+1):sweeps

    for (i in seq) {
        tauhat.draws[, i - burnin] = obj$tauhats[,i] * model$sdy * (model$b_draws[i,2] - model$b_draws[i,1])
    }

    return(tauhat.draws)
}

# predict function returning treatment point-estimates (average over draws)
predictTaus <- function(model, X, X_tau, burnin = NULL) {
    obj = .Call(`_XBCF_xbcf_predict`, X, X_tau,
                model$model_list$tree_pnt_pr, model$model_list$tree_pnt_trt)  # model$tree_pnt

    # TODO: add a check for matrix dimensions (may need to be somewhat sophisticated)

    sweeps <- model$model_params$num_sweeps
    if(is.null(burnin)) {
        burnin <- model$model_params$burnin
    }

    if(burnin >= sweeps){
        stop(paste0('burnin (',burnin,') cannot exceed or match the total number of sweeps (',sweeps,')'))
    }

    tauhat.draws <- matrix(NA, nrow(X), sweeps - burnin)
    seq <- (burnin+1):sweeps

    for (i in seq) {
        tauhat.draws[, i - burnin] = obj$tauhats[,i] * model$sdy * (model$b_draws[i,2] - model$b_draws[i,1])
    }

    tauhats <- rowMeans(tauhat.draws)

    return(tauhats)
}

# predict function returning draws of prognostic estimates
predictMuDraws <- function(model, X, X_tau, burnin = NULL) {
    obj = .Call(`_XBCF_xbcf_predict`, X, X_tau,
                model$model_list$tree_pnt_pr, model$model_list$tree_pnt_trt)  # model$tree_pnt

    # TODO: add a check for matrix dimensions (may need to be somewhat sophisticated)

    sweeps <- model$model_params$num_sweeps
    if(is.null(burnin)) {
        burnin <- model$model_params$burnin
    }

    if(burnin >= sweeps){
        stop(paste0('burnin (',burnin,') cannot exceed or match the total number of sweeps (',sweeps,')'))
    }

    muhat.draws <- matrix(NA, nrow(X), sweeps - burnin)
    seq <- (burnin+1):sweeps

    for (i in seq) {
        muhat.draws[, i - burnin] = obj$muhats[,i] * model$sdy * (model$a_draws[i]) + model$meany
    }

    return(muhat.draws)
}

# predict function returning prognostic point-estimates (average over draws)
predictMu <- function(model, X, X_tau, burnin = NULL) {
    obj = .Call(`_XBCF_xbcf_predict`, X, X_tau,
                model$model_list$tree_pnt_pr, model$model_list$tree_pnt_trt)  # model$tree_pnt

    # TODO: add a check for matrix dimensions (may need to be somewhat sophisticated)

    sweeps <- model$model_params$num_sweeps
    if(is.null(burnin)) {
        burnin <- model$model_params$burnin
    }

    if(burnin >= sweeps){
        stop(paste0('burnin (',burnin,') cannot exceed or match the total number of sweeps (',sweeps,')'))
    }

    muhat.draws <- matrix(NA, nrow(X), sweeps - burnin)
    seq <- (burnin+1):sweeps

    for (i in seq) {
        muhat.draws[, i - burnin] = obj$muhats[,i] * model$sdy * (model$a_draws[i]) + model$meany
    }

    muhats <- rowMeans(muhat.draws)
    return(muhats)
}