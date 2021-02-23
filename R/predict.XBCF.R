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

