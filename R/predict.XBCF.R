predict.XBCF <- function(model, X, X_tau) {
    obj = .Call(`_XBCF_xbcf_predict`, X, X_tau,
                model$model_list$tree_pnt_pr, model$model_list$tree_pnt_trt)  # model$tree_pnt
    #obj = as.matrix(obj$tauhats)
    return(obj)
}

