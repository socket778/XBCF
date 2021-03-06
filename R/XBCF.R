XBCF <- function(y, X, X_tau, z,
                num_sweeps = 60, burnin = 20,
                max_depth = 50, Nmin = 1L,
                num_cutpoints = 100,
                no_split_penality = "Auto", mtry_pr = 0L, mtry_trt = 0L,
                p_categorical_pr = NULL,
                p_categorical_trt = NULL,
                num_trees_pr = 30L,
                alpha_pr = 0.95, beta_pr = 1.25, tau_pr = NULL,
                kap_pr = 16, s_pr = 4,
                pr_scale = FALSE,
                num_trees_trt = 10L,
                alpha_trt = 0.25, beta_trt = 3, tau_trt = NULL,
                kap_trt = 16, s_trt = 4,
                trt_scale = FALSE,
                verbose = FALSE, parallel = TRUE,
                random_seed = NULL, sample_weights_flag = TRUE,
                a_scaling = TRUE, b_scaling = TRUE) {

    #index = order(z, decreasing=TRUE)

    #y = y[index]
    #X = matrix(c(x[,1][index],x[,2][index],x[,3][index]),nrow=length(x[,1]))
    #z = z[index]

    if(class(X)[1] != "matrix"){
        cat("Msg: input X is not a matrix, try to convert type.\n")
        X = as.matrix(X)
    }
    if(class(X_tau)[1] != "matrix"){
        cat("Msg: input X_tau is not a matrix, try to convert type.\n")
        X_tau = as.matrix(X_tau)
    }
    if(class(z) != "matrix"){
        cat("Msg: input z is not a matrix, try to convert type.\n")
        z = as.matrix(z)
    }
    if(class(y) != "matrix"){
        cat("Msg: input y is not a matrix, try to convert type.\n")
        y = as.matrix(y)
    }

    p_X <- ncol(X)
    p_Xt <- ncol(X_tau)

    # replace X_tau with pihat and assemble this matrix here
    #if(nrow(X) != nrow(X_tau)) {
    #    stop('row number mismatch ...')
    #}
    if (nrow(X) != nrow(y)) {
        stop(paste0('row number mismatch between X (', nrow(X), ') and y (', nrow(y), ')'))
    }
    if (nrow(X) != nrow(z)) {
        stop(paste0('row number mismatch between X (', nrow(X), ') and z (', nrow(z), ')'))
    }

    # check if p_categorical was not provided
    if(is.null(p_categorical_pr)) {
        stop('number of categorical variables p_categorical_pr is not specified')
    }
    if(is.null(p_categorical_trt)) {
        stop('number of categorical variables p_categorical_trt is not specified')
    }

    # check if p_categorical exceeds the number of columns
    if(p_categorical_pr > p_X) {
        stop('number of categorical variables (p_categorical_pr) cannot exceed number of columns')
    }
    if(p_categorical_trt > p_Xt) {
        stop('number of categorical variables (p_categorical_trt) cannot exceed number of columns')
    }

    # check if p_categorical is negative
    if(p_categorical_pr < 0 || p_categorical_trt < 0) {
        stop('number of categorical values can not be negative: check p_categorical_pr and p_categorical_trt')
    }

    # check if mtry exceeds the number of columns
    if(mtry_pr > p_X) {
        cat('Msg: mtry value cannot exceed number of columns; set to default.\n')
        mtry_pr <- 0
    }
    if(mtry_trt > p_Xt) {
        cat('Msg: mtry value cannot exceed number of columns; set to default.\n')
        mtry_trt <- 0
    }

    # check if mtry is negative
    if(mtry_pr < 0) {
        cat('Msg: mtry value cannot exceed number of columns; set to default.\n')
        mtry_pr <- 0
    }
    if(mtry_trt < 0) {
        cat('Msg: mtry value cannot exceed number of columns; set to default.\n')
        mtry_trt <- 0
    }

    meany = mean(y)
    y = y - meany
    sdy = sd(y)

    if(sdy == 0) {
        stop('y is a constant variable; sdy = 0')
    } else {
        y = y / sdy
    }

    # compute default values for taus if none provided
    if(is.null(tau_pr)) {
        tau_pr <- 0.6*var(y)/num_trees_pr
    }

    if(is.null(tau_trt)) {
        tau_trt <- 0.1*var(y)/num_trees_trt
    }

    if(is.null(random_seed)){
        set_random_seed = FALSE
        random_seed = 0;
    }else{
        cat("Set random seed as ", random_seed, "\n")
        set_random_seed = TRUE
    }

    if(burnin >= num_sweeps){
        stop(paste0('burnin (',burnin,') cannot exceed or match the total number of sweeps (',sweeps,')'))
    }
    if(no_split_penality == "Auto"){
        no_split_penality = log(num_cutpoints)
    }

    obj = XBCF_cpp(y, X, X_tau, z,
                         num_sweeps, burnin,
                         max_depth, Nmin,
                         num_cutpoints,
                         no_split_penality, mtry_pr, mtry_trt,
                         p_categorical_pr,
                         p_categorical_trt,
                         num_trees_pr,
                         alpha_pr, beta_pr, tau_pr,
                         kap_pr, s_pr,
                         pr_scale,
                         num_trees_trt,
                         alpha_trt, beta_trt, tau_trt,
                         kap_trt, s_trt,
                         trt_scale,
                         verbose, parallel, set_random_seed,
                         random_seed, sample_weights_flag,
                         a_scaling, b_scaling)
    class(obj) = "XBCF"

    #obj$sdy_use = sdy_use
    obj$sdy = sdy
    obj$meany = meany
    obj$tauhats = obj$tauhats * sdy
    obj$muhats = obj$muhats * sdy

    obj$tauhats.adjusted <- matrix(NA, length(y), num_sweeps-burnin)
    obj$muhats.adjusted <- matrix(NA, length(y), num_sweeps-burnin)
    seq <- (burnin+1):num_sweeps
    for (i in seq) {
        obj$tauhats.adjusted[, i - burnin] = obj$tauhats[,i] * (obj$b_draws[i,2] - obj$b_draws[i,1])
        obj$muhats.adjusted[, i - burnin] = obj$muhats[,i] * (obj$a_draws[i]) + meany
    }
    return(obj)
}
