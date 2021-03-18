#' This function runs the warm-start model on the given data.
#'
#' @param y An array of outcome variables of length n (expected to be continuos).
#' @param z A binary array of treatment assignments of length n.
#' @param x_con An input matrix for the prognostic term of size n by p1. Column order matters: continuos features should all bgo before of categorical.
#' @param x_mod An input matrix for the treatment term of size n by p2 (default x_mod = x_con). Column order matters: continuos features should all go beforeof categorical.
#' @param pihat An array of propensity score estimates (default is NULL). In the default case propensity scores are estimated inside wsbcf using nnet function.
#' @param n_sim The number of post-burnin iterations (default is 100).
#' @param n_burn The number of burnin iterations (default is 10).
#' @param cores The number of cores available for the model (default is NULL, i.e. autodect and use all).
#' @param pihat An array of propensity score estimates of length n (default is NULL). In the default case propensity scores are evaluated within the wsbcf function with nnet.
#' @param xbcf_fit A fit object from XBCF model (default is NULL). In the default case XBCF model is run within the wsbcf function.
#' @param xbcf_sweeps Total number of sweeps for the XBCF run (default is 60).
#' @param xbcf_burn Total number of burnin sweeps for the XBCF run (default is 20).
#' @param pcat_con The number of categorical inputs in the prognostic term input matrix x_con.
#' @param pcat_mod The number of categorical inputs in the treatment term input matrix x_mod.
#' @param n_trees_con The number of trees in the prognostic forest (default is 30).
#' @param n_trees_mod The number of trees in the treatment forest (default is 10).
#' @param alpha_con Base parameter for tree prior on trees in prognostic forest (default is 0.95).
#' @param beta_con Power parameter for tree prior on trees in prognostic forest (default is 1.25).
#' @param tau_con Prior variance over the mean on on trees in prognostic forest (default is 0.6*var(y)/n_trees_con)
#' @param alpha_mod Base parameter for tree prior on trees in treatment forest (default is 0.25).
#' @param beta_mod Power parameter for tree prior on trees in treatment forest (default is 3).
#' @param tau_mod Prior variance over the mean on on trees in treatment forest (default is 0.1*var(y)/n_trees_mod)
#'
#' @return A fit file, which contains the draws from the model.
#' @export

wsbcf <- function(y, z, x_con, x_mod = x_con, n_sim = 100, n_burn = 10, cores = NULL,
                  pihat = NULL, xbcf_fit = NULL, xbcf_sweeps = 60, xbcf_burn = 20,
                  pcat_con = NULL, pcat_mod = pcat_con, n_trees_con = 30, n_trees_mod = 10,
                  alpha_con = 0.95, beta_con = 1.25, tau_con = NULL,
                  alpha_mod = 0.25, beta_mod = 3, tau_mod = NULL) {

  tm <- proc.time() # time tracker

  # re-define dopar to run it within the package
  `%dopar%` <- foreach::`%dopar%`

  # helper function to combine the results from parallel runs to stack them together
  comb <- function(x, ...) {
    lapply(seq_along(x),
           function(i) c(x[[i]], lapply(list(...), function(y) y[[i]])))
  }


  #### INPUT CHECKS
  # check if xbcf, bcf2 packages are installed
  if (!requireNamespace("XBART", quietly = TRUE)) {
    stop("Package \"XBCF\" needed for this function to work. Please install it.",
         call. = FALSE)
  }
  if (!requireNamespace("bcf2", quietly = TRUE)) {
    stop("Package \"bcf2\" needed for this function to work. Please install it.",
         call. = FALSE)
  }
  if (!requireNamespace("doParallel", quietly = TRUE)) {
    stop("Package \"doParallel\" needed for this function to work. Please install it.",
         call. = FALSE)
  }

  # set defaults for taus if it wasn't provided with the call
  if(is.null(tau_con)) {
    tau_con = 0.6 * var(y) / n_trees_con
  }
  # set defaults for taus if it wasn't provided with the call
  if(is.null(tau_mod)) {
    tau_mod = 0.1 * var(y) / n_trees_mod
  }

  # compute pihat if it wasn't provided with the call
  if(is.null(pihat)) {
    sink("/dev/null") # silence output
    fitz = nnet::nnet(z~.,data = x_con, size = 3,rang = 0.1, maxit = 1000, abstol = 1.0e-8, decay = 5e-2)
    sink() # close the stream
    pihat = fitz$fitted.values
  }

  # run xbcf if fit objects was not provided in the function call / check object class if the fit is provided with the call
  if(is.null(xbcf_fit)) {
    if(is.null(pcat_con) || is.null(pcat_mod)){
      stop("the number of categorical variables needs to be provided for XBCF")
    }
    xm <- as.matrix(x_mod)
    xc <- as.matrix(x_con)
    xbcf_fit = XBCF::XBCF(y, z, xc, xm, pihat, pcat_con = pcat_con,  pcat_mod = pcat_mod)
  } else if (class(xbcf_fit) != "XBCF") {
    stop("xbcf_fit should be an object of class XBCF")
  }

  # detect cores and use them all if cores were not specified
  if(is.null(cores)){
    cores = parallel::detectCores()
    doParallel::registerDoParallel(cores)
  }


  #### WARMSTART LOOP

  mod_tree_scaling = 1
  sw = 1 # choose a single sweep from XBCF fit above, which we will use to extract trees

  # the first initial run of warmstart to calculate cutpoint index matrices (we run it for only 2 iterations: nburn = nsim = 1)
  fit_temp =  bcf2::bcf_ini(treedraws_con = as.vector(xbcf_fit$treedraws_pr[sw]), treedraws_mod = as.vector(xbcf_fit$treedraws_trt[sw]),
                            muscale_ini = xbcf_fit$a_draws[sw, 1], bscale0_ini = 1, bscale1_ini = 1,
                            sigma_ini = 1, pi_con_tau = sqrt(tau_con), pi_con_sigma = 1, pi_mod_tau = sqrt(tau_mod),
                            pi_mod_sigma = 1, mod_tree_scaling = mod_tree_scaling,
                            y = y, z = z, x_control = x_con, x_moderate = x_mod, pihat = pihat,
                            nburn = 1, nsim = 1, include_pi = 'control',
                            use_tauscale = TRUE, ntree_control = n_trees_con, ntree_moderate = n_trees_mod, ini_bcf = FALSE, verbose = FALSE,
                            base_control = alpha_con, power_control = beta_con, base_moderate = alpha_mod, power_moderate = beta_mod)

  ws.list <- foreach::foreach (i= (xbcf_burn + 1):(xbcf_sweeps - xbcf_burn), .packages = c('bcf2'),
                      .combine='comb', .multicombine=TRUE, .init=list(list(), list())) %dopar%
    {
      # compute initialization parameters per every sweep of XBCF we initialize warm-start at
      pi_con_sigma_ini = xbcf_fit$sigma0_draws[2,i] / xbcf_fit$a_draws[i, 1]
      pi_mod_sigma_ini = xbcf_fit$sigma0_draws[2,i]
      b0_ini = xbcf_fit$b_draws[i, 1]
      b1_ini = xbcf_fit$b_draws[i, 2]

      # the main warm-start BCF function call inside the foreach loop
      # (also include tree draws, sigma draws and a draws from each XBCF sweep we initialize at)
      fit_warmstart = bcf2::bcf_ini(treedraws_con = as.vector(xbcf_fit$treedraws_pr[i]), treedraws_mod = as.vector(xbcf_fit$treedraws_trt[i]),
                                    muscale_ini = xbcf_fit$a_draws[i, 1], bscale0_ini = b0_ini, bscale1_ini = b1_ini,
                                    sigma_ini = xbcf_fit$sigma0_draws[2,i], pi_con_tau = sqrt(tau_con), pi_con_sigma = pi_con_sigma_ini, pi_mod_tau = sqrt(tau_mod),
                                    pi_mod_sigma = pi_mod_sigma_ini, mod_tree_scaling = mod_tree_scaling,
                                    y = y, z = z, x_control = x_con, x_moderate = x_mod, pihat = pihat,
                                    nburn = n_burn, nsim = n_sim, include_pi = 'control',
                                    use_tauscale = TRUE, ntree_control = n_trees_con, ntree_moderate = n_trees_mod, ini_bcf = FALSE, verbose = FALSE,
                                    base_control = alpha_con, power_control = beta_con, base_moderate = alpha_mod, power_moderate = beta_mod,
                                    x_c = fit_temp$x_c, x_m = fit_temp$x_m, cutpoint_list_c = fit_temp$xi_con, cutpoint_list_m = fit_temp$xi_mod, lambda = fit_temp$lambda)

      return(list(t(fit_warmstart$tau),t(fit_warmstart$yhat)))
    }

  # unwrap the results obtained from dopar (we don't need y draws for this experiment but they are provided for completion)
  list1 <- ws.list[[1]]
  list2 <- ws.list[[2]]
  tau_draws <- matrix(unlist(list1), nrow = length(y), byrow = FALSE)
  y_draws <- matrix(unlist(list2), nrow = length(y), byrow = FALSE)

  tm <- proc.time() - tm # time tracker

  #### OUTPUT OBJECT
  obj <- list(tau_draws = tau_draws, y_draws = y_draws)
  class(obj) <- "wsbcf"

  return(obj)
}