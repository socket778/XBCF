#' This function allows to get individual level treatment estimates.
#'
#' @param fit A wsbcf or XBCF fit object.
#'
#' @return An array of treatment effect point estimates.
#' @export
getTaus <- function(fit) {
    if(class(fit) != "XBCF" && class(fit) != "wsbcf")
        stop("Can only get taus for an XBCF or WSBCF object.")
    if(class(fit) == "XBCF")
        tauhats <- rowMeans(fit$tauhats.adjusted)
    if(class(fit) == "wsbcf")
        tauhats <- rowMeans(fit$tau_draws)

    return(tauhats)
}

#' This function allows to get individual level prognostic estimates.
#'
#' @param fit A wsbcf or XBCF fit object.
#'
#' @return An array of prognostic effect point estimates.
#' @export
getMus <- function(fit) {
    if(class(fit) != "XBCF" && class(fit) != "wsbcf")
        stop("Can only get taus for an XBCF or WSBCF object.")
    if(class(fit) == "XBCF")
        muhats <- rowMeans(fit$muhats.adjusted)
    if(class(fit) == "wsbcf")
        muhats <- rowMeans(fit$mu_draws)

    return(muhats)
}