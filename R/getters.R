#' This function allows to get individual level treatment estimates.
#'
#' @param fit An XBCF fit object.
#'
#' @return An array of treatment effect point estimates.
#' @export
getTaus <- function(fit) {
    if(class(fit) != "XBCF")
        stop("Can only get taus for an XBCF object.")
    else
        tauhats <- rowMeans(fit$tauhats.adjusted)

    return(tauhats)
}

#' This function allows to get individual level prognostic estimates.
#'
#' @param fit An XBCF fit object.
#'
#' @return An array of prognostic effect point estimates.
#' @export
getMus <- function(fit) {
    if(class(fit) != "XBCF")
        stop("Can only get taus for an XBCF object.")
    else
        muhats <- rowMeans(fit$muhats.adjusted)

    return(muhats)
}