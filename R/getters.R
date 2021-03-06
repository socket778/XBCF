# get individual level treatment estimates
getTaus <- function(model) {

    tauhats <- rowMeans(model$tauhats.adjusted)

    return(tauhats)
}

# get individual level prognostic estimates
getMus <- function(model) {

    tauhats <- rowMeans(model$muhats.adjusted)

    return(muhats)
}