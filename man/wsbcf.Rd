% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/wsbcf.R
\name{wsbcf}
\alias{wsbcf}
\title{This function runs the warm-start model on the given data.}
\usage{
wsbcf(
  y,
  z,
  x_con,
  x_mod = x_con,
  n_sim = 100,
  n_burn = 10,
  cores = NULL,
  pihat = NULL,
  xbcf_fit = NULL,
  xbcf_sweeps = 60,
  xbcf_burn = 20,
  pcat_con = NULL,
  pcat_mod = NULL,
  n_trees_con = 30,
  n_trees_mod = 10,
  alpha_con = 0.95,
  beta_con = 1.25,
  tau_con = NULL,
  alpha_mod = 0.25,
  beta_mod = 3,
  tau_mod = NULL
)
}
\arguments{
\item{y}{An array of outcome variables of length n (expected to be continuos).}

\item{z}{A binary array of treatment assignments of length n.}

\item{x_con}{An input matrix for the prognostic term of size n by p1. Column order matters: continuos features should all bgo before of categorical.}

\item{x_mod}{An input matrix for the treatment term of size n by p2 (default x_mod = x_con). Column order matters: continuos features should all go beforeof categorical.}

\item{n_sim}{The number of post-burnin iterations (default is 100).}

\item{n_burn}{The number of burnin iterations (default is 10).}

\item{cores}{The number of cores available for the model (default is NULL, i.e. autodect and use all).}

\item{pihat}{An array of propensity score estimates of length n (default is NULL). In the default case propensity scores are evaluated within the wsbcf function with nnet.}

\item{xbcf_fit}{A fit object from XBCF model (default is NULL). In the default case XBCF model is run within the wsbcf function.}

\item{xbcf_sweeps}{Total number of sweeps for the XBCF run (default is 60).}

\item{xbcf_burn}{Total number of burnin sweeps for the XBCF run (default is 20).}

\item{pcat_con}{The number of categorical inputs in the prognostic term input matrix x_con.}

\item{pcat_mod}{The number of categorical inputs in the treatment term input matrix x_mod.}

\item{n_trees_con}{The number of trees in the prognostic forest (default is 30).}

\item{n_trees_mod}{The number of trees in the treatment forest (default is 10).}

\item{alpha_con}{Base parameter for tree prior on trees in prognostic forest (default is 0.95).}

\item{beta_con}{Power parameter for tree prior on trees in prognostic forest (default is 1.25).}

\item{tau_con}{Prior variance over the mean on on trees in prognostic forest (default is 0.6*var(y)/n_trees_con)}

\item{alpha_mod}{Base parameter for tree prior on trees in treatment forest (default is 0.25).}

\item{beta_mod}{Power parameter for tree prior on trees in treatment forest (default is 3).}

\item{tau_mod}{Prior variance over the mean on on trees in treatment forest (default is 0.1*var(y)/n_trees_mod)}
}
\value{
A fit file, which contains the draws from the model.
}
\description{
This function runs the warm-start model on the given data.
}
