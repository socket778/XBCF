#ifndef GUARD_utility_rcpp_
#define GUARD_utility_rcpp_

// #include<RcppArmadillo.h>
#include "Rcpp.h"
#include <armadillo>
#include "X_struct.h"
#include "omp.h"

using namespace arma;

// utility functions that rely on Rcpp packages

void rcpp_to_std2(mat y, mat X, mat Xtest, std::vector<double> &y_std, double &y_mean, Rcpp::NumericMatrix &X_std, Rcpp::NumericMatrix &Xtest_std, matrix<size_t> &Xorder_std);

void rcpp_to_std2(mat X, mat Xtest, Rcpp::NumericMatrix &X_std, Rcpp::NumericMatrix &Xtest_std, matrix<size_t> &Xorder_std);

// FUNCTION std_to_Rcpp
// transfers data from an std matrix object to an Rcpp NumericMatrix object
void std_to_rcpp(const matrix<double> &matrix_in, Rcpp::NumericMatrix &matrix_out);

#endif