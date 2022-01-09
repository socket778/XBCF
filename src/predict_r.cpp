#include <ctime>
// #include <RcppArmadillo.h>
#include "Rcpp.h"
#include <armadillo>
#include "tree.h"
#include "forest.h"
#include <chrono>
#include "mcmc_loop.h"
#include "utility.h"
#include "json_io.h"
#include "xbcf_model.h"

using namespace arma;

// [[Rcpp::export]]
Rcpp::List xbcf_predict(mat X,
                        Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt)
{

    // Process the prognostic input
    size_t N = X.n_rows;
    size_t p = X.n_cols;

    // Init X_std matrix
    Rcpp::NumericMatrix X_std(N, p);
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < p; j++)
        {
            X_std(i, j) = X(i, j);
        }
    }
    double *Xpointer = &X_std[0];

    // Trees
    std::vector<std::vector<tree>> *trees = tree_pnt;

    // Result Container
    matrix<double> pred_xinfo;
    size_t N_sweeps = (*trees).size();
    ini_xinfo(pred_xinfo, N, N_sweeps);

    xbcfModel *model = new xbcfModel();

    // Predict
    model->predict_std(Xpointer, N, p, N_sweeps,
                       pred_xinfo, *trees);

    // Convert back to Rcpp
    Rcpp::NumericMatrix preds(N, N_sweeps);

    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < N_sweeps; j++)
        {
            preds(i, j) = pred_xinfo[j][i];
        }
    }

    return Rcpp::List::create(Rcpp::Named("predicted_values") = preds);
}

// [[Rcpp::export]]
Rcpp::StringVector r_to_json(double y_mean, Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt)
{

    Rcpp::StringVector result(1);
    std::vector<std::vector<tree>> *trees = tree_pnt;
    json j = get_forest_json(*trees, y_mean);
    result[0] = j.dump(4);
    return result;
}

// [[Rcpp::export]]
Rcpp::List json_to_r(Rcpp::StringVector json_string_r)
{

    std::vector<std::string> json_string(json_string_r.size());
    //std::string json_string = json_string_r(0);
    json_string[0] = json_string_r(0);
    double y_mean;

    // Create trees
    vector<vector<tree>> *trees2 = new std::vector<vector<tree>>();

    // Load
    from_json_to_forest(json_string[0], *trees2, y_mean);

    // Define External Pointer
    Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt(trees2, true);

    return Rcpp::List::create(Rcpp::Named("model_list") = Rcpp::List::create(Rcpp::Named("tree_pnt") = tree_pnt,
                                                                             Rcpp::Named("y_mean") = y_mean));
}