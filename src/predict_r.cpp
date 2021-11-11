#include <ctime>
#include <RcppArmadillo.h>
#include "tree.h"
#include "forest.h"
#include <chrono>
#include "mcmc_loop.h"
#include "utility.h"
#include "json_io.h"
#include "xbcf_model.h"

// [[Rcpp::export]]
Rcpp::List xbcf_predict(arma::mat X,
                        arma::mat X_tau,
                        Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt_pr,
                        Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt_trt)
{

    // Process the prognostic input
    size_t N_pr = X.n_rows;
    size_t p_pr = X.n_cols;

    // Init X_std matrix
    Rcpp::NumericMatrix X_std(N_pr, p_pr);
    for (size_t i = 0; i < N_pr; i++)
    {
        for (size_t j = 0; j < p_pr; j++)
        {
            X_std(i, j) = X(i, j);
        }
    }
    double *Xpointer = &X_std[0];

    // Process the treatment input
    size_t N_trt = X_tau.n_rows;
    size_t p_trt = X_tau.n_cols;

    // Init X_std matrix
    Rcpp::NumericMatrix X_tau_std(N_trt, p_trt);
    for (size_t i = 0; i < N_trt; i++)
    {
        for (size_t j = 0; j < p_trt; j++)
        {
            X_tau_std(i, j) = X_tau(i, j);
        }
    }
    double *X_tau_pointer = &X_tau_std[0];

    // add a check N_trt == N_pr (maybe a lot earlier)
    size_t N = N_trt;

    // Trees
    std::vector<std::vector<tree>> *trees_pr = tree_pnt_pr;
    std::vector<std::vector<tree>> *trees_trt = tree_pnt_trt;

    // Result Container
    matrix<double> tauhats_test_xinfo;
    matrix<double> muhats_test_xinfo;
    size_t N_sweeps = (*trees_trt).size();
    ini_xinfo(tauhats_test_xinfo, N, N_sweeps);
    ini_xinfo(muhats_test_xinfo, N, N_sweeps);

    // number of trees
    size_t M_pr = (*trees_pr)[0].size();
    size_t M_trt = (*trees_trt)[0].size();

    xbcfModel *model = new xbcfModel();

    // Predict
    model->predict_std(Xpointer, N, p_pr, N_sweeps,
                       muhats_test_xinfo, *trees_pr);
    model->predict_std(X_tau_pointer, N, p_trt, N_sweeps,
                       tauhats_test_xinfo, *trees_trt);

    // Convert back to Rcpp
    Rcpp::NumericMatrix muhats(N, N_sweeps);
    Rcpp::NumericMatrix tauhats(N, N_sweeps);

    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < N_sweeps; j++)
        {
            muhats(i, j) = muhats_test_xinfo[j][i];
            tauhats(i, j) = tauhats_test_xinfo[j][i];
        }
    }

    return Rcpp::List::create(Rcpp::Named("muhats") = muhats,
                              Rcpp::Named("tauhats") = tauhats);
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