#include <ctime>
#include "tree.h"
#include "forest.h"
#include <chrono>
#include "mcmc_loop.h"
#include "utility.h"
#include "json_io.h"
#include "xbcf_model.h"


void mcmc_loop_trt(matrix<size_t> &Xorder_tau_std, matrix<size_t> &Xtestorder_tau_std,
                    const double *X_tau_std, const double *Xtest_tau_std,
                    bool verbose,
                    matrix<double> &sigma0_draw_xinfo,
                    matrix<double> &sigma1_draw_xinfo,
                    matrix<double> &a_xinfo,
                    matrix<double> &b0_xinfo,
                    matrix<double> &b1_xinfo,
                    vector<vector<tree>> &trees_trt,
                    std::unique_ptr<State> &state,
                    std::unique_ptr<X_struct> &x_struct_trt,
                    std::unique_ptr<X_struct> &xtest_struct_trt,
                    matrix<double> &mu_fit_std,
                    matrix<double> &y0_test_xinfo,
                    matrix<double> &y1_test_xinfo,
                    matrix<double> &X_range, 
                    const double &theta, const double &tau
                    );

void mcmc_loop_pr(matrix<size_t> &Xorder_std, matrix<size_t> &Xtestorder_std,
                const double *X_std, const double *Xtest_std,
                bool verbose,
                matrix<double> &sigma0_draw_xinfo,
                matrix<double> &sigma1_draw_xinfo,
                matrix<double> &a_xinfo,
                matrix<double> &b0_xinfo,
                matrix<double> &b1_xinfo,
                vector<vector<tree>> &trees_pr,
                std::unique_ptr<State> &state,
                std::unique_ptr<X_struct> &x_struct_pr,
                std::unique_ptr<X_struct> &xtest_struct_pr,
                matrix<double> &tau_fit_std,
                matrix<double> &mu0_test_xinfo,
                matrix<double> &mu1_test_xinfo,
                matrix<double> &X_range, 
                const double &theta, const double &tau
                );