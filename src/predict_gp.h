#include <ctime>
#include "tree.h"
#include "forest.h"
#include <chrono>
#include "mcmc_loop.h"
#include "utility.h"
#include "json_io.h"
#include "xbcf_model.h"


void mcmc_loop_gp(matrix<size_t> &Xorder_tau_std, matrix<size_t> &Xtestorder_tau_std,
                    const double *X_tau_std, const double *Xtest_tau_std,
                    bool verbose,
                    matrix<double> &sigma0_draw_xinfo,
                    matrix<double> &sigma1_draw_xinfo,
                    matrix<double> &b_xinfo,
                    matrix<double> &a_xinfo,
                    // vector<vector<tree>> &trees_ps,
                    vector<vector<tree>> &trees_trt,
                    // double no_split_penality,
                    std::unique_ptr<State> &state,
                    //std::unique_ptr<State> &state_trt,
                    // xbcfModel *model_ps,
                    xbcfModel *model_trt,
                    // std::unique_ptr<X_struct> &x_struct_ps,
                    std::unique_ptr<X_struct> &x_struct_trt,
                    std::unique_ptr<X_struct> &xtest_struct_trt,
                    // bool a_scaling,
                    // bool b_scaling,
                    matrix<double> &mu_fit_std
                    );