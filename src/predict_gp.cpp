#include "Rcpp.h"
#include <armadillo>
#include "predict_gp.h"
// #include "train_all.cpp"
#include "utility.h"
#include "utility_rcpp.h"

using namespace arma;

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
                    matrix<double>& X_range, 
                    const double &theta, const double &tau
                    )
{
    //cout << "size of Xorder std " << Xorder_std.size() << endl;
    //cout << "size of Xorder tau " << Xorder_tau_std.size() << endl;
    if (state->parallel)
        thread_pool.start();

    // model_trt->set_state_status(state, 1, X_tau_std, Xorder_tau_std);
    state->fl = 1; // value can only be 0 or 1 (to alternate between arms)
    state->X_std = X_tau_std;
    state->Xorder_std = Xorder_tau_std;
    state->p = state->p_trt;
    state->p_categorical = state->p_categorical_trt;
    state->p_continuous = state->p_continuous_trt;

    tree::tree_p bn; // pointer to bottom node
    std::vector<bool> active_var(state->p);
    double scale0, scale1;

    // init residual
    for (size_t i = 0; i < Xorder_tau_std[0].size();i++){
        state->residual[i] = (*state->y_std)[i];
    }
    for (size_t sweeps = 0; sweeps < state->num_sweeps; sweeps++)
    {
        //cout << "sweep: " << sweeps << endl;
        if (verbose == true)
        {
            COUT << "--------------------------------" << endl;
            COUT << "number of sweeps " << sweeps << endl;
            COUT << "--------------------------------" << endl;
        }

        ///////////////////////////////////////////////////////////////////////////
        // Input predicted values from mu trees to replace mu_fit
        std::copy(mu_fit_std[sweeps].begin(), mu_fit_std[sweeps].end(), state->mu_fit.begin());
        ///////////////////////////////////////////////////////////////////////////

        // update a, b0 and b1
        state->a = a_xinfo[sweeps][state->num_trees_vec[0] - 1];
        state->b_vec[0] = b0_xinfo[sweeps][state->num_trees_vec[0] - 1];
        state->b_vec[1] = b1_xinfo[sweeps][state->num_trees_vec[0] - 1];

        ////////////// Treatment term loop
        for (size_t tree_ind = 0; tree_ind < state->num_trees_vec[1]; tree_ind++)
        {
            // if (state->z[0] == 1){
            //     cout << "sweep = " << sweeps << " tree " << tree_ind << " resid = " << (*state->y_std)[0] - state->a * state->mu_fit[0] - state->b_vec[1] * state->tau_fit[0] << endl;
            // }
            // else{
            //     cout << "sweep = " << sweeps << " tree " << tree_ind << " resid = " << (*state->y_std)[0] - state->a * state->mu_fit[0] - state->b_vec[0] * state->tau_fit[0] << endl;
            // }
            if (verbose == true)
            {
                COUT << "--------------------------------" << endl;
                COUT << "number of trees " << tree_ind << endl;
                COUT << "--------------------------------" << endl;
            }
            // state->update_residuals(); // update residuals
            state->update_sigma(sigma0_draw_xinfo[sweeps][state->num_trees_vec[0] + tree_ind], 0);
            state->update_sigma(sigma1_draw_xinfo[sweeps][state->num_trees_vec[0] + tree_ind], 1);

            // subtract_old_tree_fit
            for (size_t i = 0; i < state->tau_fit.size(); i++)
            {
                state->tau_fit[i] -= (*(x_struct_trt->data_pointers[tree_ind][i]))[0];
            }
            
            for (size_t i = 0; i < Xorder_tau_std[0].size();i++){
                if (state->z[i] == 1)
                {
                    state->residual[i] = ((*state->y_std)[i] - state->a * state->mu_fit[i] - state->b_vec[1] * state->tau_fit[i]) / state->b_vec[1];
                }
                else
                {
                    state->residual[i] = ((*state->y_std)[i] - state->a * state->mu_fit[i] - state->b_vec[0] * state->tau_fit[i]) / state->b_vec[0];
                }
            }
            std::fill(active_var.begin(), active_var.end(), false);

            // assign predicted values to data_pointers
            // trees_trt[sweeps][tree_ind].predict_from_2gp(Xorder_tau_std, x_struct_trt, x_struct_trt->X_counts, x_struct_trt->X_num_unique, 
            // Xtestorder_tau_std, xtest_struct_trt, xtest_struct_trt->X_counts, xtest_struct_trt->X_num_unique,
            // state, X_range, active_var, y0_test_xinfo[sweeps], y1_test_xinfo[sweeps], 
            // tree_ind, theta, tau, true);
            trees_trt[sweeps][tree_ind].predict_from_root_gp(Xorder_tau_std, x_struct_trt, x_struct_trt->X_counts, x_struct_trt->X_num_unique, 
            Xtestorder_tau_std, xtest_struct_trt, xtest_struct_trt->X_counts, xtest_struct_trt->X_num_unique,
            state, X_range, active_var,  y1_test_xinfo[sweeps], tree_ind, theta, tau, true);
            
            // // check residuals and theta value
            // bn = trees_trt[sweeps][tree_ind].search_bottom_std(x_struct_trt->X_std, 0, state->p, Xorder_tau_std[0].size());
            // x_struct_trt->data_pointers[tree_ind][0] = &bn->theta_vector;
            // if (state->z[0] == 1){
            //     cout << "sweeps " << sweeps << " tree " << tree_ind << " resid = " << (*state->y_std)[0] - state->a * state->mu_fit[0] - state->b_vec[1] * state->tau_fit[0] << " theta = " << (*(x_struct_trt->data_pointers[tree_ind][0]))[0] << endl;
            // }
            // else{
            //     cout << "sweeps " << sweeps << " tree " << tree_ind << " resid = " << (*state->y_std)[0] - state->a * state->mu_fit[0] - state->b_vec[0] * state->tau_fit[0] << " theta = " << (*(x_struct_trt->data_pointers[tree_ind][0]))[0] << endl;
            // }
            
            // update parital residuals here based on subtracted tau_fit
            for (size_t i = 0; i < Xorder_tau_std[0].size();i++){
                bn = trees_trt[sweeps][tree_ind].search_bottom_std(x_struct_trt->X_std, i, state->p, Xorder_tau_std[0].size());
                x_struct_trt->data_pointers[tree_ind][i] = &bn->theta_vector;
                state->tau_fit[i] += (*(x_struct_trt->data_pointers[tree_ind][i]))[0];
            }

            // update a, b0 and b1
            state->a = a_xinfo[sweeps][state->num_trees_vec[0] + tree_ind];
            state->b_vec[0] = b0_xinfo[sweeps][state->num_trees_vec[0] + tree_ind];
            state->b_vec[1] = b1_xinfo[sweeps][state->num_trees_vec[0] + tree_ind];
        }
        // if (state->z[0] == 1){
        //     cout << "sweeps " << sweeps << " resid = " << (*state->y_std)[0] - state->a * state->mu_fit[0] - state->b_vec[1] * state->tau_fit[0] << endl;
        // }
        // else{
        //     cout << "sweeps " << sweeps << " resid = " << (*state->y_std)[0] - state->a * state->mu_fit[0] - state->b_vec[0] * state->tau_fit[0] << endl;
        // }
    }

    thread_pool.stop();
    return;
    }


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
                    matrix<double>& X_range, 
                    const double &theta, const double &tau
                    )
{
    //cout << "size of Xorder std " << Xorder_std.size() << endl;
    //cout << "size of Xorder tau " << Xorder_tau_std.size() << endl;
    if (state->parallel)
        thread_pool.start();

    // model_trt->set_state_status(state, 1, X_tau_std, Xorder_tau_std);
    state->fl = 0; // value can only be 0 or 1 (to alternate between arms)
    state->X_std = X_std;
    state->Xorder_std = Xorder_std;
    state->p = state->p_pr;
    state->p_categorical = state->p_categorical_pr;
    state->p_continuous = state->p_continuous_pr;

    tree::tree_p bn; // pointer to bottom node
    std::vector<bool> active_var(state->p);

    for (size_t sweeps = 0; sweeps < state->num_sweeps; sweeps++)
    {
        //cout << "sweep: " << sweeps << endl;
        if (verbose == true)
        {
            COUT << "--------------------------------" << endl;
            COUT << "number of sweeps " << sweeps << endl;
            COUT << "--------------------------------" << endl;
        }

        ////////////// Prognostic term loop
        for (size_t tree_ind = 0; tree_ind < state->num_trees_vec[0]; tree_ind++)
        {
            if (verbose == true)
            {
                COUT << "--------------------------------" << endl;
                COUT << "number of trees " << tree_ind << endl;
                COUT << "--------------------------------" << endl;
            }
            // state->update_residuals(); // update residuals
            state->update_sigma(sigma0_draw_xinfo[sweeps][tree_ind], 0);
            state->update_sigma(sigma1_draw_xinfo[sweeps][tree_ind], 1);

            // subtract_old_tree_fit
            for (size_t i = 0; i < state->mu_fit.size(); i++)
            {
                state->mu_fit[i] -= (*(x_struct_pr->data_pointers[tree_ind][i]))[0];
            }
            // update parital residuals here based on subtracted tau_fit
            for (size_t i = 0; i < Xorder_std[0].size(); i++){
                if (state->z[i] == 1)
                {
                    state->residual[i] = ((*state->y_std)[i] - state->a * state->mu_fit[i] - state->b_vec[1] * state->tau_fit[i]) / state->a;
                }
                else
                {
                    state->residual[i] = ((*state->y_std)[i] - state->a * state->mu_fit[i] - state->b_vec[0] * state->tau_fit[i]) / state->a;
                }
            }
            std::fill(active_var.begin(), active_var.end(), false);

            // assign predicted values to data_pointers
            trees_pr[sweeps][tree_ind].predict_from_root_gp(Xorder_std, x_struct_pr, x_struct_pr->X_counts, x_struct_pr->X_num_unique, 
            Xtestorder_std, xtest_struct_pr, xtest_struct_pr->X_counts, xtest_struct_pr->X_num_unique,
            state, X_range, active_var, mu1_test_xinfo[sweeps], tree_ind, theta, tau, false);

            // trees_pr[sweeps][tree_ind].predict_from_2gp(Xorder_std, x_struct_pr, x_struct_pr->X_counts, x_struct_pr->X_num_unique, 
            // Xtestorder_std, xtest_struct_pr, xtest_struct_pr->X_counts, xtest_struct_pr->X_num_unique,
            // state, X_range, active_var, mu0_test_xinfo[sweeps], mu1_test_xinfo[sweeps], 
            // tree_ind, theta, tau, false);
            
            // // check residuals and theta value
            bn = trees_pr[sweeps][tree_ind].search_bottom_std(x_struct_pr->X_std, 0, state->p, Xorder_std[0].size());
            x_struct_pr->data_pointers[tree_ind][0] = &bn->theta_vector;
            // if (state->z[0] == 1){
            //     cout << "sweeps " << sweeps << " tree " << tree_ind << " resid = " << (*state->y_std)[0] - state->a * state->mu_fit[0] - state->b_vec[1] * state->tau_fit[0] << " theta = " << (*(x_struct_pr->data_pointers[tree_ind][0]))[0] << endl;
            // }
            // else{
            //     cout << "sweeps " << sweeps << " tree " << tree_ind << " resid = " << (*state->y_std)[0] - state->a * state->mu_fit[0] - state->b_vec[0] * state->tau_fit[0] << " theta = " << (*(x_struct_pr->data_pointers[tree_ind][0]))[0] << endl;
            // }
            
            // update parital residuals here based on subtracted tau_fit
            for (size_t i = 0; i < Xorder_std[0].size();i++){
                bn = trees_pr[sweeps][tree_ind].search_bottom_std(x_struct_pr->X_std, i, state->p, Xorder_std[0].size());
                x_struct_pr->data_pointers[tree_ind][i] = &bn->theta_vector;
                state->mu_fit[i] += (*(x_struct_pr->data_pointers[tree_ind][i]))[0];
            }

            // update a, b0 and b1
            state->a = a_xinfo[sweeps][tree_ind];
            state->b_vec[0] = b0_xinfo[sweeps][tree_ind];
            state->b_vec[1] = b1_xinfo[sweeps][tree_ind];
        }

        ///////////////////////////////////////////////////////////////////////////
        // Input predicted values from mu trees to replace mu_fit
        std::copy(tau_fit_std[sweeps].begin(), tau_fit_std[sweeps].end(), state->tau_fit.begin());
        ///////////////////////////////////////////////////////////////////////////
        // update a, b0 and b1
        state->a = a_xinfo[sweeps][state->num_trees_vec[0] + state->num_trees_vec[1] - 1];
        state->b_vec[0] = b0_xinfo[sweeps][state->num_trees_vec[0] + state->num_trees_vec[1] - 1];
        state->b_vec[1] = b1_xinfo[sweeps][state->num_trees_vec[0] + state->num_trees_vec[1] - 1];

    }

    thread_pool.stop();
    return;
    }


// [[Rcpp::export]]
Rcpp::List predict_gp(size_t fl, mat y, mat z, mat X, mat Xtest, Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt, // a_draws, b_draws,
                    mat partial_fit, mat sigma0_draws, mat sigma1_draws, mat a_draws, mat b0_draws, mat b1_draws,
                    double theta, double tau, size_t p_categorical = 0,
                    bool verbose = false, bool parallel = true, bool set_random_seed = false, size_t random_seed = 0)
{
    // should be able to run in parallel
    cout << "predict with gaussian process" << endl;

    // Size of data
    size_t N = X.n_rows;
    size_t p = X.n_cols;
    size_t N_test = Xtest.n_rows;
    // number of continuous variables
    size_t p_continuous = p - p_categorical; // only work for continuous for now
    matrix<size_t> Xorder_std;
    ini_matrix(Xorder_std, N, p);

    std::vector<double> y_std(N);
    std::vector<double> b(N);
    double y_mean = 0.0;

    Rcpp::NumericMatrix X_std(N, p);
    Rcpp::NumericMatrix Xtest_std(N_test, p);

    rcpp_to_std2(y, X, Xtest, y_std, y_mean, X_std, Xtest_std, Xorder_std);

    matrix<size_t> Xtestorder_std;
    ini_matrix(Xtestorder_std, N_test, p);
    // Create Xtestorder
    umat Xtestorder(Xtest.n_rows, Xtest.n_cols);
    for (size_t i = 0; i < Xtest.n_cols; i++)
    {
        Xtestorder.col(i) = sort_index(Xtest.col(i));
    }
    for (size_t i = 0; i < N_test; i++)
    {
        for (size_t j = 0; j < p; j++)
        {
            Xtestorder_std[j][i] = Xtestorder(i, j);
        }
    }

    // Trees
    std::vector<std::vector<tree>> *trees = tree_pnt;
    size_t num_sweeps = (*trees).size();
    size_t num_trees = (*trees)[0].size();
    std::vector<size_t> num_trees_vec(2); // vector of tree number for each of mu and tau
    num_trees_vec[1 - fl] = sigma0_draws.n_rows - num_trees;
    num_trees_vec[fl] = num_trees;

    // Create sigma0/1_draw_xinfo
    matrix<double> sigma0_draw_xinfo;
    matrix<double> sigma1_draw_xinfo;
    ini_matrix(sigma0_draw_xinfo, sigma0_draws.n_rows, sigma0_draws.n_cols);
    ini_matrix(sigma1_draw_xinfo, sigma0_draws.n_rows, sigma0_draws.n_cols);
    for (size_t i = 0; i < sigma0_draws.n_rows; i++){
        for (size_t j = 0; j < sigma0_draws.n_cols; j++){
            sigma0_draw_xinfo[j][i] = sigma0_draws(i, j);
            sigma1_draw_xinfo[j][i] = sigma1_draws(i, j);
        }
    }

    // Create a_xinfo, b_xinfo
    matrix<double> a_xinfo;
    matrix<double> b0_xinfo;
    matrix<double> b1_xinfo;
    ini_matrix(a_xinfo, a_draws.n_rows, a_draws.n_cols);
    ini_matrix(b0_xinfo, b0_draws.n_rows, b0_draws.n_cols);
    ini_matrix(b1_xinfo, b1_draws.n_rows, b1_draws.n_cols);
    for (size_t i = 0; i < a_xinfo.size(); i++){
        for (size_t j = 0; j < a_xinfo[0].size(); j++){
            a_xinfo[i][j] = a_draws(j, i);
            b0_xinfo[i][j] = b0_draws(j, i);
            b1_xinfo[i][j] = b1_draws(j, i);
        }
    }

    // double *ypointer = &y_std[0];
    double *Xpointer = &X_std[0];
    double *Xtestpointer = &Xtest_std[0]; 

    matrix<double> partial_fit_std;
    ini_matrix(partial_fit_std, N, num_sweeps);
    for (size_t i = 0; i < num_sweeps; i++){
        for (size_t j = 0; j < N; j++){
            partial_fit_std[i][j] = partial_fit(j, i);
        }
    }
    ///////////////////////////////////////////////////////////////////

    std::vector<double> sigma_vec(2); // vector of sigma0, sigma1
    sigma_vec[0] = 1.0;
    sigma_vec[1] = 1.0;

    double bscale0 = -0.5;
    double bscale1 = 0.5;

    std::vector<double> b_vec(2); // vector of b0, b1
    b_vec[0] = bscale0;
    b_vec[1] = bscale1;

    size_t n_trt = 0; // number of treated individuals TODO: remove from here and from constructor as well
    std::vector<size_t> z_std(N);
    for (size_t i = 0; i < N; i++)
    {
        z_std[i] = z(i, 0);
        b[i] = z_std[i] * bscale1 + (1 - z_std[i]) * bscale0;
        if (z_std[i] == 1)
            n_trt++;
    }

    // Get X_range
    matrix<double> X_range;
    bool overlap;
    get_overlap(Xpointer, Xorder_std, z_std, X_range, overlap);

    // State settings for the prognostic term
    std::unique_ptr<State> state(new xbcfState(Xpointer, Xorder_std, N, n_trt, p, p, num_trees_vec, p_categorical, p_categorical, 
                                p_continuous, p_continuous, set_random_seed, random_seed, 0, 1, parallel, p, p, Xpointer, 
                                num_sweeps, true, &y_std, b, z_std, sigma_vec, b_vec, 1, y_mean, 0, 1));
    state->fl = fl;
    // initialize X_struct
    std::vector<double> initial_theta(1, 0);
    std::unique_ptr<X_struct> x_struct(new X_struct(Xpointer, &y_std, N, Xorder_std, p_categorical, p_continuous, &initial_theta, num_trees));
    std::unique_ptr<X_struct> xtest_struct(new X_struct(Xtestpointer, &y_std, N_test, Xtestorder_std, p_categorical, p_continuous, &initial_theta, num_trees));
    x_struct->n_y = N;
    xtest_struct->n_y = N_test;


    matrix<double> y0_test_xinfo;
    matrix<double> y1_test_xinfo;
    ini_matrix(y0_test_xinfo, N_test, num_sweeps);
    ini_matrix(y1_test_xinfo, N_test, num_sweeps);
    for (size_t i = 0; i < num_sweeps; i++){
        std::fill(y0_test_xinfo[i].begin(), y0_test_xinfo[i].end(), 0.0);
        std::fill(y1_test_xinfo[i].begin(), y1_test_xinfo[i].end(), 0.0);
    }

    std::vector<bool> active_var(p);
    std::fill(active_var.begin(), active_var.end(), false);

    if (state->fl == 0){
        mcmc_loop_pr(Xorder_std, Xtestorder_std, Xpointer, Xtestpointer, verbose, sigma0_draw_xinfo, sigma1_draw_xinfo, 
                a_xinfo, b0_xinfo, b1_xinfo, *trees, state, x_struct, xtest_struct, partial_fit_std, y0_test_xinfo, y1_test_xinfo, 
                X_range, theta, tau);
    }else{
        mcmc_loop_trt(Xorder_std, Xtestorder_std, Xpointer, Xtestpointer, verbose, sigma0_draw_xinfo, sigma1_draw_xinfo, 
                a_xinfo, b0_xinfo, b1_xinfo, *trees, state, x_struct, xtest_struct, partial_fit_std, y0_test_xinfo, y1_test_xinfo, 
                X_range, theta, tau);
    }

    Rcpp::NumericMatrix y0_test(N_test, num_sweeps);
    Rcpp::NumericMatrix y1_test(N_test, num_sweeps);
    std_to_rcpp(y0_test_xinfo, y0_test);
    std_to_rcpp(y1_test_xinfo, y1_test);

    return Rcpp::List::create(Rcpp::Named("y0") = y0_test, Rcpp::Named("y1") = y1_test);
    
}

