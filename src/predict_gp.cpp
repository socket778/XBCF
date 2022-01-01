#include "Rcpp.h"
#include <armadillo>
#include "predict_gp.h"


void mcmc_loop_gp(matrix<size_t> &Xorder_tau_std, matrix<size_t> &Xtestorder_tau_std,
                    const double *X_tau_std, const double *Xtest_tau_std,
                    bool verbose,
                    matrix<double> &sigma0_draw_xinfo,
                    matrix<double> &sigma1_draw_xinfo,
                    matrix<double> &b_xinfo,
                    matrix<double> &a_xinfo,
                    // vector<vector<tree>> &trees_ps,
                    vector<vector<tree>> &trees_trt,
                    std::unique_ptr<State> &state,
                    //std::unique_ptr<State> &state_trt,
                    // xbcfModel *model_ps,
                    xbcfModel *model_trt,
                    std::unique_ptr<X_struct> &x_struct_trt,
                    std::unique_ptr<X_struct> &xtest_struct_trt,
                    matrix<double> &mu_fit_std,
                    matrix<double> &yhats_test_xinfo,
                    std::vector<std::vector<double>> X_range
                    )
{
    //cout << "size of Xorder std " << Xorder_std.size() << endl;
    //cout << "size of Xorder tau " << Xorder_tau_std.size() << endl;
    if (state->parallel)
        thread_pool.start();

    model_ps->set_state_status(state, 1, X_tau_std, Xorder_tau_std);

    tree::tree_p bn; // pointer to bottom node

    for (size_t sweeps = 0; sweeps < state->num_sweeps; sweeps++)
    {
        //cout << "sweep: " << sweeps << endl;
        if (verbose == true)
        {
            COUT << "--------------------------------" << endl;
            COUT << "number of sweeps " << sweeps << endl;
            COUT << "--------------------------------" << endl;
        }

        // update a, b0 and b1, although they are updated per tree, they are saved per forest.
        state->a = a_xinfo[0][sweeps];
        state->b_vec[0] = b_xinfo[0][sweeps];
        state->b_vec[1] = b_xinfo[1][sweeps];

        
        ///////////////////////////////////////////////////////////////////////////
        // Input predicted values from mu trees to replace mu_fit
        std::copy(mu_fit_std[sweeps].begin(), mu_fit_std[sweeps].end(), state->mu_fit.begin())
        ///////////////////////////////////////////////////////////////////////////

        ////////////// Treatment term loop
        for (size_t tree_ind = 0; tree_ind < state->num_trees_vec[1]; tree_ind++)
        {
            // state->update_residuals(); // update residuals
            state->update_sigma(sigma1_draw_xinfo[sweeps][state->num_trees_vec[0] + tree_ind], 1);

            // model_trt->subtract_old_tree_fit(tree_ind, state->tau_fit, x_struct_trt); // for GFR we will need partial tau_fit -- thus take out the old fitted values
            // subtract_old_tree_fit
            for (size_t i = 0; i < state->tau_fit.size(); i++)
            {
                state->tau_fit[i] -= (*(x_struct->data_pointers[tree_ind][i]))[0];
            }
            
            // update parital residuals here based on subtracted tau_fit
            for (size_t i = 0; i < Xorder_tau_std[0].size();i++){
                if (state->z[i] == 1)
                {
                    state->residual[i] = ((*state->y_std)[i] - state->a * state->mu_fit[i] - state->b_vec[1] * state->tau_fit[i]) ;
                }
                else
                {
                    state->residual[i] = ((*state->y_std)[i] - state->a * state->mu_fit[i] - state->b_vec[0] * state->tau_fit[i]);
                }
            }

            // assign predicted values to data_pointers
            trees_trt[sweeps][tree_ind].predict_from_root_gp(state, Xorder_tau_std, x_struct_trt, x_struct_trt->X_counts, x_struct_trt->X_num_unique, 
            Xtestorder_tau_std, xtest_struct_trt, xtest_struct_trt->X_counts, xtest_struct_trt->X_num_unique,
            state, active_var, p_categorical, tree_ind, theta, tau);
            
            // update data pointers and state_sweep
            for (size_t i = 0; i < state->tau_fit.size(); i++)
            {
                bn = trees_trt[sweeps][tree_ind].search_bottom_std(x_struct_trt->X_std, i, state->p, Xorder_tau_std[0].size());
                (*(x_struct_trt->data_pointers[tree_ind][i]))[0] = bn->theta_vector;
                state->tau_fit[i] += (*(x_struct_trt->data_pointers[tree_ind][i]))[0];
            }

            // assign xtest pointer to yhats_test_info
            for (size_t i = 0; i < Xtestorder_tau_std[0].size(); i++){
                yhats_test_xinfo[sweeps][i] += (*(xtest_struct_trt->data_pointers[tree_ind][i]))[0]
            }

        }
    }

    thread_pool.stop();
    return;
    }

// [[Rcpp::export]]
Rcpp::List predict_gp(mat y, mat z, mat X, mat Xtest, Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt, // a_draws, b_draws,
                    mat mu_fit, mat sigma0_draws, mat sigma1_draws, double theta, double tau, size_t p_categorical = 0,
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
    std::vector<size_t> z_std(N);
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

    // Create sigma0/1_draw_xinfo
    matrix<size_t> sigma0_draw_xinfo;
    matrix<size_t> sigma1_draw_xinfo;
    ini_matrix(sigma0_draw_xinfo, sigma0_draws.n_rows, sigma0_draws.n_cols);
    ini_matrix(sigma1_draw_xinfo, sigma0_draws.n_rows, sigma0_draws.n_cols);
    for (size_t i = 0; i < sigma0_draws.n_rows; i++){
        for (size_t j = 0; j < sigma0_draws.n_cols; j++){
            sigma0_draw_xinfo[j][i] = sigma0_draws(i, j);
            sigma1_draw_xinfo[j][i] = sigma1_draws(i, j);
        }
    }

    // double *ypointer = &y_std[0];
    double *Xpointer = &X_std[0];
    double *Xtestpointer = &Xtest_std[0];

    // Trees
    std::vector<std::vector<tree>> *trees = tree_pnt;
    size_t num_sweeps = (*trees).size();
    size_t num_trees = (*trees)[0].size();

    std::vector<double> sigma_std(num_sweeps);
    for (size_t i = 0; i < num_sweeps; i++){
        sigma_std[i] = sigma(i);
    }

    matrix<size_t> mu_fit_std;
    ini_matrix(mu_fit_std, N, num_sweeps);
    for (size_t i = 0; i < num_sweeps; i++){
        for (size_t j = 0; j < N; j++){
            mu_fit_std[i][j] = mu_fit(j, i);
        }
    }

    ///////////////////////////////////////////////////////////////////
    std::vector<double> sigma_vec(2); // vector of sigma0, sigma1
    sigma_vec[0] = 1.0;
    sigma_vec[1] = 1.0;

    double bscale0 = -0.5;
    double bscale1 = 0.5;

    std::vector<double> b_vec(2); // vector of sigma0, sigma1
    b_vec[0] = bscale0;
    b_vec[1] = bscale1;

    size_t n_trt = 0; // number of treated individuals TODO: remove from here and from constructor as well
    for (size_t i = 0; i < N; i++)
    {
        z_std[i] = z(i, 0);
        b[i] = z_std[i] * bscale1 + (1 - z_std[i]) * bscale0;
        if (z_std[i] == 1)
            n_trt++;
    }

    // Get X_range
    std::vector<std::vector<double>> X_range;
    get_overlap(X_std, Xorder_std, z_std, X_range);
    
    // State settings for the prognostic term
    std::unique_ptr<State> state(new xbcfState(Xpointer, Xorder_std, N, n_trt, p, p, num_trees, p_categorical, p_categorical, 
                                p_continuous, p_continuous, set_random_seed, random_seed, 0, 1, parallel, p, p, Xpointer, 
                                num_sweeps, true, &y_std, b, z_std, sigma_vec, b_vec, 1, y_mean, 0, 1));

    // initialize X_struct
    std::vector<double> initial_theta(1, 0);
    std::unique_ptr<X_struct> x_struct(new X_struct(Xpointer, &y_std, N, Xorder_std, p_categorical, p_continuous, &initial_theta_trt, num_trees));
    std::unique_ptr<X_struct> xtest_struct(new X_struct(Xtestpointer, &y_std, N_test, Xtestorder_std, p_categorical, p_continuous, &initial_theta, num_trees));
    x_struct->n_y = N;
    xtest_struct->n_y = N_test;


    matrix<double> yhats_test_xinfo;
    ini_matrix(yhats_test_xinfo, N_test, num_sweeps);
    for (size_t i = 0; i < num_sweeps; i++){
        std::fill(yhats_test_xinfo[i].begin(), yhats_test_xinfo[i].end(), 0.0);
    }

    std::vector<bool> active_var(p);
    std::fill(active_var.begin(), active_var.end(), false);

    mcmc_loop_gp(Xorder_std, Xtestorder_std, Xpointer, Xtestpointer, verbose, sigma0_draw_xinfo, sigma1_draw_xinfo, 
                b_xinfo, a_xinfo, *trees, state, model, x_struct, xtest_struct, mu_fit_std, yhats_test_xinfo, X_range);

    Rcpp::NumericMatrix yhats_test(N_test, num_sweeps);
    Matrix_to_NumericMatrix(yhats_test_xinfo, yhats_test);

    return Rcpp::List::create(Rcpp::Named("predicted_values") = yhats_test);
}
