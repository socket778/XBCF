#include <cstddef>
#include <iostream>
#include <vector>
#include "xbcf.h"
#include <utility.h>
#include <forest.h>

//using namespace std;

// TODO: 1. - rewrite constructors with the correct set of parameters
//			 2. try to ignore json (ask Saar about it if seems like an integrtal part of the code)
// 			 3. rewrite tree getters
//			 4. rewrite fit fuction (use train_all.cpp as a template)
//			 5. check logic of private helpers
//			 6. check logic of other getters
//       7. remove unnecessary pieces of code
//       8. - find ini_xinfo in the main code [utility.cpp for its description]
//       9. - re-write np_to_vec_d for std::vector<type> (no need to remove the old function)
//      10. - re-define vectors from vec_d to std::vector<type> for y and z
//      11. - overload np_to_vec_d for size_t vectors

// Constructors
XBCFcpp::XBCFcpp(XBCFcppParams params)
{
  this->params = params;
}
XBCFcpp::XBCFcpp(size_t num_sweeps, size_t burnin,                  // burnin is the # of burn-in sweeps
                 size_t max_depth, size_t n_min,                    // n_min is the minimum node size
                 size_t num_cutpoints,                              // # of adaptive cutpoints considered at each split for cont variables
                 double no_split_penality,                          // penalty for not splitting
                 size_t mtry_pr, size_t mtry_trt,                   // mtry is the # of variables considered at each split
                 size_t p_categorical_pr,                           // # of categorical regressors
                 size_t p_categorical_trt,                          // # of categorical regressors
                 size_t num_trees_pr,                               // --- Prognostic term parameters start here
                 double alpha_pr, double beta_pr, double tau_pr,    // BART prior parameters
                 double kap_pr, double s_pr,                        // prior parameters of sigma
                 bool pr_scale,                                     // use half-Cauchy prior
                 size_t num_trees_trt,                              // --- Treatment term parameters start here
                 double alpha_trt, double beta_trt, double tau_trt, // BART priot parameters
                 double kap_trt, double s_trt,                      // prior parameters of sigma
                 bool trt_scale,                                    // use half-Normal prior
                 bool verbose, bool parallel,                       // optional parameters
                 bool set_random_seed, size_t random_seed,
                 bool sample_weights_flag, bool a_scaling, bool b_scaling)
{
  this->params.num_sweeps = num_sweeps;
  this->params.burnin = burnin;
  this->params.max_depth = max_depth;
  this->params.n_min = n_min;
  this->params.num_cutpoints = num_cutpoints;
  this->no_split_penality = no_split_penality;
  this->params.mtry_pr = mtry_pr;
  this->params.mtry_trt = mtry_trt;
  this->params.p_categorical_pr = p_categorical_pr;
  this->params.p_categorical_trt = p_categorical_trt;
  this->params.num_trees_pr = num_trees_pr;
  this->params.alpha_pr = alpha_pr;
  this->params.beta_pr = beta_pr;
  this->params.tau_pr = tau_pr;
  this->params.kap_pr = kap_pr;
  this->params.s_pr = s_pr;
  this->params.pr_scale = pr_scale;
  this->params.num_trees_trt = num_trees_trt;
  this->params.alpha_trt = alpha_trt;
  this->params.beta_trt = beta_trt;
  this->params.tau_trt = tau_trt;
  this->params.kap_trt = kap_trt;
  this->params.s_trt = s_trt;
  this->params.trt_scale = trt_scale;
  this->params.verbose = verbose;
  this->params.parallel = parallel;
  this->params.set_random_seed = set_random_seed;
  this->params.random_seed = random_seed;
  this->params.sample_weights_flag = sample_weights_flag;
  this->params.a_scaling = a_scaling;
  this->params.b_scaling = b_scaling;
  this->trees_trt = vector<vector<tree>>(num_sweeps);
  this->trees_pr = vector<vector<tree>>(num_sweeps);

  // handling seed

  if (random_seed == -1)
  {
    this->seed_flag = false;
    this->seed = 0;
  }
  else
  {
    this->seed_flag = true;
    this->seed = (size_t)random_seed;
  }

  // Create trees
  for (size_t i = 0; i < num_sweeps; i++)
  {
    this->trees_pr[i] = vector<tree>(num_trees_pr);
    this->trees_trt[i] = vector<tree>(num_trees_trt);
  }

  // Initialize model
  //if(this->model_num == 0){ // NORMAL
  //define model
  // this->model = new NormalModel(this->params.kap, this->params.s, this->params.tau, this->params.alpha, this->params.beta);
  // this->model->setNoSplitPenality(no_split_penality);

  //}

  return;
}

/* don't seem to need it
XBARTcpp::XBARTcpp(std::string json_string)
{
  //std::vector<std::vector<tree>> temp_trees;
  from_json_to_forest(json_string, this->trees, this->y_mean);
  this->params.N_sweeps = this->trees.size();
  this->params.M = this->trees[0].size();
}

std::string XBARTcpp::_to_json(void)
{
  json j = get_forest_json(this->trees, this->y_mean);
  return j.dump();
}

// Getter
int XBARTcpp::get_M() { return ((int)params.M); }
*/

void XBCFcpp::_fit(int n_t, int d_t, double *a_t, // treatment
                   int n_p, int d_p, double *a_p, // prognostic
                   int n_y, double *a_y,          // y
                   int n_z, int *a_z)             // z
{

  size_t N = n_p; // just for convenience to have a single n
  size_t p_pr = d_p;
  size_t p_trt = d_t;
  // number of continuous variables
  size_t p_continuous_pr = p_pr - this->params.p_categorical_pr;
  size_t p_continuous_trt = p_trt - this->params.p_categorical_trt;

  //cout << "FLATTENING STEP" << endl;
  // Convert row major *a to column major std::vector
  // for prognostic term
  vec_d x_std_flat(n_p * d_p);
  XBCFcpp::np_to_col_major_vec(n_p, d_p, a_p, x_std_flat);
  //cout << "vector size: " << x_std_flat.size() << endl;
  //cout << "dim product: " << n_p * d_p << endl;

  // for treatment term
  vec_d xt_std_flat(n_t * d_t);
  XBCFcpp::np_to_col_major_vec(n_t, d_t, a_t, xt_std_flat);
  /* for (int i = 0; i < x_std_flat.size(); i++)
  {
    if (i % n_t == 0)
      std::cout << endl;
    std::cout << x_std_flat.at(i) << ' ';
  }*/
  //cout << endl;
  // Convert a_y to std::vector
  // vec_d y_std(n_y);
  // XBCFcpp::np_to_vec_d(n_y, a_y, y_std);
  //vec_d z_std(n_z);
  // XBCFcpp::np_to_vec_d(n_z, a_z, z_std);

  //cout << "TO VEC STEP" << endl;
  std::vector<double> y_std(N);
  XBCFcpp::np_to_vec(n_y, a_y, y_std);
  std::vector<int> zi(N);
  XBCFcpp::np_to_vec(n_z, a_z, zi);
  std::vector<double> b(N); // we don't need to process any data from the inputs for b
  //cout << "EXTRA INITS" << endl;

  std::vector<size_t> z(N);
  for (size_t i = 0; i < zi.size(); i++)
  {
    z[i] = zi[i];
  }
  //cout << "EXTRA EXTRA INITS" << endl;
  // Calculate y_mean -- can replace it with compute mean function
  double y_mean = 0.0;
  for (size_t i = 0; i < N; i++)
  {
    y_mean = y_mean + y_std[i];
  }
  y_mean = y_mean / (double)N;
  this->y_mean = y_mean;

  //cout << "XORDER INIT" << endl;
  // xorder containers
  // for prognostic term
  matrix<size_t> Xorder_std;
  ini_xinfo_sizet(Xorder_std, N, d_p);

  //cout << "matrix dim: " << Xorder_std.size() << "x" << Xorder_std[0].size() << endl;
  XBCFcpp::compute_Xorder(N, d_p, x_std_flat, Xorder_std);
  /*for (size_t i = 0; i < N; i++)
  {
    for (size_t j = 0; j < d_t; j++)
    {
      std::cout << Xorder_std[j][i] << ' ';
    }
    std::cout << endl;
  }*/

  // for treatment term
  matrix<size_t> Xorder_tau_std;
  ini_xinfo_sizet(Xorder_tau_std, N, d_t);
  XBCFcpp::compute_Xorder(N, d_t, xt_std_flat, Xorder_tau_std);

  // NOT USED ANYWHERE (stuff about depth was here)

  // parameter initialization
  //cout << "VAR DEFINITIONS" << endl;
  std::vector<double> sigma_vec(2); // vector of sigma0, sigma1
  sigma_vec[0] = 1.0;
  sigma_vec[1] = 1.0;

  double bscale0 = -0.5;
  double bscale1 = 0.5;

  std::vector<double> b_vec(2); // vector of sigma0, sigma1
  b_vec[0] = bscale0;
  b_vec[1] = bscale1;

  std::vector<size_t> num_trees(2); // vector of tree number for each of mu and tau
  num_trees[0] = this->params.num_trees_pr;
  num_trees[1] = this->params.num_trees_trt;

  size_t n_trt = 0; // number of treated individuals TODO: remove from here and from constructor as well

  // assuming we have presorted data (treated individuals first, then control group)

  for (size_t i = 0; i < N; i++)
  {
    b[i] = z[i] * bscale1 + (1 - z[i]) * bscale0;
    if (z[i] == 1)
      n_trt++;
  }
  //cout << "ntrt: " << n_trt << endl;
  /*cout << "x_std_flat: " << endl;
  for (int i = 0; i < x_std_flat.size(); i++)
  {
    if (i % n_t == 0)
      std::cout << endl;
    std::cout << x_std_flat.at(i) << ' ';
  }

  cout << "xt_std_flat: " << endl;
  for (int i = 0; i < xt_std_flat.size(); i++)
  {
    if (i % n_t == 0)
      std::cout << endl;
    std::cout << xt_std_flat.at(i) << ' ';
  }*/
  // Cpp native objects to return
  //matrix<double> yhats_xinfo; // Temp Change
  //ini_xinfo(yhats_xinfo, n, this->params.num_sweeps);
  // ini_matrix is the same with ini_xinfo with three attributes

  matrix<double> tauhats_xinfo;
  ini_matrix(tauhats_xinfo, N, this->params.num_sweeps);
  matrix<double> muhats_xinfo;
  ini_matrix(muhats_xinfo, N, this->params.num_sweeps);

  matrix<double> sigma0_draw_xinfo;
  ini_matrix(sigma0_draw_xinfo, this->params.num_trees_trt, this->params.num_sweeps);

  matrix<double> sigma1_draw_xinfo;
  ini_matrix(sigma1_draw_xinfo, this->params.num_trees_trt, this->params.num_sweeps);

  //matrix<double> a_xinfo;
  ini_matrix(this->a_xinfo, this->params.num_sweeps, 1);

  //matrix<double> b_xinfo;
  ini_matrix(this->b_xinfo, this->params.num_sweeps, 2);

  // Temp Change
  // ini_xinfo(this->sigma_draw_xinfo, this->params.num_trees_trt, this->params.num_sweeps);
  // this->mtry_weight_current_tree.resize(d); // don't need it now
  // ini_xinfo(this->split_count_all_tree, d, this->params.M); // initialize at 0
  //double *ypointer = &a_y[0];
  double *Xpointer = &x_std_flat[0];
  double *Xpointer_tau = &xt_std_flat[0];

  // Model initialization -- main chunck of code (train_all.cpp as a refernce)

  //cout << "MODEL INIT STEP" << endl;
  xbcfModel *model_pr = new xbcfModel(this->params.kap_pr, this->params.s_pr, this->params.tau_pr, this->params.alpha_pr, this->params.beta_pr);
  model_pr->setNoSplitPenality(no_split_penality);

  // define the model for the treatment term
  xbcfModel *model_trt = new xbcfModel(this->params.kap_trt, this->params.s_trt, this->params.tau_trt, this->params.alpha_trt, this->params.beta_trt);
  model_trt->setNoSplitPenality(no_split_penality);
  //cout << "STATE INIT STEP" << endl;
  // State settings for the prognostic term
  std::unique_ptr<State> state(new xbcfState(Xpointer, Xorder_std, N, n_trt, p_pr, p_trt,
                                             num_trees, this->params.p_categorical_pr, this->params.p_categorical_trt,
                                             p_continuous_pr, p_continuous_trt,
                                             this->params.set_random_seed, this->params.random_seed,
                                             this->params.n_min, this->params.num_cutpoints,
                                             this->params.parallel, this->params.mtry_pr,
                                             this->params.mtry_trt, Xpointer, this->params.num_sweeps,
                                             this->params.sample_weights_flag, &y_std, b, z, sigma_vec, b_vec,
                                             this->params.max_depth, y_mean, this->params.burnin, model_trt->dim_residual));

  //cout << "STRUCT INIT STEP" << endl;
  // initialize X_struct for the prognostic term
  std::vector<double> initial_theta_pr(1, y_mean / (double)this->params.num_trees_pr);
  std::unique_ptr<X_struct> x_struct_pr(new X_struct(Xpointer, &y_std, N, Xorder_std,
                                                     this->params.p_categorical_pr, p_continuous_pr,
                                                     &initial_theta_pr, this->params.num_trees_pr));

  // initialize X_struct for the treatment term
  std::vector<double> initial_theta_trt(1, 0);
  std::unique_ptr<X_struct> x_struct_trt(new X_struct(Xpointer_tau, &y_std, N, Xorder_tau_std,
                                                      this->params.p_categorical_trt, p_continuous_trt,
                                                      &initial_theta_trt, this->params.num_trees_trt));

  // >>>> ignoring pointers to the tree matrices -- may lead to failures
  //cout << "MCMC LOOP STEP" << endl;
  // mcmc_loop returns tauhat [N x sweeps] matrix
  mcmc_loop_xbcf(Xorder_std, Xorder_tau_std, Xpointer, Xpointer_tau, this->params.verbose,
                 sigma0_draw_xinfo, sigma1_draw_xinfo, this->b_xinfo, this->a_xinfo,
                 this->trees_pr, this->trees_trt, no_split_penality,
                 state, model_pr, model_trt, x_struct_pr, x_struct_trt,
                 this->params.a_scaling, this->params.b_scaling);

  //cout << "PREDICT STEP" << endl;

  //predict tauhats and muhats
  ini_matrix(this->muhats_xinfo, N, this->params.num_sweeps);
  for (size_t i = 0; i < N; i++)
    for (size_t j = 0; j < this->params.num_sweeps; j++)
      this->muhats_xinfo[j][i] = 0;

  ini_matrix(this->tauhats_xinfo, N, this->params.num_sweeps);
  for (size_t i = 0; i < N; i++)
    for (size_t j = 0; j < this->params.num_sweeps; j++)
      this->tauhats_xinfo[j][i] = 0;

  //cout << this->muhats_xinfo.size() << "x" << this->muhats_xinfo[0].size() << endl;
  //cout << muhats_xinfo.size() << "x" << muhats_xinfo[0].size() << endl;
  model_trt->predict_std(Xpointer_tau, N, p_trt,
                         this->params.num_sweeps, this->tauhats_xinfo, this->trees_trt);
  model_pr->predict_std(Xpointer, N, p_pr,
                        this->params.num_sweeps, this->muhats_xinfo, this->trees_pr);
  /*
  cout << "b_vec: " << endl;
  for (size_t j = 0; j < this->params.num_sweeps; j++)
    cout << this->b_xinfo[0][j] << " " << this->b_xinfo[1][j] << endl;


  cout << "muhats0:" << endl;
  for (size_t j = 0; j < this->params.num_sweeps; j++)
    cout << this->muhats_xinfo[j][0] << " ";


  cout << "tauhats0:" << endl;
  for (size_t j = 0; j < this->params.num_sweeps; j++)
    cout << this->tauhats_xinfo[j][30] << " ";
*/
  /*
  Rcpp::NumericMatrix tauhats(N, num_sweeps);
  Rcpp::NumericMatrix muhats(N, num_sweeps);
  // Rcpp::NumericMatrix b_tau(N, num_sweeps);
  Rcpp::NumericMatrix sigma0_draws(num_trees_trt, num_sweeps);
  Rcpp::NumericMatrix sigma1_draws(num_trees_trt, num_sweeps);
  // Rcpp::NumericMatrix b0_draws(num_trees_trt, num_sweeps);
  // Rcpp::NumericMatrix b1_draws(num_trees_trt, num_sweeps);
  Rcpp::NumericMatrix b_draws(num_sweeps, 2);
  Rcpp::NumericMatrix a_draws(num_sweeps, 1);

  std_to_rcpp(tauhats_xinfo, tauhats);
  std_to_rcpp(muhats_xinfo, muhats);
  // std_to_rcpp(total_fit, b_tau);
  std_to_rcpp(sigma0_draw_xinfo, sigma0_draws);
  std_to_rcpp(sigma1_draw_xinfo, sigma1_draws);
  // std_to_rcpp(b0_draw_xinfo, b0_draws);
  // std_to_rcpp(b1_draw_xinfo, b1_draws);
  std_to_rcpp(b_xinfo, b_draws);
  std_to_rcpp(a_xinfo, a_draws);
*/
  delete model_trt;
  delete model_pr;
  state.reset();
  x_struct_trt.reset();
  x_struct_pr.reset();
  //cout << "END" << endl;
}

void XBCFcpp::_predict(int n_t, int d_t, double *a_t, int term)
{ //,int size, double *arr){

  // Convert *a to col_major std::vector
  vec_d x_test_std_flat(n_t * d_t);
  XBCFcpp::np_to_col_major_vec(n_t, d_t, a_t, x_test_std_flat);

  if(term == 1) {
    // Initialize result
    ini_matrix(this->tauhats_test_xinfo, n_t, this->params.num_sweeps);
    for (size_t i = 0; i < n_t; i++)
      for (size_t j = 0; j < this->params.num_sweeps; j++)
        this->tauhats_test_xinfo[j][i] = 0;
    // Convert column major vector to pointer

    const double *Xtestpointer = &x_test_std_flat[0]; //&x_test_std[0][0];

    xbcfModel *model = new xbcfModel(this->params.kap_trt, this->params.s_trt, this->params.tau_trt, this->params.alpha_trt, this->params.beta_trt);

    model->predict_std(Xtestpointer, n_t, d_t,
                      this->params.num_sweeps, this->tauhats_test_xinfo, this->trees_trt);

    delete model;
  } else {

    // Initialize result
    ini_matrix(this->muhats_test_xinfo, n_t, this->params.num_sweeps);
    for (size_t i = 0; i < n_t; i++)
      for (size_t j = 0; j < this->params.num_sweeps; j++)
        this->muhats_test_xinfo[j][i] = 0;
    // Convert column major vector to pointer

    const double *Xtestpointer = &x_test_std_flat[0]; //&x_test_std[0][0];

    xbcfModel *model = new xbcfModel(this->params.kap_pr, this->params.s_pr, this->params.tau_pr, this->params.alpha_pr, this->params.beta_pr);

    model->predict_std(Xtestpointer, n_t, d_t,
                      this->params.num_sweeps, this->muhats_test_xinfo, this->trees_pr);

    delete model;
  }

}

// Getters
void XBCFcpp::get_muhats(int size, double *arr)
{
  xinfo_to_np(this->muhats_xinfo, arr);
}

void XBCFcpp::get_muhats_test(int size, double *arr)
{
  xinfo_to_np(this->muhats_test_xinfo, arr);
}

void XBCFcpp::get_tauhats(int size, double *arr)
{
  xinfo_to_np(this->tauhats_xinfo, arr);
}

void XBCFcpp::get_tauhats_test(int size, double *arr)
{
  xinfo_to_np(this->tauhats_test_xinfo, arr);
}

void XBCFcpp::get_b(int size, double *arr)
{
  xinfo_to_np(this->b_xinfo, arr);
}

void XBCFcpp::get_a(int size, double *arr)
{
  xinfo_to_np(this->a_xinfo, arr);
}
/*void XBCFcpp::get_yhats_test(int size, double *arr)
{
  xinfo_to_np(this->yhats_test_xinfo, arr);
}
void XBARTcpp::get_yhats_test_multinomial(int size, double *arr)
{
  for (size_t i = 0; i < size; i++)
  {
    arr[i] = this->yhats_test_multinomial[i];
  }
}
*/
/*
void XBCFcpp::get_sigma_draw(int size, double *arr)
{
  xinfo_to_np(this->sigma_draw_xinfo, arr);
}
*/
/*
void XBCFcpp::_get_importance(int size, double *arr)
{
  for (size_t i = 0; i < size; i++)
  {
    arr[i] = this->mtry_weight_current_tree[i];
  }
}
*/
// Private Helper Functions

// Numpy 1D array to vec_d - std_vector of doubles
void XBCFcpp::np_to_vec_d(int n, double *a, vec_d &y_std)
{
  for (size_t i = 0; i < n; i++)
  {
    y_std[i] = a[i];
  }
}

// two new functions for converting np vectors to std vectors
void XBCFcpp::np_to_vec(int n, double *a, std::vector<double> &vec_std)
{
  for (size_t i = 0; i < n; i++)
  {
    vec_std[i] = a[i];
  }
}

void XBCFcpp::np_to_vec(int n, int *a, std::vector<int> &vec_std)
{
  for (size_t i = 0; i < n; i++)
  {
    vec_std[i] = a[i];
  }
}

void XBCFcpp::np_to_col_major_vec(int n, int d, double *a, vec_d &x_std)
{
  for (size_t i = 0; i < n; i++)
  {
    for (size_t j = 0; j < d; j++)
    {
      size_t index = i * d + j;
      size_t index_std = j * n + i;
      x_std[index_std] = a[index];
    }
  }
}

void XBCFcpp::xinfo_to_np(matrix<double> x_std, double *arr)
{
  // Fill in array values from xinfo
  for (size_t i = 0, n = (size_t)x_std[0].size(); i < n; i++)
  {
    for (size_t j = 0, d = (size_t)x_std.size(); j < d; j++)
    {
      size_t index = i * d + j;
      arr[index] = x_std[j][i];
    }
  }
  return;
}

void XBCFcpp::compute_Xorder(size_t n, size_t d, const vec_d &x_std_flat, matrix<size_t> &Xorder_std)
{
  // Create Xorder
  std::vector<size_t> temp;
  std::vector<size_t> *xorder_std;
  for (size_t j = 0; j < d; j++)
  {
    size_t column_start_index = j * n;
    std::vector<double>::const_iterator first = x_std_flat.begin() + column_start_index;
    std::vector<double>::const_iterator last = x_std_flat.begin() + column_start_index + n;
    std::vector<double> colVec(first, last);

    temp = sort_indexes(colVec);

    xorder_std = &Xorder_std[j];
    for (size_t i = 0; i < n; i++)
      (*xorder_std)[i] = temp[i];
  }
}
