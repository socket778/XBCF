#include <iostream>
#include <vector>
#include <xbcf_mcmc_loop.h>
#include <json_io.h>
#include <model.h>

// TODO: 1. find where this get_M getter is used and what for
// 			 2. rewrite fit

struct XBCFcppParams
{
	size_t num_sweeps;
	size_t burnin;
	size_t max_depth;
	size_t n_min;
	size_t num_cutpoints;
	double no_split_penalty;
	size_t mtry_pr;
	size_t mtry_trt;
	size_t p_categorical_pr;
	size_t p_categorical_trt;
	size_t num_trees_pr;
	double alpha_pr;
	double beta_pr;
	double tau_pr;
	double kap_pr;
	double s_pr;
	bool pr_scale;
	size_t num_trees_trt;
	double alpha_trt;
	double beta_trt;
	double tau_trt;
	double kap_trt;
	double s_trt;
	bool trt_scale;
	bool verbose;
	bool parallel;
	bool set_random_seed;
	size_t random_seed;
	bool sample_weights_flag;
	bool a_scaling;
	bool b_scaling;
};

class XBCFcpp
{
private:
	XBCFcppParams params;
	vector<vector<tree>> trees;
	double y_mean;
	size_t n_train;
	size_t n_test;
	size_t d;
	matrix<double> yhats_xinfo;
	matrix<double> yhats_test_xinfo;
	matrix<double> sigma_draw_xinfo;
	vec_d mtry_weight_current_tree;

	// multinomial
	vec_d yhats_test_multinomial;
	size_t num_classes;
	//xinfo split_count_all_tree;

	// helper functions
	void np_to_vec_d(int n, double *a, vec_d &y_std);
	void np_to_col_major_vec(int n, int d, double *a, vec_d &x_std);
	void xinfo_to_np(matrix<double> x_std, double *arr);
	void compute_Xorder(size_t n, size_t d, const vec_d &x_std_flat, matrix<size_t> &Xorder_std);
	size_t seed;
	bool seed_flag;
	size_t model_num; // 0 : normal, 1 : multinomial; 2 : probit
	double no_split_penality;

public:
	// Constructors
	XBCFcpp(XBCFcppParams params);
	XBCFcpp(size_t num_sweeps, size_t burnin,									 // burnin is the # of burn-in sweeps
					size_t max_depth, size_t n_min,										 // n_min is the minimum node size
					size_t num_cutpoints,															 // # of adaptive cutpoints considered at each split for cont variables
					double no_split_penality,													 // penalty for not splitting
					size_t mtry_pr, size_t mtry_trt,									 // mtry is the # of variables considered at each split
					size_t p_categorical_pr,													 // # of categorical regressors
					size_t p_categorical_trt,													 // # of categorical regressors
					size_t num_trees_pr,															 // --- Prognostic term parameters start here
					double alpha_pr, double beta_pr, double tau_pr,		 // BART prior parameters
					double kap_pr, double s_pr,												 // prior parameters of sigma
					bool pr_scale,																		 // use half-Cauchy prior
					size_t num_trees_trt,															 // --- Treatment term parameters start here
					double alpha_trt, double beta_trt, double tau_trt, // BART priot parameters
					double kap_trt, double s_trt,											 // prior parameters of sigma
					bool trt_scale,																		 // use half-Normal prior
					bool verbose, bool parallel,											 // optional parameters
					bool set_random_seed, size_t random_seed,
					bool sample_weights_flag, bool a_scaling, bool b_scaling);

	XBCFcpp(std::string json_string);

	std::string _to_json(void);

	void _fit(int n, int d, double *a, // Train X
						int n_y, double *a_y, size_t p_cat);

	// Getters
	int get_M(void);
	int get_N_sweeps(void) { return ((int)params.num_sweeps); };
	int get_burnin(void) { return ((int)params.burnin); };
	void get_yhats(int size, double *arr);
	void get_yhats_test(int size, double *arr);
	void get_yhats_test_multinomial(int size, double *arr);
	void get_sigma_draw(int size, double *arr);
	void _get_importance(int size, double *arr);
};
