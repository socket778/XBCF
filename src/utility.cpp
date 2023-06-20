#include "utility.h"

ThreadPool thread_pool;

void ini_xinfo(matrix<double> &X, size_t N, size_t p)
{
    // matrix<double> X;
    X.resize(p);

    for (size_t i = 0; i < p; i++)
    {
        X[i].resize(N);
    }

    // return std::move(X);
    return;
}

void ini_xinfo(matrix<double> &X, size_t N, size_t p, double var)
{
    // matrix<double> X;
    X.resize(p);

    for (size_t i = 0; i < p; i++)
    {
        X[i].resize(N, var);
    }

    // return std::move(X);
    return;
}

void ini_xinfo_sizet(matrix<size_t> &X, size_t N, size_t p)
{
    // matrix<size_t> X;
    X.resize(p);

    for (size_t i = 0; i < p; i++)
    {
        X[i].resize(N);
    }

    // return std::move(X);
    return;
}

void row_sum(matrix<double> &X, std::vector<double> &output)
{
    size_t p = X.size();
    size_t N = X[0].size();
    // std::vector<double> output(N);
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < p; j++)
        {
            // COUT << X[j][i] << endl;
            output[i] = output[i] + X[j][i];
        }
    }
    return;
}

void col_sum(matrix<double> &X, std::vector<double> &output)
{
    size_t p = X.size();
    size_t N = X[0].size();
    // std::vector<double> output(p);
    for (size_t i = 0; i < p; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            output[i] = output[i] + X[i][j];
        }
    }
    return;
}

double sum_squared(std::vector<double> &v)
{
    size_t N = v.size();
    double output = 0.0;
    for (size_t i = 0; i < N; i++)
    {
        output = output + pow(v[i], 2);
    }
    return output;
}

double sum_vec(std::vector<double> &v)
{
    size_t N = v.size();
    double output = 0;
    for (size_t i = 0; i < N; i++)
    {
        output = output + v[i];
    }
    return output;
}

void seq_gen_std(size_t start, size_t end, size_t length_out, std::vector<size_t> &vec)
{
    // generate a sequence of integers, save in std vector container
    double incr = (double)(end - start) / (double)length_out;

    for (size_t i = 0; i < length_out; i++)
    {
        vec[i] = (size_t)incr * i + start;
    }

    return;
}

void seq_gen_std2(size_t start, size_t end, size_t length_out, std::vector<size_t> &vec)
{
    // generate a sequence of integers, save in std vector container
    // different from seq_gen_std
    // always put the first element 0, actual vector output have length length_out + 1!
    double incr = (double)(end - start) / (double)length_out;

    vec[0] = 0;

    for (size_t i = 1; i < length_out + 1; i++)
    {
        vec[i] = (size_t)incr * (i - 1) + start;
    }

    return;
}

void vec_sum(std::vector<double> &vector, double &sum)
{
    sum = 0.0;
    for (size_t i = 0; i < vector.size(); i++)
    {
        sum = sum + vector[i];
    }
    return;
}

void vec_sum_sizet(std::vector<size_t> &vector, size_t &sum)
{
    sum = 0;
    for (size_t i = 0; i < vector.size(); i++)
    {
        sum = sum + vector[i];
    }
    return;
}

double sq_vec_diff(std::vector<double> &v1, std::vector<double> &v2)
{
    assert(v1.size() == v2.size());
    size_t N = v1.size();
    double output = 0.0;
    for (size_t i = 0; i < N; i++)
    {
        output = output + pow(v1[i] - v2[i], 2);
    }
    return output;
}

double sq_vec_diff_sizet(std::vector<size_t> &v1, std::vector<size_t> &v2)
{
    assert(v1.size() == v2.size());
    size_t N = v1.size();
    double output = 0.0;
    for (size_t i = 0; i < N; i++)
    {
        output = output + pow(v1[i] - v2[i], 2);
    }
    return output;
}

void unique_value_count2(const double *Xpointer, matrix<size_t> &Xorder_std, //std::vector<size_t> &X_values,
                         std::vector<double> &X_values, std::vector<size_t> &X_counts, std::vector<size_t> &variable_ind, size_t &total_points, std::vector<size_t> &X_num_unique, size_t &p_categorical, size_t &p_continuous)
{
    size_t N = Xorder_std[0].size();
    size_t p = Xorder_std.size();
    double current_value = 0.0;
    size_t count_unique = 0;
    size_t N_unique;
    variable_ind[0] = 0;

    total_points = 0;
    for (size_t i = p_continuous; i < p; i++)
    {
        // only loop over categorical variables
        // suppose p = (p_continuous, p_categorical)
        // index starts from p_continuous
        X_counts.push_back(1);
        current_value = *(Xpointer + i * N + Xorder_std[i][0]);
        X_values.push_back(current_value);
        count_unique = 1;

        for (size_t j = 1; j < N; j++)
        {
            if (*(Xpointer + i * N + Xorder_std[i][j]) == current_value)
            {
                X_counts[total_points]++;
            }
            else
            {
                current_value = *(Xpointer + i * N + Xorder_std[i][j]);
                X_values.push_back(current_value);
                X_counts.push_back(1);
                count_unique++;
                total_points++;
            }
        }
        variable_ind[i + 1 - p_continuous] = count_unique + variable_ind[i - p_continuous];
        X_num_unique[i - p_continuous] = count_unique;
        total_points++;
    }

    return;
}

double normal_density(double y, double mean, double var, bool take_log)
{
    // density of normal distribution
    double output = 0.0;

    output = -0.5 * log(2.0 * 3.14159265359 * var) - pow(y - mean, 2) / 2.0 / var;
    if (!take_log)
    {
        output = exp(output);
    }
    return output;
}

bool is_non_zero(size_t x) { return (x > 0); }

size_t count_non_zero(std::vector<double> &vec)
{
    size_t output = 0;
    for (size_t i = 0; i < vec.size(); i++)
    {
        if (vec[i] != 0)
        {
            output++;
        }
    }
    return output;
}


///// !!! Note that N should be the total number of data, not the local level. This funciton needs to be updated.
void get_X_range(const double *Xpointer, std::vector< std::vector<size_t> > &Xorder_std, std::vector<std::vector<double>> &X_range)
{
    size_t N = Xorder_std[0].size();
    size_t p = Xorder_std.size();
    ini_matrix(X_range, 2, p);

    for (size_t i = 0; i < p; i++)
    {
        X_range[i][0] = *(Xpointer + i * N + Xorder_std[i][0]);
        X_range[i][1] = *(Xpointer + i * N + Xorder_std[i][N-1]);
    }

    // std::cout << "total_points " << total_points << std::endl;

    return;
}

void get_treated_range(const double *Xpointer, std::vector< std::vector<size_t> > &Xorder_std, std::vector<size_t> &z_std,
                    std::vector<std::vector<double>> &X_range)
{
    size_t n_y = z_std.size();
    size_t N = Xorder_std[0].size();
    size_t p = Xorder_std.size();
    ini_matrix(X_range, 2, p);

    for (size_t j = 0; j < p; j++)
    {
        for (size_t i = 0; i < N; i++){
            if (z_std[Xorder_std[j][i]] == 1){
                X_range[j][0] = *(Xpointer + j * n_y + Xorder_std[j][i]);
                break;
            }
        }
        for (size_t i = N; i-->0;)
        {
            if (z_std[Xorder_std[j][i]] == 1){
                X_range[j][1] = *(Xpointer + j * n_y + Xorder_std[j][i]);
                break;
            }
        }
    }

    return;
}

void get_overlap(const double *Xpointer, std::vector< std::vector<size_t> > &Xorder_std, std::vector<size_t> &z_std,
                matrix<double> &X_range, size_t &p_continuous, bool &overlap)
{
    size_t N = Xorder_std[0].size();
    size_t n_y = z_std.size();
    size_t p = p_continuous;
    ini_matrix(X_range, 2, p);
    // cout << "N = " << N << ", p = " << p <<endl;

    std::vector<std::vector<double>> X_range_trt;
    std::vector<std::vector<double>> X_range_ctrl;
    ini_matrix(X_range_trt, 2, p);
    ini_matrix(X_range_ctrl, 2, p);
    // initialize range
    for (size_t j = 0; j < p_continuous; j++){
        X_range_trt[j][0] = std::numeric_limits<double>::max();
        X_range_trt[j][1] = -std::numeric_limits<double>::max();
        X_range_ctrl[j][0] = std::numeric_limits<double>::max();
        X_range_ctrl[j][1] = -std::numeric_limits<double>::max();
    }

    // count number of samples in each group
    size_t n_trt = 0;
    size_t n_ctrl = 0;
    for (size_t i = 0; i < N; i++){
        if (z_std[Xorder_std[0][i]] == 1){
            n_trt += 1;
        }else{
            n_ctrl += 1;
        }
    }

    if ((n_trt <= 1) | (n_ctrl <= 1)) {
        overlap = false;
        for (size_t i = 0; i < p; i++){
            X_range[i][0] = std::numeric_limits<double>::max();
            X_range[i][1] = -std::numeric_limits<double>::max();
        }
        return;
    }

    // find the 2.5% quantile of treated and control sample
    // size_t cnt_trt, cnt_ctrl;
    double alpha = 0.05;
    size_t trt_low = (size_t) ceil((double)n_trt * alpha / 2);
    size_t trt_up = (size_t) floor((double)n_trt * (1-alpha / 2));
    size_t ctrl_low = (size_t) ceil((double)n_ctrl * alpha / 2);
    size_t ctrl_up = (size_t) floor((double)n_ctrl * (1-alpha / 2));
    size_t ind, idx, cnt_trt, cnt_ctrl;
    for (size_t j = 0; j < p; j++){
        cnt_trt = 0;
        cnt_ctrl = 0;
        for(size_t i = 0; i < N; i++){
            idx = Xorder_std[j][i];
            if (z_std[idx] == 0){
                cnt_ctrl += 1;
                if (cnt_ctrl == ctrl_low) X_range_ctrl[j][0] = *(Xpointer + j * n_y + idx);
            } else{
                cnt_trt += 1;
                if (cnt_trt == trt_low) X_range_trt[j][0] = *(Xpointer + j * n_y + idx);
            }
            if ((cnt_ctrl >= ctrl_low) & (cnt_trt >= trt_low)){
                break;
            }
        }

        cnt_trt = n_trt;
        cnt_ctrl = n_ctrl;
        for (size_t i = 0; i < N; i++){
            idx = Xorder_std[j][N-1-i];
            if (z_std[idx] == 0){
                if (cnt_ctrl == ctrl_up) X_range_ctrl[j][1] = *(Xpointer + j * n_y + idx);
                cnt_ctrl -= 1;
            } else{
                if (cnt_trt == trt_up) X_range_trt[j][1] = *(Xpointer + j * n_y + idx);
                cnt_trt -= 1;
            }
            if ((cnt_ctrl < ctrl_up) & (cnt_trt < trt_up)){
                break;
            }
        }

    }


    // Get overlap
    for (size_t i = 0; i < p; i++){
        X_range[i][0] = (X_range_trt[i][0] > X_range_ctrl[i][0]) ? X_range_trt[i][0] : X_range_ctrl[i][0];
        X_range[i][1] = (X_range_trt[i][1] < X_range_ctrl[i][1]) ? X_range_trt[i][1] : X_range_ctrl[i][1];
        if (X_range[i][0] >= X_range[i][1]) overlap = false;
        // cout << "X_range = " << X_range[i] << ", overlap = " << overlap << endl;
    }

    // if (!overlap){
    //     cout << "ntrt = " << n_trt << ", low = " << trt_low << ", up = " << trt_up << endl; 
    //     cout << "nctrl = " << n_ctrl << ", low = " << ctrl_low << ", up = " << ctrl_up << endl;
    //     cout << "X_range_trt = " << X_range_trt << endl;
    //     cout << "X_range_ctrl = " << X_range_ctrl << endl;
    // }

    return;
}

void get_rel_covariance(mat &cov, mat &X, std::vector<double> X_range, double theta, double tau)
{
    double temp;
    for (size_t i = 0; i < X.n_rows; i++){
        for (size_t j = i; j < X.n_rows; j++){
            // (tau*exp(-sum(theta * abs(x - y) / range)))
            temp = 0;
            for (size_t k = 0; k < X.n_cols; k++){
                temp += pow(X(i, k) - X(j, k), 2) / pow(X_range[k], 2);
                // temp += std::abs(X(i,k) - X(j, k)) / X_range[k];
            }
            // cout << "distance = " << temp << endl;
            cov(i, j) = tau * exp( - temp / pow(theta, 2) / 2);
            cov(j, i) = cov(i, j);
        }
    } 
    return;  
}

void count_overlap(const double *Xpointer, std::vector< std::vector<size_t> > &Xorder_std, std::vector<size_t> &z_std,
                size_t &p_continuous, size_t &n_y, size_t &n_min, size_t &N_overlap)
{
    N_overlap = 0;
    matrix<double> local_X_range;
    size_t N_Xorder = Xorder_std[0].size();
    bool overlap{true};
    get_overlap(Xpointer, Xorder_std, z_std, local_X_range, p_continuous, overlap);
    bool check_overlap;
    size_t N_trt = 0;
    size_t N_ctrl = 0;
    if (overlap){
        for (size_t i = 0; i < N_Xorder; i++){
            // check every dimension
            check_overlap = true;
            for (size_t j = 0; j < p_continuous; j++){
                if( (*(Xpointer + j * n_y + Xorder_std[0][i]) > local_X_range[j][1]) |  ( *(Xpointer + j * n_y + Xorder_std[0][i]) < local_X_range[j][0] ) ){
                    check_overlap = false;
                    break;
                }
            }
            if (check_overlap){
                if (z_std[Xorder_std[0][i]] == 0) {
                    N_ctrl += 1;
                }else {
                    N_trt += 1;
                }
            } 
        
            if ((N_trt > n_min) & (N_ctrl > n_min)) break;
        }
        N_overlap = N_trt < N_ctrl ? N_trt : N_ctrl;
    }
    // cout << "overlap = " << overlap << ", N = " << N_Xorder << ", N_overlap = " << N_overlap << ", range = " << local_X_range << endl;
}