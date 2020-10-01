%module(package = "xbcf", moduleimport = "import _xbcf_cpp_") xbcf_cpp_

%{
/* inserts a macro that specifies that the resulting
C++ file should be built as a python extension */

#define SWIG_FILE_WITH_INIT
#include "xbcf.h"
%}

/* JSON */
%include<std_string.i>

/* Numpy */
%include "numpy.i" 
%init %{
  import_array();
%}

%apply(int DIM1, double *IN_ARRAY1){(int n, double *a)};
%apply(int DIM1, double *ARGOUT_ARRAY1){(int size, double *arr)};
%apply(int DIM1, int DIM2, double *IN_ARRAY2){(int n_t, int d_t, double *a_t)};
%apply(int DIM1, double *IN_ARRAY1){(int n_y, double *a_y)};
// need to add one more parameter to take care of z (tried below) ...
%apply(int DIM1, int *IN_ARRAY1){(int n_z, int *a_z)};
// as well as of X_pr with additional column for the propensity scores (may be unnecessary though -- check with Saar)
%apply(int DIM1, int DIM2, double *IN_ARRAY2){(int n_p, int d_p, double *a_p)};

%include "xbcf.h" // Include code for a static version of Python
