from __future__ import absolute_import

from .xbcf_cpp_ import XBCFcpp
import collections
from collections import OrderedDict
import numpy as np
import json

## Optional Import Pandas ##
try:
    from pandas import DataFrame
    from pandas import Series
except ImportError:

    class DataFrame(object):
        pass

    class Series(object):
        pass


"""
// TODO: 1. remove unnecessary functions
//			 2. try to ignore json -- comment it out (ask Saar about it if seems like an integrtal part of the code)
// 			 3. - rewrite __init__ function
//			 4. - review logic of the checkers and updaters (add to the list if need to rewrite)
//			 5. - rewrite __convert_params_check_types
//			 6. - rewrite fit
//           7. - fix description of XBCF object
//           8. update object description
//           9. return the json functions
"""


class XBCF(object):
    """
	Python extension for Accelerated Bayesian Additive Regression Trees
    Parameters
    ----------
	num_trees : int
        Number of trees in each iteration.
	num_sweeps : int
        Number of sweeps (MCMC draws).
	n_min: int
		Minimum number of samples in each final node.
	num_cutpoints: int
		For continuous variable, number of adaptive cutpoint candidates
		considered in each split .
	alpha: double
		Tree prior hyperparameter : alpha * (1 + depth) ^ beta.
	beta: double
		Tree prior hyperparameter : alpha * (1 + depth) ^ beta.
	tau: double / "auto"
		Prior for leaf mean variance : mu_j ~ N(0,tau)
	burnin: int
		Number of sweeps used to burn in - not used for prediction.
	max_depth_num: int
		Represents the maximum size of each tree - size is usually determined via tree prior.
		Use this only when wanting to determnistically cap the size of each tree.
	mtry: int / "auto"
		Number of variables considered at each split - like random forest.
	kap: double
		Prior for sigma :  sigma^2 | residaul ~  1/ sqrt(G)
		where G ~ gamma( (num_samples + kap) / 2, 2/(sum(residual^2) + s) )
	s: double
		Prior for sigma :  sigma^2 | residaul ~  1/ sqrt(G)
		where G ~ gamma( (num_samples + kap) / 2, 2/(sum(residual^2) + s) )
	verbose: bool
		Print the progress
	parallel: bool
		Do computation in parallel
	seed: int
		Random seed, should be a positive integer
	model: str
		"Normal": Regression problems
				: Classification problems (encode Y \in{ -1,1})
		"Multinomial" : Classes encoded as integers
		"Probit": Classification problems (encode Y \in{ -1,1})

	no_split_penality: double
		Weight of no-split option. The default value in the normal model is log(num_cutpoints).
		Values should be considered in log scale.
	sample_weights_flag: bool (True)
		To sample weights according to Dirchlet distribution
	num_classes: int (1)
		Number of classes

	"""

    def __init__(
        self,
        num_sweeps: int = 40,
        burnin: int = 15,
        max_depth: int = 250,
        Nmin: int = 1,
        num_cutpoints: int = 100,
        no_split_penality="auto",
        mtry_pr: int = 0,
        mtry_trt: int = 0,
        p_categorical_pr: int = 0,
        p_categorical_trt: int = 0,
        num_trees_pr: int = 100,
        alpha_pr: float = 0.95,
        beta_pr: float = 1.25,
        tau_pr: float = 0.0,
        kap_pr: float = 16.0,
        s_pr: float = 4.0,
        pr_scale: bool = False,
        num_trees_trt: int = 100,
        alpha_trt: float = 0.25,
        beta_trt: float = 3.0,
        tau_trt: float = 0.0,
        kap_trt: float = 16.0,
        s_trt: float = 4.0,
        trt_scale: bool = False,
        verbose: bool = False,
        parallel: bool = False,
        set_random_seed: bool = False,
        random_seed: int = 0,
        sample_weights_flag: bool = True,
        a_scaling: bool = True,
        b_scaling: bool = True,
    ):
        assert num_sweeps > burnin, "num_sweep must be greater than burnin"
        self.params = OrderedDict(
            [
                ("num_sweeps", num_sweeps),
                ("burnin", burnin),
                ("max_depth", max_depth),
                ("Nmin", Nmin),
                ("num_cutpoints", num_cutpoints),
                ("no_split_penality", no_split_penality),
                ("mtry_pr", mtry_pr),
                ("mtry_trt", mtry_trt),
                ("p_categorical_pr", p_categorical_pr),
                ("p_categorical_trt", p_categorical_trt),
                ("num_trees_pr", num_trees_pr),
                ("alpha_pr", alpha_pr),
                ("beta_pr", beta_pr),
                ("tau_pr", tau_pr),
                ("kap_pr", kap_pr),
                ("s_pr", s_pr),
                ("pr_scale", pr_scale),
                ("num_trees_trt", num_trees_trt),
                ("alpha_trt", alpha_trt),
                ("beta_trt", beta_trt),
                ("tau_trt", tau_trt),
                ("kap_trt", kap_trt),
                ("s_trt", s_trt),
                ("trt_scale", trt_scale),
                ("verbose", verbose),
                ("parallel", parallel),
                ("set_random_seed", set_random_seed),
                ("random_seed", random_seed),
                ("sample_weights_flag", sample_weights_flag),
                ("a_scaling", a_scaling),
                ("b_scaling", b_scaling),
            ]
        )
        self.__convert_params_check_types(**self.params)
        self._xbcf_cpp = None

        # Additional Members
        # self.importance = None
        self.sigma_draws = None
        self.is_fit = False

    def __repr__(self):
        items = ("%s = %r" % (k, v) for k, v in self.params.items())
        return str(self.__class__.__name__) + "(" + str((", ".join(items))) + ")"

    def __add_columns(self, x_t, x):
        """
		Keep columns internally
		"""
        if isinstance(x_t, DataFrame):
            self.columns_trt = x_t.columns
        else:
            self.columns_trt = range(x_t.shape[1])
        self.num_columns_trt = len(self.columns_trt)

        if isinstance(x, DataFrame):
            self.columns_pr = x.columns
        else:
            self.columns_pr = range(x.shape[1])
        self.num_columns_pr = len(self.columns_pr)

    def __update_fit(
        self, x_t, fit_x_t, x=None, fit_x=None, y=None, fit_y=None, z=None, fit_z=None
    ):
        """
		Convert DataFrame to numpy
		"""
        if isinstance(x_t, DataFrame):
            fit_x_t = x_t.values
        if x is not None:
            if isinstance(x, DataFrame):
                fit_x = x.values
        if y is not None:
            if isinstance(y, Series):
                fit_y = y.values
        if z is not None:
            if isinstance(z, Series):
                fit_z = z.values

    def __check_input_type(self, x_t, x=None, y=None, z=None):
        """
		Dimension check
		"""

        if not isinstance(x_t, (np.ndarray, DataFrame)):
            raise TypeError("x_t must be numpy array or pandas DataFrame")

        if np.any(np.isnan(x_t)) or np.any(~np.isfinite(x_t)):
            raise TypeError("Cannot have missing values!")

        if x is not None:
            if not isinstance(x, (np.ndarray, DataFrame)):
                raise TypeError("x must be numpy array or pandas DataFrame")

            if np.any(np.isnan(x)) or np.any(~np.isfinite(x)):
                raise TypeError("Cannot have missing values!")

        if y is not None:
            if not isinstance(y, (np.ndarray, Series)):
                raise TypeError("y must be numpy array or pandas Series")

            if np.any(np.isnan(y)):
                raise TypeError("Cannot have missing values!")

            assert x.shape[0] == y.shape[0], "X and y must be the same length"

        # add checks for z
        # assert all(z >= 0) and all(
        #    z.astype(int) == z
        # ), "z must be a positive integer"

    def __check_test_shape(self, x):
        assert x.shape[1] == self.num_columns_trt, "Mismatch on number of columns"

    def __check_params(self, p_cat):
        assert p_cat <= self.num_columns_pr, "p_cat must be <= number of columns"
        assert (
            self.params["mtry_pr"] <= self.num_columns_pr
        ), "mtry must be <= number of columns"

    def __update_mtry_tau_penality(self, x_t, x):
        """
		Handle mtry, tau, and no_split_penality defaults
		"""
        if self.params["mtry_trt"] == "auto":
            self.params["mtry_trt"] = self.num_columns_trt
        if self.params["tau_trt"] == "auto":
            self.params["tau_trt"] = 0.0

        if self.params["mtry_pr"] == "auto":
            self.params["mtry_pr"] = self.num_columns_pr
        if self.params["tau_pr"] == "auto":
            self.params["tau_pr"] = float(1 / self.params["num_trees_pr"])

        from math import log

        if self.params["no_split_penality"] == "auto":
            self.params["no_split_penality"] = log(self.params["num_cutpoints"])

    def __convert_params_check_types(self, **params):
        """
		This function converts params to list and handles type conversions
		If a wrong type is provided function raises exceptions
		"""
        import warnings
        from collections import OrderedDict

        DEFAULT_PARAMS = OrderedDict(
            [
                ("num_sweeps", 40),
                ("burnin", 15),
                ("max_depth", 250),
                ("Nmin", 1),
                ("num_cutpoints", 100),
                ("no_split_penality", 0.0),
                ("mtry_pr", 0),
                ("mtry_trt", 0),
                ("p_categorical_pr", 0),
                ("p_categorical_trt", 0),
                ("num_trees_pr", 100),
                ("alpha_pr", 0.95),
                ("beta_pr", 1.25),
                ("tau_pr", 0.0),
                ("kap_pr", 16.0),
                ("s_pr", 4.0),
                ("pr_scale", False),
                ("num_trees_trt", 100),
                ("alpha_trt", 0.25),
                ("beta_trt", 3.0),
                ("tau_trt", 0.0),
                ("kap_trt", 16.0),
                ("s_trt", 4.0),
                ("trt_scale", False),
                ("verbose", False),
                ("parallel", False),
                ("set_random_seed", False),
                ("random_seed", 0),
                ("sample_weights_flag", True),
                ("a_scaling", True),
                ("b_scaling", True),
            ]
        )

        DEFAULT_PARAMS_ = OrderedDict(
            [
                ("num_sweeps", int),
                ("burnin", int),
                ("max_depth", int),
                ("Nmin", int),
                ("num_cutpoints", int),
                ("no_split_penality", float),
                ("mtry_pr", int),
                ("mtry_trt", int),
                ("p_categorical_pr", int),
                ("p_categorical_trt", int),
                ("num_trees_pr", int),
                ("alpha_pr", float),
                ("beta_pr", float),
                ("tau_pr", float),
                ("kap_pr", float),
                ("s_pr", float),
                ("pr_scale", bool),
                ("num_trees_trt", int),
                ("alpha_trt", float),
                ("beta_trt", float),
                ("tau_trt", float),
                ("kap_trt", float),
                ("s_trt", float),
                ("trt_scale", bool),
                ("verbose", bool),
                ("parallel", bool),
                ("set_random_seed", bool),
                ("random_seed", int),
                ("sample_weights_flag", bool),
                ("a_scaling", bool),
                ("b_scaling", bool),
            ]
        )

        for param, type_class in DEFAULT_PARAMS_.items():
            default_value = DEFAULT_PARAMS[param]
            new_value = params.get(param, default_value)

            if (param in ["mtry", "tau", "no_split_penality"]) and new_value == "auto":
                continue

            try:
                self.params[param] = type_class(new_value)
            except:
                raise TypeError(
                    str(param) + " should conform to type " + str(type_class)
                )

    def fit(self, x_t, x, y, z, p_cat=0):
        """
		Fit XBART model
        Parameters
        ----------
		x : DataFrame or numpy array
            Feature matrix (predictors)
        y : array_like
            Target (response)
		p_cat: int
			Number of features to treat as categorical for cutpoint options. More efficient.
			To use this feature set place the categorical features as the last p_cat columns of x
		"""

        # Check inputs #
        self.__check_input_type(x_t, x, y, z)
        # add checks for p_cat -> store columns internally will be needed then
        self.__add_columns(x_t, x)
        fit_x_t = x_t
        fit_x = x
        fit_y = y
        fit_z = z

        # Update Values #
        self.__update_fit(x_t, fit_x_t, x, fit_x, y, fit_y, z, fit_z)
        self.__update_mtry_tau_penality(fit_x_t, fit_x)
        # self.__check_params(p_cat)

        # Create xbart_cpp object #
        if self._xbcf_cpp is None:
            # self.args = self.__convert_params_check_types(**self.params)
            args = list(self.params.values())
            self._xbcf_cpp = XBCFcpp(*args)  # Makes C++ object
        # print(type(fit_x_t[0][0]))
        # print(type(fit_x[0][0]))
        # print(type(fit_y[0]))
        # print(type(fit_z[0]))
        # fit #
        self._xbcf_cpp._fit(fit_x_t, fit_x, fit_y, fit_z, p_cat)

        muhats = self._xbcf_cpp.get_muhats(self.params["num_sweeps"] * fit_x.shape[0])
        tauhats = self._xbcf_cpp.get_tauhats(
            self.params["num_sweeps"] * fit_x_t.shape[0]
        )
        b = self._xbcf_cpp.get_b(self.params["num_sweeps"] * 2)
        a = self._xbcf_cpp.get_a(self.params["num_sweeps"] * 1)

        # b1 = self._xbart_cpp.get_bs(self.params["num_sweeps"]*fit_x.shape[0], 0)
        # b2 = self._xbart_cpp.get_bs(self.params["num_sweeps"]*fit_x.shape[0], 1)
        # a = self._xbart_cpp.get_a(self.params["num_sweeps"]*fit_x_t.shape[0])
        # Convert from colum major
        self.muhats = muhats.reshape(
            (fit_x.shape[0], self.params["num_sweeps"]), order="C"
        )
        self.tauhats = tauhats.reshape(
            (fit_x_t.shape[0], self.params["num_sweeps"]), order="C"
        )
        self.b = b.reshape((self.params["num_sweeps"]), 2, order="C")
        self.a = a.reshape((self.params["num_sweeps"]), 1, order="C")

        # Additionaly Members
        # self.importance = self._xbart_cpp._get_importance(fit_x.shape[1])
        # self.importance = dict(zip(self.columns, self.importance.astype(int)))

        # if self.model == "Normal":
        #    self.sigma_draws = self._xbart_cpp.get_sigma_draw(
        #        self.params["num_sweeps"] * self.params["num_trees"]
        #    )
        # Convert from colum major
        #    self.sigma_draws = self.sigma_draws.reshape(
        #        (self.params["num_sweeps"], self.params["num_trees"]), order="F"
        #    )

        self.is_fit = True
        return self

    def predict(self, x_test, return_mean=True):

        assert self.is_fit, "Must run fit before running predict"

        # Check inputs #

        self.__check_input_type(x_test)
        pred_x = x_test.copy()
        self.__check_test_shape(pred_x)
        self.__update_fit(x_test, pred_x)  # unnecessary in general?

        # Run Predict
        self._xbcf_cpp._predict(pred_x)
        # Convert to numpy
        tauhats_test = self._xbcf_cpp.get_tauhats_test(
            self.params["num_sweeps"] * pred_x.shape[0]
        )

        # Convert from colum major
        self.tauhats_test = tauhats_test.reshape(
            (pred_x.shape[0], self.params["num_sweeps"]), order="C"
        )
        # Compute mean
        # get bs and compute mean here?
        # self.yhats_mean =  self.yhats_test[:,self.params["burnin"]:].mean(axis=1)

        # if return_mean:
        # 	return self.yhats_mean
        # else:
        return self.tauhats_test

    def get_params(self):
        return self.params
