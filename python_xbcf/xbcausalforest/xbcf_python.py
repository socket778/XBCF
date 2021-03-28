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



# todo: 1. remove unnecessary functions
#	    2. update json function -- ask Saar if needed
#       8. update object description


class XBCF(object):
    """
    Model for fitting Accelerated Bayesian Causal Forest

    Parameters
    ----------
    num_sweeps : int
        Number of sweeps.
    burnin: int
        Number of sweeps used to burn in - not used for prediction.
    max_depth: int
        Represents the maximum size of each tree - size is usually determined via tree prior.
        Use this only when wanting to determnistically cap the size of each tree.
    Nmin: int
        Minimum number of samples in each final node.
    num_cutpoints: int
        For continuous variable, number of adaptive cutpoint candidates
        considered in each split.
    no_split_penality: double
        Weight of no-split option. The default value in the normal model is log(num_cutpoints).
        Values should be considered in log scale.
    mtry_pr: int
        Number of variables considered at each split  - like random forest (prognostic forest).
    mtry_trt: int
        Number of variables considered at each split  - like random forest (treatment forest).
    p_categorical_pr: int
        Number of categorical variable (data used for prognostic forest).
    p_categorical_trt: int
        Number of categorical variable (data used for treatment forest).
    num_trees_pr: int
        Number of trees in each iteration (prognostic forest).
    alpha_pr: double
        Tree prior hyperparameter (prognostic forest) : alpha_pr * (1 + depth) ^ beta_pr.
    beta_pr: double
        Tree prior hyperparameter (prognostic forest) : alpha_pr * (1 + depth) ^ beta_pr.
    tau_pr: double
        Prior for leaf mean variance (prognostic forest) : mu_j ~ N(0,tau_pr)
    kap_pr: double
        Prior for sigma_0 (control group):  sigma_0^2 | residual ~  1/ sqrt(G)
        where G ~ gamma( (num_samples_0 + kap_pr) / 2, 2/(sum(residual^2) + s_pr) )
    s_pr: double
        Prior for sigma_0 (control group):  sigma_0^2 | residual ~  1/ sqrt(G)
        where G ~ gamma( (num_samples_0 + kap_pr) / 2, 2/(sum(residual^2) + s_pr) )
    num_trees_trt: int
        Number of trees in each iteration (treatment forest).
    alpha_trt: double
        Tree prior hyperparameter (treatment forest) : alpha_trt * (1 + depth) ^ beta_trt.
    beta_trt: double
        Tree prior hyperparameter (treatment forest) : alpha_trt * (1 + depth) ^ beta_trt.
    tau_trt: double
        Prior for leaf mean variance (treatment forest) : mu_j ~ N(0,tau_trt)
    kap_trt: double
        Prior for sigma_1 (moderated group):  sigma_1^2 | residual ~  1/ sqrt(G)
        where G ~ gamma( (num_samples_1 + kap_trt) / 2, 2/(sum(residual^2) + s_trt) )
    s_trt: double
        Prior for sigma_1 (moderated group):  sigma_1^2 | residual ~  1/ sqrt(G)
        where G ~ gamma( (num_samples_1 + kap_trt) / 2, 2/(sum(residual^2) + s_trt) )
    verbose: bool
        Print the progress
    parallel: bool (True)
        Do computation in parallel
    seed: int
        Random seed, should be a positive integer
    sample_weights_flag: bool (True)
        To sample weights according to Dirchlet distribution
    a_scaling: bool (True)
        Use scaling model parameter (a) for the prognostic term.
    b_scaling: bool (True)
        Use scaling model parameters (b_0, b_1) for the treatment term.
    standardize_target: bool (True)
        Standardize the target variable.
    """

    def __init__(
        self,
        num_sweeps: int = 40,
        burnin: int = 15,
        max_depth: int = 250,
        Nmin: int = 1,
        num_cutpoints: int = 100,
        no_split_penality: int = -1,
        mtry_pr: int = 0,
        mtry_trt: int = 0,
        p_categorical_pr = None,
        p_categorical_trt = None,
        num_trees_pr: int = 30,
        alpha_pr: float = 0.95,
        beta_pr: float = 1.25,
        tau_pr: int = -1,
        kap_pr: float = 16.0,
        s_pr: float = 4.0,
        pr_scale: bool = False,
        num_trees_trt: int = 10,
        alpha_trt: float = 0.25,
        beta_trt: float = 3.0,
        tau_trt: int = -1,
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
        standardize_target: bool = True,
    ):
        assert num_sweeps > burnin, "num_sweep must be greater than burnin"
        assert p_categorical_pr is not None, "p_categorical_pr must be provided as an input"
        assert p_categorical_trt is not None, "p_categorical_trt must be provided as an input"
        assert p_categorical_pr >= 0, "p_categorical_pr must be a non-negative integer"
        assert p_categorical_trt >= 0, "p_categorical_trt must be a non-negative integer"
        if not isinstance(p_categorical_pr, int):
            raise TypeError("p_categorical_pr must be integer")
        if not isinstance(p_categorical_trt, int):
            raise TypeError("p_categorical_trt must be integer")
        if not isinstance(mtry_pr, int):
            raise TypeError("mtry_pr must be integer")
        if not isinstance(mtry_trt, int):
            raise TypeError("mtry_trt must be integer")
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
        self.standardize_target=standardize_target

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

    def __check_test_shape(self, x, term):
        if term == 1:
            assert x.shape[1] == self.num_columns_trt, "Mismatch on number of columns for tau input matrix"
        else:
            assert x.shape[1] == self.num_columns_pr, "Mismatch on number of columns for mu input matrix"

    def __check_params(self):
        assert (self.params["p_categorical_pr"] <= self.num_columns_pr), "p_categorical_pr must be <= number of columns"
        assert (self.params["p_categorical_trt"] <= self.num_columns_trt), "p_categorical_trt must be <= number of columns"
        assert (
            self.params["mtry_pr"] <= self.num_columns_pr
        ), "mtry_pr must be <= number of columns"
        assert (
            self.params["mtry_trt"] <= self.num_columns_trt
        ), "mtry_trt must be <= number of columns"

    def __update_mtry_tau_penality(self, x_t, x, y):
        """
		Handle mtry, tau, and no_split_penality defaults
		"""
        if self.params["mtry_trt"] <= 0:
            self.params["mtry_trt"] = self.num_columns_trt
        if self.params["tau_trt"] <= 0:
            self.params["tau_trt"] = 0.1 * np.var(y) / self.params["num_trees_trt"]

        if self.params["mtry_pr"] <= 0:
            self.params["mtry_pr"] = self.num_columns_pr
        if self.params["tau_pr"] <= 0:
            self.params["tau_pr"] = 0.6 * np.var(y) / self.params["num_trees_pr"]

        from math import log

        if self.params["no_split_penality"] < 0:
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
                ("no_split_penality", -1),
                ("mtry_pr", -1),
                ("mtry_trt", -1),
                ("p_categorical_pr", -1),
                ("p_categorical_trt", -1),
                ("num_trees_pr", 30),
                ("alpha_pr", 0.95),
                ("beta_pr", 1.25),
                ("tau_pr", -1),
                ("kap_pr", 16.0),
                ("s_pr", 4.0),
                ("pr_scale", False),
                ("num_trees_trt", 10),
                ("alpha_trt", 0.25),
                ("beta_trt", 3.0),
                ("tau_trt", -1),
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
                ("no_split_penality", int),
                ("mtry_pr", int),
                ("mtry_trt", int),
                ("p_categorical_pr", int),
                ("p_categorical_trt", int),
                ("num_trees_pr", int),
                ("alpha_pr", float),
                ("beta_pr", float),
                ("tau_pr", int),
                ("kap_pr", float),
                ("s_pr", float),
                ("pr_scale", bool),
                ("num_trees_trt", int),
                ("alpha_trt", float),
                ("beta_trt", float),
                ("tau_trt", int),
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

    def fit(self, x_t, x, y, z):
        """
        Fit XBCF model

        Parameters
        ----------
        x_t : DataFrame or numpy array
            Feature matrix (predictors), treatment term
        x : DataFrame or numpy array
            Feature matrix (predictors), prognostic term
        y : array_like
            Target (response)
        z : array_like (binary)
            Treatment assignment

        Returns
        -------
        A fit object, which contains unscaled and scaled draws for mu and tau.
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
        self.__update_mtry_tau_penality(fit_x_t, fit_x, fit_y)
        # self.__check_params(p_cat)

        # Standardize target variable
        if self.standardize_target:
            self.meany_ = np.mean(fit_y)
            self.sdy_ = np.std(fit_y)
            if self.sdy_ == 0:
                ValueError("Target variable is constant with variance 0.")
            #elif self.sdy_ == 1:
            #    fit_y = (fit_y - self.meany_)
            else:
                fit_y = (fit_y - self.meany_)/self.sdy_

        # Create xbcf_cpp object #
        if self._xbcf_cpp is None:
            # self.args = self.__convert_params_check_types(**self.params)
            args = list(self.params.values())
            self._xbcf_cpp = XBCFcpp(*args)  # Makes C++ object
        # print(type(fit_x_t[0][0]))
        # print(type(fit_x[0][0]))
        # print(type(fit_y[0]))
        # print(type(fit_z[0]))
        # fit #
        self._xbcf_cpp._fit(fit_x_t, fit_x, fit_y, fit_z)

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

        # Unstandardize
        if self.standardize_target:
            a = self.a.transpose()
            b = self.b.transpose()

            self.tauhats = self.sdy_ * self.tauhats
            self.muhats = self.sdy_ * self.muhats

            self.muhats_adjusted = (self.muhats * a) + self.meany_
            self.tauhats_adjusted = self.tauhats * (b[1] - b[0])

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

    def predict(self, X, X1=None, return_mean=True, return_muhat=False):
        """Estimate tau for data X

        Parameters
        ----------
        X : DataFrame or numpy array
            Feature matrix (predictors); features match the treatment input matrix used when fitting the model.
        X1 : DataFrame or numpy array
            Feature matrix (predictors); features match the prognostic input matrix used when fitting the model.
        return_mean : bool
            Return mean of samples excluding burn-in as point estimate, default: True
        return_muhat : bool
            Also return mu hat, the estimated outcome without the treatment effect, default: False

        Returns
        -------
        array or list of two arrays (if return_muhat=True)
            Estimated tau or estimated tau and estimated mu (of return_muhat=True)
        """

        assert self.is_fit, "Must run fit before running predict"

        # Check inputs #

        self.__check_input_type(X)
        pred_x = X.copy()
        self.__check_test_shape(pred_x, 1) # 1 = check treatment input
        self.__update_fit(X, pred_x)  # unnecessary in general?

        if return_muhat is True:
            assert X1 is not None, "X1, input matrix for mu, must be provided as a separate input."

        # Run Predict (1 = treatment forest is used)
        self._xbcf_cpp._predict(pred_x, 1)
        # Convert to numpy
        tauhats_test = self._xbcf_cpp.get_tauhats_test(
            self.params["num_sweeps"] * pred_x.shape[0]
        )

        # Convert from colum major
        tauhats_test = tauhats_test.reshape(
            (pred_x.shape[0], self.params["num_sweeps"]), order="C"
        )

        b = self.b.transpose()
        thats =  tauhats_test* (b[1] - b[0])

        # Unstandardize prediction
        if self.standardize_target:
            thats = self.sdy_ * thats

        # Point-estimate from samples
        if return_mean:
            thats = np.mean(thats[:, self.params['burnin']:], axis=1)

        if return_muhat is False: # Return only treatment estimate
            return thats
        else: # also calculate and return estimate for mu

            pred_x = X1.copy()
            self.__check_test_shape(pred_x, 0) # 0 = check prognostic input

            # Run Predict (0 = prognostic forest is used)
            self._xbcf_cpp._predict(pred_x, 0)

            # Convert to numpy
            muhats = self._xbcf_cpp.get_muhats_test(self.params["num_sweeps"] * pred_x.shape[0])
            # Convert from colum major
            muhats = muhats.reshape(
                (pred_x.shape[0], self.params["num_sweeps"]), order="C"
            )
            a = self.a.transpose()

            muhats = muhats * a
            if self.standardize_target:
                muhats = (muhats * self.sdy_) + self.meany_

            if return_mean:
                muhats = np.mean(muhats[:, self.params['burnin']:], axis=1)

            return thats, muhats


    def getParams(self):
        """
        Returns
        -------
        A list of model parametrs.
        """
        return self.params

    def getTau(self):
        """
        Get point-estimates of tau from the fitted values.

        Returns
        -------
        Array of individual-level tau estimates.
        """
        tauhats = np.mean(self.tauhats_adjusted[:, self.params["burnin"]:], axis=1)
        return tauhats