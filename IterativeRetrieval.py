import os
import pyrttov
import warnings
import sys
import pdb
import h5py
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy import interpolate
import multiprocessing
import subprocess
import healpy as hp
import pandas as pn
import xarray as xr
import scipy.stats as stats
import pyOptimalEstimation as pyOE
from datetime import datetime
import glob
import pandas as pd
#--------------------------------------------------
Reference       = {}
Truth           = {}
ATMS            = {}
MERRA2          = {}
forwardKwArgs   = {}
Errorprior      = True
ErrorRTTOV      = False

def splitTQ(x):
    t_index = [i for i in x.index if i.endswith('t')]
    q_index = [i for i in x.index if i.endswith('q')]
    h_index = [float(i.split('_')[0]) for i in x.index if i.endswith('q')]

    assert len(t_index) == len(q_index)
    assert len(t_index) == len(h_index)
    assert len(t_index)*2 == len(x)

    xt = x[t_index]
    xt.index = h_index

    xq = x[q_index]
    xq.index = h_index

    xt.index.name = 'height'
    xq.index.name = 'height'

    return xt, xq
# =================================================================================================================
class optimalEstimation2(object):
    r'''
    The core optimalEstimation class, which contains all required parameters.
    See [1]_ for an extensive introduction into Optimal Estimation theory, 
    [2]_ discusses this library

    Parameters
    ----------
    x_vars : list of str
        names of the elements of state vector x.
    x_a : pd.Series or list or np.ndarray
        prior information of state x.
    S_a : pd.DataFrame or list or np.ndarray
        covariance matrix of state x.
    y_vars : list of str
        names of the elements of state vector x
    y_obs : pd.Series or list or np.ndarray
        observed measurement vector y.
    S_y : pd.DataFrame or list or np.ndarray
        covariance matrix of measurement y. If there is no b vector, S_y
        is sequal to S_e
    forward : function
        forward model expected as ``forward(xb,**forwardKwArgs): return y``
        with xb = pd.concat((x,b)).
    userJacobian : function, optional
        For forwarld models that can calculate the Jacobian internally (e.g.
        RTTOV), a call to estiamte the Jacobian can be added. Otherwise, the 
        Jacobian is estimated by pyOEusing the standard 'forward' call. The 
        fucntion is expected as ``self.userJacobian(xb, self.perturbation, \
        self.y_vars, **self.forwardKwArgs): return jacobian``
        with xb = pd.concat((x,b)). Defaults to None
    x_truth : pd.Series or list or np.ndarray, optional
        If truth of state x is known, it can added to the data object. If
        provided, the value will be used for the routines linearityTest and
        plotIterations, but _not_ by the retrieval itself. Defaults to None/
    b_vars : list of str, optional
        names of the elements of parameter vector b. Defaults to [].
    b_p : pd.Series or list or np.ndarray.
        parameter vector b.  defaults to []. Note that defining b_p makes
        only sence if S_b != 0. Otherwise it is easier (and cheaper) to
        hardcode b into the forward operator.
    S_b : pd.DataFrame or list or np.ndarray
        covariance matrix of parameter b. Defaults to [[]].
    forwardKwArgs : dict,optional
        additional keyword arguments for ``forward`` function.
    multipleForwardKwArgs : dict,optional
        additional keyword arguments for forward function in case multiple 
        profiles should be provided to the forward operator at once. If not .
        defined, ``forwardKwArgs`` is used instead and ``forward`` is called 
        for every profile separately
    x_lowerLimit : dict, optional
        reset state vector x[key] to x_lowerLimit[key] in case x_lowerLimit is
        undercut. Defaults to {}.
    x_upperLimit : dict, optional
        reset state vector x[key] to x_upperLimit[key] in case x_upperLimit is
        exceeded. Defaults to {}.
    perturbation : float or dict of floats, optional
        relative perturbation of statet vector x to estimate the Jacobian. Can
        be specified for every element of x seperately. Defaults to 0.1 of
        prior.
    disturbance : float or dict of floats, optional
        DEPRECATED: Identical to ``perturbation`` option. If both option are 
        provided, ``perturbation``  is used instead. 
    useFactorInJac : bool,optional
        True if disturbance should be applied by multiplication, False if it
        should by applied by addition of fraction of prior. Defaults to False.
    gammaFactor : list of floats, optional
        Use additional gamma parameter for retrieval, see [3]_.
    convergenceTest : {'x', 'y', 'auto'}, optional
        Apply convergence test in x or y-space. If 'auto' is 
        selected, the test will be done in x-space if len(x) <= len(y) and in 
        y-space otherwise. Experience shows that in both cases convergence is 
        faster in x-space without impacting retrieval quality. Defaults to 'x'.
    convergenceFactor : int, optional
        Factor by which the convergence criterion needs to be smaller than
        len(x) or len(y) 
    verbose: bool, optional
        True or not present: iteration, residual, etc. printed to screen during
        normal operation. If False, it will turn off such notifications.     

    Attributes
    ----------
    converged : boolean
      True if retriveal converged successfully
    x_op : pd.Series
      optimal state given the observations, i.e. retrieval solution
    y_op : pd.Series
      Optimal y, i.e. observation associated with retrieval solution
    S_op : pd.DataFrame
      covariance of x_op, i.e. solution uncertainty
    x_op_err : pd.Series
      1 sigma errors of x_op. derived with sqrt(diag(S_op))
    convI : int
      iteration where convergence was achieved
    K_i : list of pd.DataFrame
      list of Jacobians for iteration i.
    x_i : list of pd.Series
      iterations of state vector x
    y_i : list of pd.Series
      iterations of measurement vector y
    dgf_i : list of float
      degrees of freedom for each iteration
    A_i  : list of pd.DataFrame
      Averaging kernel for each iteration
    d_i2 : list of float
      convergence criteria for each iteration
    S_aposteriori_i : list of pd.DataFrame
      a posteriori covariance matrix of x for each iteration
    gam_i : list of floats
      gamma parameters used in retrievals, see also `gammaFactor` and  [1]_.
    dgf : float
      total degrees of freedom for signal of the retrieval solution
    dgf_x : pd.Series
      degrees of freedom for signal per state variable


    Returns
    -------

    pyOptimalEstimation object
      returns the pyOptimalEstimation object


    References
    ----------
    .. [1] Rodgers, C. D., 2000: Inverse Methods for Atmospheric Sounding:
    Theory and Practice. World Scientific Publishing Company, 240 pp. 
    https://library.wmo.int/index.php?lvl=notice_display&id=12279.

    .. [2] Maahn, M., D. D. Turner, U. Löhnert, D. J. Posselt, K. Ebell, G. 
    G. Mace, and J. M. Comstock, 2020: Optimal Estimation Retrievals and Their
    Uncertainties: What Every Atmospheric Scientist Should Know. Bull. Amer. 
    Meteor. Soc., 101, E1512–E1523, https://doi.org/10.1175/BAMS-D-19-0027.1.

    .. [3] Turner, D. D., and U. Löhnert, 2014: Information Content and
    Uncertainties in Thermodynamic Profiles and Liquid Cloud Properties
    Retrieved from the Ground-Based Atmospheric Emitted Radiance
    Interferometer (AERI). Journal of Applied Meteorology & Climatology, 53,
    752–771, doi:10.1175/JAMC-D-13-0126.1.

    '''

    def __init__(self,
                 x_vars,
                 x_a,
                 S_a,
                 y_vars,
                 y_obs,
                 S_y,
                 forward,
                 userJacobian=None,
                 x_truth=None,
                 b_vars=[],
                 b_p=[],
                 S_b=[[]],
                 x_lowerLimit={},
                 x_upperLimit={},
                 useFactorInJac=False,
                 gammaFactor=None,
                 perturbation=0.1,
                 disturbance=None,
                 convergenceFactor=10,
                 convergenceTest='x',
                 forwardKwArgs={},
                 multipleForwardKwArgs=None,
                 verbose=None
                 ):

        # some initital tests
        assert np.linalg.matrix_rank(S_a) == S_a.shape[-1],\
            'S_a must not be singular'
        assert np.linalg.matrix_rank(S_y) == S_y.shape[-1],\
            'S_y must not be singular'
        for inVar in [x_a, S_a, S_y, y_obs]:
            assert not np.any(np.isnan(inVar))

        self.x_vars = list(x_vars)
        self.x_a = pd.Series(x_a, index=self.x_vars)
        self.S_a = pd.DataFrame(
            S_a, index=self.x_vars, columns=self.x_vars)
        self.x_a_err = np.sqrt(
            pd.Series(np.diag(self.S_a), index=self.x_vars)
        )
        self.x_n = len(self.x_vars)
        self.y_vars = list(y_vars)
        self.S_y = pd.DataFrame(
            S_y, index=self.y_vars, columns=self.y_vars)
        self.y_obs = pd.Series(y_obs, index=self.y_vars)
        self.y_n = len(self.y_vars)
        self.forward = forward
        self.userJacobian = userJacobian
        self.x_truth = pd.Series(x_truth, index=self.x_vars, dtype=np.float64)

        # We want to save at least the name because the forward function
        # is removed for saving
        try:
            self.forward_name = forward.__name__
        except AttributeError:
            self.forward_name = None
        try:
            self.userJacobian_name = userJacobian.__name__  
        except AttributeError:
            self.userJacobian_name = None

        self.b_vars = list(b_vars)
        self.b_n = len(self.b_vars)
        assert self.b_n == len(b_p)
        self.b_p = pd.Series(b_p, index=self.b_vars, dtype=np.float64)
        self.S_b = pd.DataFrame(
            S_b, index=self.b_vars, columns=self.b_vars)
        self.b_p_err = np.sqrt(
            pd.Series(np.diag(self.S_b), index=self.b_vars)
        )
        self.forwardKwArgs = forwardKwArgs
        self.multipleForwardKwArgs = multipleForwardKwArgs
        self.verbose = verbose
        self.x_lowerLimit = x_lowerLimit
        self.x_upperLimit = x_upperLimit
        self.useFactorInJac = useFactorInJac
        self.gammaFactor = gammaFactor
        if disturbance is not None:
            self.perturbation = disturbance
            print('Warning. The option "disturbance" is deprecated, use '
                  '"perturbation" instead')
        self.perturbation = perturbation
        self.convergenceFactor = convergenceFactor
        self.convergenceTest = convergenceTest

        self.converged = False
        self.K_i = None
        self.x_i = None
        self.y_i = None
        self.dgf_i = None
        self.A_i = None
        self.d_i2 = None
        self.S_aposteriori_i = None
        self.gam_i = None
        self.convI = None
        self.x_op = None
        self.y_op = None
        self.S_op = None
        self.x_op_err = None
        self.dgf = None
        self.dgf_x = None

        self._y_a = None

        return

    def getJacobian(self, xb, y):
        r'''
        Author: M. Echeverri, May 2021.

        estimate Jacobian using the forward model and the specified 
        perturbation

        Parameters
        ----------
        xb  : pd.Series or list or np.ndarray
          combination of state vector x and parameter vector b
        y : pd.Series or list or np.ndarray
          measurement vector for xb

        Returns
        -------
        pd.DataFrame
          Jacobian around x
        pd.DataFrame
          Jacobian around b
        '''

        xb_vars = self.x_vars + self.b_vars
        # xb = pd.Series(xb, index=xb_vars, dtype=float)
        xb_err = pd.concat((self.x_a_err, self.b_p_err))
        # y = pd.Series(y, index=self.y_vars, dtype=float)

        # If a factor is used to perturb xb, xb must not be zero.
        assert not (self.useFactorInJac and np.any(xb == 0))

        if type(self.perturbation) == float:
            perturbations = dict()
            for key in xb_vars:
                perturbations[key] = self.perturbation
        elif type(self.perturbation) == dict:
            perturbations = self.perturbation
        else:
            raise TypeError("perturbation must be type dict or float")

        # perturbations == perturbation, but perturbation is only a numpy array
        # order in elements of "perturbation" follows xb_vars.
        # Question to MM: does perturbations need to be dict?
        perturbation = np.zeros((len(xb_vars),), dtype=np.float64)
        i = 0
        for key, value in perturbations.items():
            perturbation[i] = value
            i += 1

        perturbedKeys = []
        for tup in xb_vars:
            perturbedKeys.append("perturbed %s" % tup)
        
        
        #------------------------------
        #------------------------------
        #------------------------------
        #pdb.set_trace()
        

        # Numpy array, dims: ("perturbedKeys","xb_bars"); "perturbedKeys" = "perturbed "+"xb_vars"
        # Initialize to xb in rows:
        aux_xb_perturbed = np.ones((len(xb_vars), len(xb_vars)), dtype=np.float64) *\
            xb.to_numpy().reshape(1, len(xb_vars))

        if self.useFactorInJac:
            np.fill_diagonal(aux_xb_perturbed,
                             (np.diag(aux_xb_perturbed) * perturbation))
        else:
            np.fill_diagonal(aux_xb_perturbed,
                             (np.diag(aux_xb_perturbed) + (xb_err.to_numpy()*perturbation)))

        self.xb_perturbed = pd.DataFrame(aux_xb_perturbed,
                                         columns=xb_vars, index=perturbedKeys, dtype=np.float64)

        # Calculate dy : the forward model for all the perturbed profiles
        
        #------------------------------
        #------------------------------
        #------------------------------
        #pdb.set_trace()
        

        if self.multipleForwardKwArgs != None:  # if forward arguments for multiple profiles are provided
            # This version exploits the advantage of calling the forward model for several
            # atmospherical profiles at the same time. In doing so the code fully uses any
            # "under the hood" optimizations inside the "forward" function.

            # Extra function arguments are needed (**self.multipleForwardKwArgs) to pass by the multiple-profiles configuration:
            # for using a forward solver like CRTM or RTTOV, a specific instrument and profile configuration
            # is needed to include the fact that more than 1 profile will be simulated in the same call. Notice
            # that this DOES NOT mean that the "forward" model needs to be modified (this is not required)

            aux_y_perturbed = self.forward(
                self.xb_perturbed.T, **self.multipleForwardKwArgs)
            # Assemble y_perturbed Dataframe to keep consistency the format of the object.
            self.y_perturbed = pd.DataFrame(aux_y_perturbed.T,
                                            columns=self.y_vars,
                                            index=perturbedKeys,
                                            dtype=np.float64
                                            )

        # if forward arguments for multiple profiles are NOT provided   (previous version)
        else:
            self.y_perturbed = pd.DataFrame(
                columns=self.y_vars,
                index=perturbedKeys,
                dtype=np.float64
            )
            for xb_dist in self.xb_perturbed.index:
                self.y_perturbed.loc[xb_dist] = self.forward(
                    self.xb_perturbed.loc[xb_dist], **self.forwardKwArgs)

            # This line only if not using the multiple calls
            aux_y_perturbed = self.y_perturbed.to_numpy().T
            # from Forward model but still using the broadcast Jacobian calculation

        # Calc Jacobian New:
        
        #------------------------------
        #------------------------------
        #------------------------------
        #pdb.set_trace()
        

        # row vector containing observations (y)
        obs = y.to_numpy(dtype=np.float64).reshape(1, len(self.y_vars))

        # Compute dx (i.e. distance to perturbed parameters)
        
        
        #------------------------------
        #------------------------------
        #------------------------------
        #pdb.set_trace()
        
        if self.useFactorInJac:
            aux_dist = (xb.to_numpy(dtype=np.float64)*(perturbation-1.0))
        else:
            aux_dist = (perturbation * xb_err.to_numpy(dtype=np.float64))

        # Check there are no zero distances:

        assert np.sum((aux_dist == 0)) == 0, 'S_a&s_b must not contain zeros on '\
            'diagonal'

        # If assertion pass, then compute the inverse of distance and reshape it into a column vector:
        
        
        #------------------------------
        #------------------------------
        #------------------------------
        #pdb.set_trace()
        
        
        # column vector
        inv_dist = (1/aux_dist).reshape(len(xb_err.to_numpy()), 1)

        # Use Numpy broadcasting rules to efficiently compute the Jacobian

        aux_jacobian = (aux_y_perturbed.T - obs) * inv_dist  # Numpy broadcast

        # Assemble Jacobian Dataframe:

        jacobian = pd.DataFrame(aux_jacobian.T,
                                index=self.y_vars, columns=perturbedKeys)

        jacobian[np.isnan(jacobian) | np.isinf(jacobian)] = 0.
        jacobian_x = jacobian[["perturbed %s" % s for s in self.x_vars]]
        jacobian_b = jacobian[["perturbed %s" % s for s in self.b_vars]]
        
        
        #------------------------------
        #------------------------------
        #------------------------------
        #pdb.set_trace()
        
        
        return jacobian_x, jacobian_b

    def getJacobian_external(self, xb, y):
        r'''
        Author: M. Echeverri, June 2021.

        estimate Jacobian using the external function provided by user and 
        the specified perturbation. This method has external dependencies 

        Parameters
        ----------
        xb  : pd.Series or list or np.ndarray
          combination of state vector x and parameter vector b
        y : pd.Series or list or np.ndarray
          measurement vector for xb

        Returns
        -------
        pd.DataFrame
          Jacobian around x
        pd.DataFrame
          Jacobian around b
        '''

        xb_vars = self.x_vars + self.b_vars
        # xb = pd.Series(xb, index=xb_vars, dtype=float)
        xb_err = pd.concat((self.x_a_err, self.b_p_err))
        # y = pd.Series(y, index=self.y_vars, dtype=float)

        # If a factor is used to perturb xb, xb must not be zero.
        assert not (self.useFactorInJac and np.any(xb == 0))

        if type(self.perturbation) == float:
            perturbations = dict()
            for key in xb_vars:
                perturbations[key] = self.perturbation
        elif type(self.perturbation) == dict:
            perturbations = self.perturbation
        else:
            raise TypeError("perturbation must be type dict or float")

        # perturbations == perturbation, but perturbation is only a numpy array
        # order in elements of "perturbation" follows xb_vars.

        perturbation = np.zeros((len(xb_vars),), dtype=np.float64)
        i = 0
        for key, value in perturbations.items():
            perturbation[i] = value
            i += 1

        perturbedKeys = []
        for tup in xb_vars:
            perturbedKeys.append("perturbed %s" % tup)

        # Compute dx (i.e. distance to perturbed parameters)

        if self.useFactorInJac:
            aux_dist = (xb.to_numpy()*(perturbation-1.0))
        else:
            aux_dist = (perturbation * xb_err.to_numpy())

        # Compute Jacobian using user's Jacobian function

        jac_numpy = self.userJacobian(xb, self.perturbation,
                                      self.y_vars, **self.forwardKwArgs)

        # Assemble  Jacobian Dataframe:

        jacobian = pd.DataFrame(jac_numpy,
                                index=self.y_vars, columns=perturbedKeys)

        jacobian[np.isnan(jacobian) | np.isinf(jacobian)] = 0.
        jacobian_x = jacobian[["perturbed %s" % s for s in self.x_vars]]
        jacobian_b = jacobian[["perturbed %s" % s for s in self.b_vars]]

        # to deprecate? (assertions are present in other parts of pyOpEst,
        #    so this is added here for compliance with those assertions):

        self.xb_perturbed = pd.DataFrame(
            columns=xb_vars, index=perturbedKeys, dtype=float)
        self.y_perturbed = pd.DataFrame(
            columns=self.y_vars,
            index=perturbedKeys,
            dtype=np.float64
        )

        return jacobian_x, jacobian_b

    def doRetrieval(self, maxIter=16, x_0=None, maxTime=1e7,Nstep=4):
        r"""
        run the retrieval

        Parameters
        ----------
        maxIter  : int, optional
          maximum number of iterations, defaults to 10
        x_0  : pd.Series or list or np.ndarray, optional
          first guess for x. If x_0 == None, x_a is taken as first guess.
        maxTime  : int, optional
          maximum runTime, defaults to 1e7 (~ 4 months).
          Note that the forward model is *not* killed if time is exceeded

        Returns
        -------
        bool
          True is convergence was obtained.

        """

        assert maxIter > 0
        self.converged = False
        startTime = time.time()

        self.K_i = [0]*maxIter  # list of jacobians
        self.K_b_i = [0]*maxIter  # list of jacobians for parameter vector
        self.x_i = [0]*(maxIter+1)
        self.y_i = [0]*(maxIter+1)
        self.dgf_i = [0]*maxIter
        self.H_i = [0]*maxIter  # Shannon information content
        self.A_i = [0]*maxIter
        self.d_i2 = [0]*maxIter  # convergence criteria
        self.S_ep_i = [0] * maxIter
        self.S_aposteriori_i = [0] * maxIter
        # self.Pxy_i = [0] *maxIter
        self.gam_i = [1]*maxIter

        S_a = np.array(self.S_a)  # Covariance of prior estimate of x
        assert np.all(S_a == S_a.T), 'S_a must be symmetric'
        S_a_inv = invertMatrix(S_a)  # S_a inverted

        if self.gammaFactor:
            assert len(self.gammaFactor) <= maxIter
            self.gam_i[:len(self.gammaFactor)] = self.gammaFactor

        # treat first guess
        if x_0 is None:
            self.x_i[0] = self.x_a
        else:
            self.x_i[0] = pd.Series(x_0, index=self.x_vars)

        # y of first guess
        xb_i0 = pd.concat((self.x_i[0], self.b_p))
        y = self.forward(xb_i0, **self.forwardKwArgs)
        self.y_i[0] = pd.Series(y, index=self.y_vars, dtype=float)

        for i in range(maxIter):
            if (ErrorRTTOV==False):
                print('************************************************** ITER'+str(i)+'**************************************************')
                print('====BEGIN OF JACOBOAN====')
                if ((int(i/Nstep)*Nstep-i)==0):                
                #if True:                
                    if (self.userJacobian != None):  # then user's Jacobian function is used

                        self.K_i[i], self.K_b_i[i] = self.getJacobian_external(
                            pd.concat((self.x_i[i], self.b_p)), self.y_i[i])

                    else:                           # uses method getJacobian                        
                        
                        self.K_i[i], self.K_b_i[i] = self.getJacobian(
                            pd.concat((self.x_i[i], self.b_p)), self.y_i[i])
                        
                else:
                    self.K_i[i] = self.K_i[i-1]
                    self.K_b_i[i] = self.K_b_i[i-1]
                        
                print('====END OF JACOBOAN====')
            
                if (ErrorRTTOV==False):
                    if np.sum(self.S_b.shape) > 0:
                        S_ep_b = self.K_b_i[i].values.dot(
                            self.S_b.values).dot(self.K_b_i[i].values.T)
                    else:
                        S_ep_b = 0
                    # S_epsilon Covariance of measurement noise including parameter
                    # uncertainty (Rodgers, sec 3.4.3)
                    self.S_ep_i[i] = self.S_y.values + S_ep_b

                    # make sure S_y and S_ep are symmetric
                    assert np.all(self.S_y.values == self.S_y.values.T), \
                        'S_y must be symmetric'
                    assert np.isclose(self.S_ep_i[i], self.S_ep_i[i].T).all(), \
                        'S_ep must be symmetric'

                    # S_ep inverted
                    S_ep_inv = invertMatrix(self.S_ep_i[i])

                    assert np.all(self.y_perturbed.keys() == self.S_y.keys())
                    assert np.all(self.S_y.keys() == self.K_i[i].index)
                    assert np.all(self.S_a.index == self.x_a.index)
                    assert np.all(self.x_a.index.tolist(
                    )+self.b_p.index.tolist() == self.xb_perturbed.columns)
                    assert np.all(self.xb_perturbed.index.tolist(
                    ) == self.K_i[i].columns.tolist()+self.K_b_i[i].columns.tolist())

                    K = np.array(self.K_i[i])

                    # reformulated using Turner and Löhnert 2013:
                    B = (self.gam_i[i] * S_a_inv) + \
                        K.T.dot(S_ep_inv.dot(K))  # eq 3
                    B_inv = invertMatrix(B)
                    self.S_aposteriori_i[i] = B_inv.dot(
                        (self.gam_i[i]**2 * S_a_inv) + K.T.dot(S_ep_inv.dot(K))
                    ).dot(B_inv)  # eq2

                    self.S_aposteriori_i[i] = pd.DataFrame(
                        self.S_aposteriori_i[i],
                        index=self.x_a.index,
                        columns=self.x_a.index
                    )
                    G = B_inv.dot(K.T.dot(S_ep_inv))
                    self.A_i[i] = G.dot(K)  # eq 4

                    # estimate next x
                    self.x_i[i+1] = self.x_a +\
                        B_inv.dot(
                        K.T.dot(S_ep_inv.dot(self.y_obs - self.y_i[i] +
                                             K.dot(self.x_i[i] - self.x_a))))  # eq 1

                    # estimate next y
                    xb_i1 = pd.concat((self.x_i[i+1], self.b_p))
                    
                        
                    y = self.forward(xb_i1, **self.forwardKwArgs)
                    self.y_i[i+1] = pd.Series(y, index=self.y_vars, dtype=float)
                    

                    self.dgf_i[i] = np.trace(self.A_i[i])
                    # eq. 2.80 Rodgers
                    self.H_i[i] = -0.5 * \
                        np.log(np.linalg.det(np.identity(self.x_n) - self.A_i[i]))

                    # check whether i+1 is valid
                    for jj, xKey in enumerate(self.x_vars):
                        if (xKey in self.x_lowerLimit.keys()) and (
                                self.x_i[i+1].iloc[jj] < self.x_lowerLimit[xKey]):
                            print("#"*60)
                            print("reset due to x_lowerLimit: %s from %f to %f in "
                                  "iteration %d" % (
                                      xKey,
                                      self.x_i[i+1].iloc[jj],
                                      self.x_a.iloc[jj], i
                                  ))
                            self.x_i[i+1].iloc[jj] = self.x_a.iloc[jj]
                        if (xKey in self.x_upperLimit.keys()) and (
                                self.x_i[i+1].iloc[jj] > self.x_upperLimit[xKey]):
                            print("#"*60)
                            print("reset due to x_upperLimit: %s from %f to %f in "
                                  "iteration %d" % (
                                      xKey,
                                      self.x_i[i+1].iloc[jj],
                                      self.x_a.iloc[jj], i
                                  ))
                            self.x_i[i+1].iloc[jj] = self.x_a.iloc[jj]
                        if np.isnan(self.x_i[i+1].iloc[jj]):
                            print("#"*60)
                            print("reset due to nan: %s from %f to %f in iteration "
                                  "%d" % (
                                      xKey,
                                      self.x_i[i+1].iloc[jj],
                                      self.x_a.iloc[jj], i
                                  ))
                            self.x_i[i+1].iloc[jj] = self.x_a.iloc[jj]

                    # test in x space for len(y) > len(x)
                    if (
                        (self.convergenceTest == 'x')
                        or
                        ((self.convergenceTest == 'auto') and (self.x_n <= self.y_n))
                    ):
                        # convergence criterion eq 5.29 Rodgers 2000
                        dx = self.x_i[i] - self.x_i[i+1]
                        self.d_i2[i] = dx.T.dot(invertMatrix(
                            self.S_aposteriori_i[i])).dot(dx)
                        d_i2_limit = self.x_n/float(self.convergenceFactor)
                        usingTest = 'x-space'
                    # test in y space for for len(y) < len(x)
                    elif (
                        (self.convergenceTest == 'y')
                        or
                        (self.convergenceTest == 'auto')
                    ):
                        # convergence criterion eqs 5.27 &  5.33 Rodgers 2000
                        dy = self.y_i[i+1] - self.y_i[i]
                        KSaKSep = K.dot(S_a).dot(K.T) + self.S_ep_i[i]
                        KSaKSep_inv = invertMatrix(KSaKSep)

                        S_deyd = self.S_ep_i[i].dot(KSaKSep_inv).dot(self.S_ep_i[i])

                        self.d_i2[i] = dy.T.dot(invertMatrix(
                            S_deyd)).dot(dy)
                        d_i2_limit = self.y_n/float(self.convergenceFactor)
                        usingTest = 'y-space'
                    else:
                        raise ValueError('Do not understand convergenceTest %s' %
                                         self.convergenceTest)

                    assert not self.d_i2[i] < 0, 'a negative convergence cirterion'
                    ' means someting has gotten really wrong'

                    # stop if we converged in the step before
                    if self.converged:
                        if(self.verbose != None):
                                if(self.verbose):
                                    print("%.2f s, iteration %i, degrees of freedom: %.2f of %i, "
                                          "done.  %.3f" % (
                                              time.time()-startTime, i, self.dgf_i[i], self.x_n,
                                              self.d_i2[i]))          
                        else:
                            print("%.2f s, iteration %i, degrees of freedom: %.2f of %i, "
                                          "done.  %.3f" % (
                                              time.time()-startTime, i, self.dgf_i[i], self.x_n,
                                              self.d_i2[i]))                                         
                        break

                    elif ((time.time()-startTime) > maxTime):
                        print("%.2f s, iteration %i, degrees of freedom: %.2f of %i."
                              " maximum Time exceeded! STOP  %.3f" % (
                                  time.time()-startTime, i, self.dgf_i[i], self.x_n,
                                  self.d_i2[i]))

                        self.converged = False

                        break

                    # calculate the convergence criteria
                    if i != 0:
                        if (np.abs(self.d_i2[i]) < d_i2_limit) and (
                                self.gam_i[i] == 1) and (self.d_i2[i] != 0):
                            if(self.verbose != None):
                                if(self.verbose):    
                                    print("%.2f s, iteration %i, degrees of freedom: %.2f of"
                                          " %i, converged (%s):  %.3f" % (
                                             time.time() -
                                              startTime, i, self.dgf_i[i], self.x_n,
                                              usingTest, self.d_i2[i]))
                            else:
                                print("%.2f s, iteration %i, degrees of freedom: %.2f of"
                                          " %i, converged (%s):  %.3f" % (
                                             time.time() -
                                              startTime, i, self.dgf_i[i], self.x_n,
                                              usingTest, self.d_i2[i]))                  
                            self.converged = True
                        elif (i > 1) and (self.dgf_i[i] == 0):
                            print("%.2f s, iteration %i, degrees of freedom: %.2f of "
                                  "%i.degrees of freedom 0! STOP  %.3f" % (
                                      time.time() -
                                      startTime, i, self.dgf_i[i], self.x_n,
                                      self.d_i2[i]))
                            self.converged = False

                            break
                        else:
                            if(self.verbose != None):
                                if(self.verbose):
                                    print("%.2f s, iteration %i, degrees of freedom:"
                                          " %.2f of %i, not converged (%s): "
                                          " %.3f" % (
                                              time.time()-startTime, i, self.dgf_i[i],
                                              self.x_n, usingTest, self.d_i2[i]))
                            else:
                                print("%.2f s, iteration %i, degrees of freedom:"
                                          " %.2f of %i, not converged (%s): "
                                          " %.3f" % (
                                              time.time()-startTime, i, self.dgf_i[i],
                                              self.x_n, usingTest, self.d_i2[i]))                  

                    #print("%.2f s , TimeRest" % (time.time()-startTimeRest))
            
        if (ErrorRTTOV==False):
            self.K_i = self.K_i[:i+1]
            self.K_b_i = self.K_b_i[:i+1]
            self.x_i = self.x_i[:i+2]
            self.y_i = self.y_i[:i+2]
            self.dgf_i = self.dgf_i[:i+1]
            self.A_i = self.A_i[:i+1]
            self.H_i = self.H_i[:i+1]
            self.d_i2 = self.d_i2[:i+1]
            self.S_ep_i = self.S_ep_i[:i+1]

            self.S_aposteriori_i = self.S_aposteriori_i[:i+1]

            self.gam_i = self.gam_i[:i+1]
            if self.converged:
                self.convI = i

                self.x_op = self.x_i[i]
                self.y_op = self.y_i[i]
                self.S_op = self.S_aposteriori_i[i]
                self.x_op_err = np.sqrt(
                    pd.Series(np.diag(
                        self.S_aposteriori_i[self.convI]), index=self.x_vars)
                )
                self.dgf = self.dgf_i[i]
                self.dgf_x = pd.Series(
                    np.diag(self.A_i[i]), index=self.x_vars
                )

            else:
                self.convI = -9999
                self.x_op = np.nan
                self.y_op = np.nan
                self.S_op = np.nan
                self.x_op_err = np.nan
                self.dgf = np.nan
                self.dgf_x = np.nan
        else:    
            self.convI = -9999
            self.x_op = np.nan
            self.y_op = np.nan
            self.S_op = np.nan
            self.x_op_err = np.nan
            self.dgf = np.nan
            self.dgf_x = np.nan

        return self.converged

    def getJacobian_output(self, xb, y):
        r'''
        Author: M. Echeverri, May 2021.

        estimate Jacobian using the forward model and the specified 
        perturbation

        Parameters
        ----------
        xb  : pd.Series or list or np.ndarray
          combination of state vector x and parameter vector b
        y : pd.Series or list or np.ndarray
          measurement vector for xb

        Returns
        -------
        pd.DataFrame
          Jacobian around x
        pd.DataFrame
          Jacobian around b
        '''

        xb_vars = self.x_vars + self.b_vars
        # xb = pd.Series(xb, index=xb_vars, dtype=float)
        xb_err = pd.concat((self.x_a_err, self.b_p_err))
        # y = pd.Series(y, index=self.y_vars, dtype=float)

        # If a factor is used to perturb xb, xb must not be zero.
        assert not (self.useFactorInJac and np.any(xb == 0))

        if type(self.perturbation) == float:
            perturbations = dict()
            for key in xb_vars:
                perturbations[key] = self.perturbation
        elif type(self.perturbation) == dict:
            perturbations = self.perturbation
        else:
            raise TypeError("perturbation must be type dict or float")

        # perturbations == perturbation, but perturbation is only a numpy array
        # order in elements of "perturbation" follows xb_vars.
        # Question to MM: does perturbations need to be dict?
        perturbation = np.zeros((len(xb_vars),), dtype=np.float64)
        i = 0
        for key, value in perturbations.items():
            perturbation[i] = value
            i += 1

        perturbedKeys = []
        for tup in xb_vars:
            perturbedKeys.append("perturbed %s" % tup)        

        # Numpy array, dims: ("perturbedKeys","xb_bars"); "perturbedKeys" = "perturbed "+"xb_vars"
        # Initialize to xb in rows:
        aux_xb_perturbed = np.ones((len(xb_vars), len(xb_vars)), dtype=np.float64) *\
            xb.to_numpy().reshape(1, len(xb_vars))

        if self.useFactorInJac:
            np.fill_diagonal(aux_xb_perturbed,
                             (np.diag(aux_xb_perturbed) * perturbation))
        else:
            np.fill_diagonal(aux_xb_perturbed,
                             (np.diag(aux_xb_perturbed) + (xb_err.to_numpy()*perturbation)))

        self.xb_perturbed = pd.DataFrame(aux_xb_perturbed,
                                         columns=xb_vars, index=perturbedKeys, dtype=np.float64)

        # Calculate dy : the forward model for all the perturbed profiles

        if self.multipleForwardKwArgs != None:  # if forward arguments for multiple profiles are provided
            # This version exploits the advantage of calling the forward model for several
            # atmospherical profiles at the same time. In doing so the code fully uses any
            # "under the hood" optimizations inside the "forward" function.

            # Extra function arguments are needed (**self.multipleForwardKwArgs) to pass by the multiple-profiles configuration:
            # for using a forward solver like CRTM or RTTOV, a specific instrument and profile configuration
            # is needed to include the fact that more than 1 profile will be simulated in the same call. Notice
            # that this DOES NOT mean that the "forward" model needs to be modified (this is not required)

            aux_y_perturbed = self.forward(
                self.xb_perturbed.T, **self.multipleForwardKwArgs)
            # Assemble y_perturbed Dataframe to keep consistency the format of the object.
            self.y_perturbed = pd.DataFrame(aux_y_perturbed.T,
                                            columns=self.y_vars,
                                            index=perturbedKeys,
                                            dtype=np.float64
                                            )

        # if forward arguments for multiple profiles are NOT provided   (previous version)
        else:
            self.y_perturbed = pd.DataFrame(
                columns=self.y_vars,
                index=perturbedKeys,
                dtype=np.float64
            )
            for xb_dist in self.xb_perturbed.index:
                self.y_perturbed.loc[xb_dist] = self.forward(
                    self.xb_perturbed.loc[xb_dist], **self.forwardKwArgs)

            # This line only if not using the multiple calls
            aux_y_perturbed = self.y_perturbed.to_numpy().T
            # from Forward model but still using the broadcast Jacobian calculation

        # Calc Jacobian New:

        # row vector containing observations (y)
        obs = y.to_numpy(dtype=np.float64).reshape(1, len(self.y_vars))

        # Compute dx (i.e. distance to perturbed parameters)
        
        if self.useFactorInJac:
            aux_dist = (xb.to_numpy(dtype=np.float64)*(perturbation-1.0))
        else:
            aux_dist = (perturbation * xb_err.to_numpy(dtype=np.float64))

        # Check there are no zero distances:

        assert np.sum((aux_dist == 0)) == 0, 'S_a&s_b must not contain zeros on '\
            'diagonal'

        # If assertion pass, then compute the inverse of distance and reshape it into a column vector:
        
        # column vector
        inv_dist = (1/aux_dist).reshape(len(xb_err.to_numpy()), 1)

        # Use Numpy broadcasting rules to efficiently compute the Jacobian

        aux_jacobian = (aux_y_perturbed.T - obs) * inv_dist  # Numpy broadcast

        # Assemble Jacobian Dataframe:

        jacobian = pd.DataFrame(aux_jacobian.T,
                                index=self.y_vars, columns=perturbedKeys)

        jacobian[np.isnan(jacobian) | np.isinf(jacobian)] = 0.
        jacobian_x = jacobian[["perturbed %s" % s for s in self.x_vars]]
        jacobian_b = jacobian[["perturbed %s" % s for s in self.b_vars]]        
        
        return jacobian_x, jacobian_b
    

    def Output_Jacobian(self, maxIter=16, x_0=None, maxTime=1e7,Nstep=4):
        r"""
        run the retrieval

        Parameters
        ----------
        maxIter  : int, optional
          maximum number of iterations, defaults to 10
        x_0  : pd.Series or list or np.ndarray, optional
          first guess for x. If x_0 == None, x_a is taken as first guess.
        maxTime  : int, optional
          maximum runTime, defaults to 1e7 (~ 4 months).
          Note that the forward model is *not* killed if time is exceeded

        Returns
        -------
        bool
          True is convergence was obtained.

        """

        assert maxIter > 0
        self.converged = False
        startTime = time.time()

        self.K_i = [0]*maxIter  # list of jacobians
        self.K_b_i = [0]*maxIter  # list of jacobians for parameter vector
        self.x_i = [0]*(maxIter+1)
        self.y_i = [0]*(maxIter+1)
        self.dgf_i = [0]*maxIter
        self.H_i = [0]*maxIter  # Shannon information content
        self.A_i = [0]*maxIter
        self.d_i2 = [0]*maxIter  # convergence criteria
        self.S_ep_i = [0] * maxIter
        self.S_aposteriori_i = [0] * maxIter
        # self.Pxy_i = [0] *maxIter
        self.gam_i = [1]*maxIter

        S_a = np.array(self.S_a)  # Covariance of prior estimate of x
        assert np.all(S_a == S_a.T), 'S_a must be symmetric'
        S_a_inv = invertMatrix(S_a)  # S_a inverted

        if self.gammaFactor:
            assert len(self.gammaFactor) <= maxIter
            self.gam_i[:len(self.gammaFactor)] = self.gammaFactor

        # treat first guess
        if x_0 is None:
            self.x_i[0] = self.x_a
        else:
            self.x_i[0] = pd.Series(x_0, index=self.x_vars)

        # y of first guess
        xb_i0 = pd.concat((self.x_i[0], self.b_p))
        y = self.forward(xb_i0, **self.forwardKwArgs)
        self.y_i[0] = pd.Series(y, index=self.y_vars, dtype=float)
        
        i=0
        
        self.K_i[i], self.K_b_i[i] = self.getJacobian_output(
            pd.concat((self.x_i[i], self.b_p)), self.y_i[i])
        
        np.savetxt('OutputJacobian_'+str(self.forwardKwArgs['date'][0])+str(self.forwardKwArgs['date'][1])+'_'+str(self.forwardKwArgs['zenithAngleATMS'])+'_'+str(self.forwardKwArgs['emiss'][0])+'.txt',self.K_i[i])
        
        #-------
        T_optimal, Q_optimal = splitTQ(self.x_a)

        np.savetxt('Tprior_'+str(self.forwardKwArgs['date'][0])+str(self.forwardKwArgs['date'][1])+'_'+str(self.forwardKwArgs['zenithAngleATMS'])+'_'+str(self.forwardKwArgs['emiss'][0])+'.txt',T_optimal)
        np.savetxt('Qprior_'+str(self.forwardKwArgs['date'][0])+str(self.forwardKwArgs['date'][1])+'_'+str(self.forwardKwArgs['zenithAngleATMS'])+'_'+str(self.forwardKwArgs['emiss'][0])+'.txt',Q_optimal)
        
        return

    @property
    def y_a(self):
        '''
        Estimate the observations corresponding to the prior.
        '''
        if self._y_a is None:
            xb_a = pd.concat((self.x_a, self.b_p))
            self._y_a = pd.Series(self.forward(xb_a, **self.forwardKwArgs),
                                  index=self.y_vars)
        return self._y_a

    def linearityTest(
        self,
        maxErrorPatterns=10,
        significance=0.05,
        atol=1e-5
    ):
        """
        test whether the solution is moderately linear following chapter
        5.1 of Rodgers 2000.
        values lower than 1 indicate that the effect of linearization is
        smaller than the measurement error and problem is nearly linear.
        Populates self.linearity.

        Parameters
        ----------
        maxErrorPatterns  : int, optional
          maximum number of error patterns to return. Provide None to return
        all.
        significance  : real, optional
          significance level, defaults to 0.05, i.e. probability is 5% that
           correct null hypothesis is rejected. Only used when testing 
           against x_truth.
        atol : float (default 1e-5)
            The absolute tolerance for comparing eigen values to zero. We 
            found that values should be than the numpy.isclose defualt value 
            of 1e-8.

        Returns
        -------
        self.linearity: float
          ratio of error due to linearization to measurement error sorted by 
          size. Should be below 1 for all.
        self.trueLinearityChi2: float
           Chi2 value that model is moderately linear based on 'self.x_truth'.
           Must be smaller than critical value to conclude thast model is
           linear.
        self.trueLinearityChi2Critical: float
           Corresponding critical Chi2 value. 
        """
        self.linearity = np.zeros(self.x_n)*np.nan
        self.trueLinearityChi2 = np.nan
        self.trueLinearityChi2Critical = np.nan

        if not self.converged:
            print("did not converge")
            return self.linearity, self.trueLinearity
        lamb, II = np.linalg.eig(self.S_aposteriori_i[self.convI])
        S_ep_inv = invertMatrix(np.array(self.S_ep_i[self.convI]))
        lamb[np.isclose(lamb, 0)] = 0
        if np.any(lamb < 0):
            print(
                "found negative eigenvalues of S_aposteriori_i, "
                " S_aposteriori_i not semipositive definite!")
            return self.linearity, self.trueLinearity
        error_pattern = lamb**0.5 * II
        for hh in range(self.x_n):
            x_hat = self.x_i[self.convI] + \
                error_pattern[:, hh]  # estimated truth
            xb_hat = pd.concat((x_hat, self.b_p))
            y_hat = self.forward(xb_hat, **self.forwardKwArgs)
            del_y = (y_hat - self.y_i[self.convI] - self.K_i[self.convI].dot(
                (x_hat - self.x_i[self.convI]).values))
            self.linearity[hh] = del_y.T.dot(S_ep_inv).dot(del_y)

        self.linearity = sorted(
            self.linearity, reverse=True)[slice(None, maxErrorPatterns)]

        if self.x_truth is not None:
            xb_truth = pd.concat((self.x_truth, self.b_p))
            y_truth = self.forward(xb_truth, **self.forwardKwArgs)
            del_y = (y_truth - self.y_i[self.convI] - self.K_i[self.convI].dot(
                (self.x_truth - self.x_i[self.convI]).values))
            self.trueLinearity = del_y.T.dot(S_ep_inv).dot(del_y)

            res = _testChi2(self.S_y.values, del_y, significance, atol)
            self.trueLinearityChi2, self.trueLinearityChi2Critical = res

        return self.linearity, self.trueLinearityChi2, \
            self.trueLinearityChi2Critical

    def chiSquareTest(self, significance=0.05):
        '''

        test with significance level 'significance' whether 
        A) optimal solution agrees with observation in Y space
        B) observation agrees with prior in Y space
        C) optimal solution agrees with prior in Y space
        D) optimal solution agrees with priot in X space

        Parameters
        ----------
        significance  : real, optional
          significance level, defaults to 0.05, i.e. probability is 5% that
           correct null hypothesis is rejected.

        Returns
        -------
        Pandas Series (dtype bool):
            True if test is passed
        Pandas Series (dtype float):
            Chi2 value for tests. Must be smaler than critical value to pass
            tests.
        Pandas Series (dtype float):
            Critical Chi2 value for tests
        '''
        chi2names = pd.Index([
            'Y_Optimal_vs_Observation',
            'Y_Observation_vs_Prior',
            'Y_Optimal_vs_Prior',
            'X_Optimal_vs_Prior',
        ], name='chi2test')

        chi2Cols = [
            'chi2value',
            'chi2critical',
        ]

        if not self.converged:
            print("did not converge")
            pd.DataFrame(
                np.zeros((4, 2)),
                index=chi2names,
                columns=chi2Cols,
            )*np.nan
        else:
            YOptimalObservation = self.chiSquareTestYOptimalObservation(
                significance=significance)
            YObservationPrior = self.chiSquareTestYObservationPrior(
                significance=significance)
            YOptimalPrior = self.chiSquareTestYOptimalPrior(
                significance=significance)
            XOptimalPrior = self.chiSquareTestXOptimalPrior(
                significance=significance)

            self.chi2Results = pd.DataFrame(
                np.array([
                    YOptimalObservation,
                    YObservationPrior,
                    YOptimalPrior,
                    XOptimalPrior,
                ]),
                index=chi2names,
                columns=chi2Cols,
            )

        passed = self.chi2Results['chi2value'] < self.\
            chi2Results['chi2critical']

        return passed, self.chi2Results['chi2value'], \
            self.chi2Results['chi2critical']

    def chiSquareTestYOptimalObservation(self, significance=0.05, atol=1e-5):
        """
        test with significance level 'significance' whether retrieval agrees
        with measurements (see chapter 12.3.2 of Rodgers, 2000)

        Parameters
        ----------
        significance  : real, optional
          significance level, defaults to 0.05, i.e. probability is 5% that
           correct null hypothesis is rejected.
        atol : float (default 1e-5)
            The absolute tolerance for comparing eigen values to zero. We 
            found that values should be than the numpy.isclose defualt value 
            of 1e-8.
        Returns
        -------
        chi2Passed : bool
          True if chi² test passed, i.e. OE  retrieval agrees with
          measurements and null hypothesis is NOT rejected.
        chi2 : real
          chi² value
        chi2TestY : real
          chi²  cutoff value with significance 'significance'

        """
        assert self.converged

        Sa = self.S_a.values
        Sep = self.S_ep_i[self.convI]
        K = self.K_i[self.convI].values

        # Rodgers eq. 12.9
        KSaKSep_inv = invertMatrix(K.dot(Sa).dot(K.T) + Sep)
        S_deyd = Sep.dot(KSaKSep_inv).dot(Sep)
        delta_y = self.y_i[self.convI] - self.y_obs

        chi2, chi2TestY = _testChi2(S_deyd, delta_y, significance, atol)

        return chi2, chi2TestY

    def chiSquareTestYObservationPrior(self, significance=0.05, atol=1e-5):
        """
        test with significance level 'significance' whether measurement agrees
        with prior (see chapter 12.3.3.1 of Rodgers, 2000)

        Parameters
        ----------
        significance  : real, optional
          significance level, defaults to 0.05, i.e. probability is 5% that
           correct null hypothesis is rejected.
        atol : float (default 1e-5)
            The absolute tolerance for comparing eigen values to zero. We 
            found that values should be than the numpy.isclose defualt value 
            of 1e-8.
        Returns
        -------
        YObservationPrior : bool
          True if chi² test passed, i.e. OE  retrieval agrees with
          measurements and null hypothesis is NOT rejected.
        YObservationPrior: real
          chi² value
        chi2TestY : real
          chi²  cutoff value with significance 'significance'

        """

        assert self.converged

        delta_y = self.y_obs - self.y_a
        Sa = self.S_a.values
        Sep = self.S_ep_i[self.convI]
        K = self.K_i[self.convI].values
        KSaKSep = K.dot(Sa).dot(K.T) + Sep

        chi2, chi2TestY = _testChi2(KSaKSep, delta_y, significance, atol)

        return chi2, chi2TestY

    def chiSquareTestYOptimalPrior(self, significance=0.05, atol=1e-5):
        """
        test with significance level 'significance' whether retrieval result agrees
        with prior in y space (see chapter 12.3.3.3 of Rodgers, 2000)

        Parameters
        ----------
        significance  : real, optional
          significance level, defaults to 0.05, i.e. probability is 5% that
           correct null hypothesis is rejected.
        atol : float (default 1e-5)
            The absolute tolerance for comparing eigen values to zero. We 
            found that values should be than the numpy.isclose defualt value 
            of 1e-8.

        Returns
        -------
        chi2Passe : bool
          True if chi² test passed, i.e. OE  retrieval agrees with
          Prior and null hypothesis is NOT rejected.
        chi2: real
          chi² value
        chi2TestY : real
          chi²  cutoff value with significance 'significance'

        """

        assert self.converged

        delta_y = self.y_i[self.convI] - self.y_a
        Sa = self.S_a.values
        S_ep = self.S_ep_i[self.convI]
        K = self.K_i[self.convI].values

        # Rodgers eq.12.16
        KSaK = K.dot(Sa).dot(K.T)
        KSaKSep_inv = invertMatrix(KSaK + S_ep)
        Syd = KSaK.dot(KSaKSep_inv).dot(KSaK)

        chi, chi2TestY = _testChi2(Syd, delta_y, significance, atol)

        #######  Alternative based on execise Rodgers 12.1 #######

        # Se = y_cov.values
        # K = self.K_i[self.convI].values
        # Sa = x_cov.sel(season=season).to_pandas().loc[x_vars,x_vars].values
        # d_y = (self.y_op[y_vars] - self.y_a[y_vars]).values

        # SaSqr = scipy.linalg.sqrtm(Sa)
        # SaSqr_inv = pyOE.pyOEcore.invertMatrix(SaSqr)

        # SeSqr = scipy.linalg.sqrtm(Se)
        # SeSqr_inv = pyOE.pyOEcore.invertMatrix(SeSqr)

        # Ktilde = SeSqr_inv.dot(K).dot(SaSqr)
        # U,s,vT = np.linalg.svd(Ktilde, full_matrices=False)
        # Lam = np.diag(s)
        # LamSq = Lam.dot(Lam)

        # m = len(y_vars)
        # invM= pyOE.pyOEcore.invertMatrix(LamSq + np.eye(m))
        # Sy = SeSqr.dot(U).dot(LamSq).dot(invM).dot(LamSq).dot(U.T).dot(SeSqr)

        # Sz4y = LamSq.dot(invM).dot(LamSq)
        # z4y = U.T.dot(SeSqr_inv).dot(d_y)

        # eigenvalues_compl = np.diag(Sz4y) # because it is diagonal

        # eigenvalues = s**4/(1+s**2) #equivalent!
        # assert np.isclose(eigenvalues_compl, eigenvalues).all()

        # notNull = ~np.isclose(0,eigenvalues)
        # chi2 = z4y[notNull]**2/eigenvalues[notNull]
        # chi2critical = scipy.stats.chi2.isf(significance, 1)

        return chi, chi2TestY

    def chiSquareTestXOptimalPrior(self, significance=0.05, atol=1e-5):
        """
        test with significance level 'significance' whether retrieval agrees
        with prior in x space (see chapter 12.3.3.3 of Rodgers, 2000)

        Parameters
        ----------
        significance  : real, optional
          significance level, defaults to 0.05, i.e. probability is 5% that
           correct null hypothesis is rejected.
        atol : float (default 1e-5)
            The absolute tolerance for comparing eigen values to zero. We 
            found that values should be than the numpy.isclose defualt value 
            of 1e-8.

        Returns
        -------
        chi2Passed : bool
          True if chi² test passed, i.e. OE  retrieval agrees with
          Prior and null hypothesis is NOT rejected.
        chi2 : real
          chi² value
        chi2TestX : real
          chi² cutoff value with significance 'significance'
        """

        assert self.converged

        delta_x = self.x_op - self.x_a

        Sa = self.S_a.values
        K = self.K_i[self.convI].values
        S_ep = self.S_ep_i[self.convI]

        # Rodgers eq. 12.12
        KSaKSep_inv = invertMatrix(K.dot(Sa).dot(K.T) + S_ep)
        Sxd = Sa.dot(K.T).dot(KSaKSep_inv).dot(K).dot(Sa)
        chi2, chi2TestX = _testChi2(Sxd, delta_x, significance, atol)

        #######  Alternative based on execise Rodgers 12.1 #######

        # Se = y_cov.values
        # K = self.K_i[self.convI].values
        # Sa = x_cov.sel(season=season).to_pandas().loc[x_vars,x_vars].values
        # d_x = (self.x_op[x_vars] - self.x_a[x_vars]).values

        # SaSqr = scipy.linalg.sqrtm(Sa)
        # SaSqr_inv = pyOE.pyOEcore.invertMatrix(SaSqr)

        # SeSqr = scipy.linalg.sqrtm(Se)
        # SeSqr_inv = pyOE.pyOEcore.invertMatrix(SeSqr)

        # Ktilde = SeSqr_inv.dot(K).dot(SaSqr)
        # U,s,vT = np.linalg.svd(Ktilde, full_matrices=False)
        # Lam = np.diag(s)

        # m = len(y_vars)
        # invM= pyOE.pyOEcore.invertMatrix(Lam.dot(Lam) + np.eye(m))
        # Sx = SaSqr.dot(vT.T).dot(Lam).dot(invM).dot(Lam).dot(vT)

        # z4x = vT.dot(SaSqr_inv).dot(d_x)
        # Sz4x = Lam.dot(invM).dot(Lam)

        # eigenvalues_compl = np.diag(Sz4x) # because it is diagonal
        # eigenvalues = s**2/(1+s**2) #equivalent!

        # assert np.isclose(eigenvalues_compl, eigenvalues).all()

        # notNull = ~np.isclose(0,eigenvalues)
        # chi2 = z4x[notNull]**2/eigenvalues[notNull]
        # chi2critical = scipy.stats.chi2.isf(significance, 1)

        return chi2, chi2TestX

    def saveResults(self, fname):
        r'''
        Helper function to save a pyOptimalEstimation object. The forward
        operator is removed from the pyOptimalEstimation object before saving.

        Parameters
        ----------
        fname : str
          filename

        Returns
        -------
        None
        '''
        oeDict = deepcopy(self.__dict__)
        for k in list(oeDict.keys()):
            if k in ['forward', 'userJacobian']:
                oeDict.pop(k)

        np.save(fname, oeDict)
        return

    def plotIterations(
        self,
        cmap='viridis',
        figsize=(8, 10),
        legend=True,
        mode='ratio',
    ):
        r'''
        Plot the retrieval results using 4 panels: (1) iterations of x
        (normalized to self.x_truth or x[0]), (2) iterations of y (normalized
        to y_obs), (3) iterations of degrees of freedom, (4) iterations of
        convergence criteria

        Parameters
        ----------
        fileName : str, optional
          plot is saved to fileName, if provided
        cmap : str, optional
          colormap for 1st and 2nd panel (default 'hsv')
        figsize : tuple, optional
          Figure size in inch (default (8, 10))
        legend : bool, optional
          Add legend for X and Y (defualt True)
        mode : str, optional
          plot 'ratio' or 'difference' to truth/prior/measurements 
          (defualt: ratio)

        Returns
        -------
        matplotlib figure object
          The created figure.
        '''
        fig, [sp1, sp2, sp3, sp4] = plt.subplots(figsize=figsize, nrows=4,
                                                 sharex=True)
        d_i2 = np.array(self.d_i2)
        dgf_i = np.array(self.dgf_i)

        try:
            gamma = np.array(self.gam_i)
            noGam = len(gamma[gamma != 1])
            ind = np.argmin(d_i2[noGam:]) + noGam - 1
        except:
            ind = 0

        if self.converged:
            fig.suptitle('Sucessfully converged. Convergence criterion: %.3g'
                         ' Degrees of freedom: %.3g' % (d_i2[ind], dgf_i[ind]))
        else:
            fig.suptitle('Not converged. Convergence criterion: %.3g  Degrees'
                         ' of freedom: %.3g' % (d_i2[ind], dgf_i[ind]))

        colors = _niceColors(len(self.x_i[0].keys()), cmap=cmap)
        for kk, key in enumerate(self.x_i[0].keys()):
            xs = list()
            for xx in self.x_i[:-1]:
                xs.append(xx[key])
            if mode == 'ratio':
                if self.x_truth is not None:
                    xs.append(self.x_truth[key])
                    xs = np.array(xs) / self.x_truth[key]
                else:
                    xs = np.array(xs) / xs[0]
            elif mode == 'difference':
                if self.x_truth is not None:
                    xs.append(self.x_truth[key])
                    xs = np.array(xs) - self.x_truth[key]
                else:
                    xs = np.array(xs) - xs[0]
            else:
                ValueError('Do not understand mode %s' % mode)
            sp1.plot(xs, label=key, color=colors[kk])
        if legend:
            leg = sp1.legend(loc="best",
                             prop=font_manager.FontProperties(size=8))
            leg.get_frame().set_alpha(0.5)
        # sp1.set_xlabel("iteration")
        if self.x_truth is not None:
            sp1.set_ylabel("x-values\n(%s to truth)" % mode)
        else:
            sp1.set_ylabel("x-values\n(%s to prior)" % mode)

        sp1.axvline(ind, color="k")
        sp1.axvline(len(self.x_i)-2, ls=":", color="k")

        colors = _niceColors(len(self.y_i[0].keys()), cmap=cmap)
        for kk, key in enumerate(self.y_i[0].keys()):
            ys = list()
            for yy in self.y_i:
                ys.append(yy[key])
            ys.append(self.y_obs[key])
            if mode == 'ratio':
                ys = np.array(ys) / ys[-1]
            elif mode == 'difference':
                ys = np.array(ys) - ys[-1]
            sp2.plot(ys, label=key, color=colors[kk])
        if legend:
            leg = sp2.legend(loc="best",
                             prop=font_manager.FontProperties(size=8))
            leg.get_frame().set_alpha(0.5)
        sp2.set_ylabel("y-values\n(%s to measurements)" % mode)
        sp2.axvline(ind, color="k")
        sp2.axvline(len(self.x_i)-2, ls=":", color="k")

        sp3.plot(dgf_i, label="degrees of freedom")
        sp3.set_ylabel("degrees of freedom")
        sp3.axvline(len(self.x_i)-2, ls=":", color="k")
        sp3.axvline(ind, color="k")

        sp4.plot(d_i2, label="d_i2")
        sp4.set_xlabel("iteration")
        sp4.set_ylabel("convergence criterion")
        fig.subplots_adjust(hspace=0.1)
        sp4.set_xlim(0, len(self.x_i)-1)
        sp4.axvline(len(self.x_i)-2, ls=":", color="k")
        sp4.axvline(ind, color="k")
        sp4.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        xlabels = list(map(lambda x: "%i" % x, sp4.get_xticks()))
        xlabels[-1] = "truth/obs."
        sp4.set_xticklabels(xlabels)

        return fig

    def summary(self, *args, **kwargs):
        DeprecationWarning('Use summarize instead of summary!')
        return self.summarize(self, *args, **kwargs)

    def summarize(self, returnXarray=False, combineXB=False):
        '''Provide a summary of the retrieval results as a dictionary.

        Parameters
        ----------
        returnXarray : {bool}, optional
          return xarray dataset instead of dict. Can be easily combined when
          applying the retrieval multiple times. (the default is False)
        combineXB : {bool}, optional
          append b parameter values to state vector X variables. Can be useful
          for comparing runs with and without b parameters.

        Returns
        -------
        dict or xarray.Dataset
          Summary of retrieval results
        '''

        if self.convI < 0:
            raise RuntimeError("Retrieval did not run successfully")

        summary = {}
        summary['x_a'] = self.x_a.rename_axis('x_vars')
        summary['x_a_err'] = self.x_a_err.rename_axis('x_vars')
        summary['S_a'] = self.S_a.rename_axis(
            'x_vars').rename_axis('x_vars_T', axis=1)
        summary['x_op'] = self.x_op.rename_axis('x_vars')
        summary['x_op_err'] = self.x_op_err.rename_axis('x_vars')
        summary['S_op'] = self.S_op.rename_axis(
            'x_vars').rename_axis('x_vars_T', axis=1)
        summary['dgf_x'] = self.dgf_x.rename_axis('x_vars')
        summary['y_obs'] = self.y_obs.rename_axis('y_vars')
        summary['S_y'] = self.S_y.rename_axis(
            'y_vars').rename_axis('y_vars_T', axis=1)

        summary['y_op'] = self.y_op.rename_axis('y_vars')
        if self.x_truth is not None:
            summary['x_truth'] = self.x_truth.rename_axis('x_vars')

        if hasattr(self, 'nonlinearity'):
            summary['nonlinearity'] = self.linearity
        if hasattr(self, 'trueLinearityChi2'):
            summary['trueLinearityChi2'] = self.trueLinearityChi2
            summary['trueLinearityChi2Critical'] = \
                self.trueLinearityChi2Critical
        if hasattr(self, 'chi2Results'):
            summary['chi2value'] = self.chi2Results['chi2value']
            summary['chi2critical'] = self.chi2Results['chi2critical']

        summary['dgf'] = self.dgf_i[self.convI]
        summary['convergedIteration'] = self.convI

        if (not combineXB) and (len(self.b_vars) > 0):
            summary['b_p'] = self.b_p.rename_axis('b_vars')
            summary['S_b'] = self.S_b.rename_axis(
                'b_vars').rename_axis('b_vars_T', axis=1)
            summary['b_p_err'] = self.b_p_err.rename_axis('b_vars')

        elif combineXB and (len(self.b_vars) > 0):
            summary['x_a'] = pd.concat(
                (summary['x_a'], self.b_p)).rename_axis('x_vars')
            summary['x_op'] = pd.concat(
                (summary['x_op'], self.b_p)).rename_axis('x_vars')
            summary['x_op_err'] = pd.concat(
                (summary['x_op_err'], self.b_p_err)).rename_axis('x_vars')
            summary['dgf_x'] = pd.concat(
                (
                    summary['dgf_x'],
                    pd.Series(np.zeros(self.b_n), index=self.b_vars)
                )
            ).rename_axis('x_vars')
            summary['S_a'] = pd.concat(
                (summary['S_a'], self.S_b), sort=False
            ).rename_axis('x_vars').rename_axis('x_vars_T', axis=1)
            summary['S_op'] = pd.concat(
                (summary['S_op'], self.S_b), sort=False
            ).rename_axis('x_vars').rename_axis('x_vars_T', axis=1)

        if returnXarray:
            import xarray as xr
            summary = xr.Dataset(summary)

        return summary

def optimalEstimation_loadResults(fname, allow_pickle=True):
    r'''
    Helper function to load a saved pyOptimalEstimation object

    Parameters
    ----------
    fname : str
      filename

    Returns
    -------
    pyOptimalEstimation object
      pyOptimalEstimation obtained from file.
    '''
    oeDict = np.load(fname, allow_pickle=allow_pickle)
    oe = _oeDict2Object(oeDict.tolist())
    return oe

def invertMatrix(A, raise_error=True):
    '''
    Wrapper funtion for np.linalg.inv, because original function reports
    LinAlgError if nan in array for some numpy versions. We want that the
    retrieval is robust with respect to that. Also, checks for singular 
    matrices were added.

    Parameters
    ----------
    A : (..., M, M) array_like
        Matrix to be inverted.
    raise_error : {bool}, optional
        ValueError is raised if A is singular (the default is True)

    Returns
    -------
    Ainv : (..., M, M) ndarray or matrix
        Inverse of the matrix `A`.
    '''

    A = np.asarray(A)

    if np.any(np.isnan(A)):
        warnings.warn("Found nan in Matrix during inversion", UserWarning)
        return np.zeros_like(A) * np.nan

    try:
        eps = np.finfo(A.dtype).eps
    except:
        A = A.astype(np.float)
        eps = np.finfo(A.dtype).eps

    if np.linalg.cond(A) > 1/eps:
        if raise_error:
            raise ValueError("Found singular matrix", UserWarning)
        else:
            warnings.warn("Found singular matrix", UserWarning)
            return np.zeros_like(A) * np.nan
    else:
        return np.linalg.inv(A)

def _oeDict2Object(oeDict):
    r'''
    Helper function to convert a oe-dictionary (usually loaded from a file) to
    a pyOptimalEstimation object

    Parameters
    ----------
    oeDict : dict
      dictionary object

    Returns
    -------
    pyOptimalEstimation object
      pyOptimalEstimation object obtained from file.
    '''
    oe = optimalEstimation(
        oeDict.pop("x_vars"),
        oeDict.pop("x_a"),
        oeDict.pop("S_a"),
        oeDict.pop("y_vars"),
        oeDict.pop("y_obs"),
        oeDict.pop("S_y"),
        None
    )
    for kk in oeDict.keys():
        oe.__dict__[kk] = oeDict[kk]
    return oe

def _niceColors(length, cmap='hsv'):
    r'''
    Helper function to provide colors for plotting

    Parameters
    ----------
    length : int
      The number of required colors
    cmap : str, optional
      Matplotlib colormap. Defaults to hsv.

    Returns
    -------
    list of colorcodes
      list of colors
    '''
    colors = list()
    cm = plt.get_cmap(cmap)
    for l in range(length):
        colors.append(cm(1.*l/length))
    return colors

def _estimateChi2(S, z, atol=1e-5):
    '''Estimate Chi^2 to estimate whether z is from distribution with 
    covariance S

    Parameters
    ----------
    S : {array}
        Covariance matrix
    z : {array}
        Vector to test
        atol : float (default 1e-5)
            The absolute tolerance for comparing eigen values to zero. We 
            found that values should be than the numpy.isclose defualt value 
            of 1e-8.

    Returns
    -------
    float
        Estimated chi2 value
    '''

    eigVals, eigVecsL = scipy.linalg.eig(S, left=True, right=False)
    z_prime = eigVecsL.T.dot(z)

    # Handle singular matrices. See Rodgers ch 12.2
    notNull = np.abs(eigVals) > atol
    dofs = np.sum(notNull)
    if dofs != len(notNull):
        print('Warning. Singular Matrix with rank %i instead of %i. '
              '(This is typically safe to ignore)       ' %
              (dofs, len(notNull)))

    # Rodgers eq. 12.1
    chi2s = z_prime[notNull]**2/eigVals[notNull]
    return chi2s, dofs

def _testChi2(S, z, significance, atol=1e-5):
    '''Test whether z is from distribution with covariance S with significance

    Parameters
    ----------
    S : {array}
        Covariance matrix
    z : {array}
        Vector to test
    significance : {float}
        Significane level
        atol : float (default 1e-5)
            The absolute tolerance for comparing eigen values to zero. We 
            found that values should be than the numpy.isclose defualt value 
            of 1e-8.

    Returns
    -------
    float
        Estimated chi2 value
    float
        Theoretical chi2 value for significance
    bool
        True if Chi^2 test passed

    '''
    chi2s_obs, dof = _estimateChi2(S, z, atol=atol)
    chi2_obs = np.real_if_close(np.sum(chi2s_obs))
    chi2_theo = scipy.stats.chi2.isf(significance, dof)
    # chi2_theo1 = scipy.stats.chi2.isf(significance, 1)

    # print(chi2_obs<= chi2_theo, np.all(chi2s_obs<= chi2_theo1))

    return chi2_obs, chi2_theo
# =================================================================================================================
def Generate_ATMS(year='2016',ARMStation='McMurdo',LatRange=[-85,-70],LonRange=[-180,180]):
    global ATMS               
    #----------- ATMS --------------#    
    if (ARMStation=='McMurdo'):
        ARMStation_LAT  = -77.85
        ARMStation_LON  = 166.66
        ARMStation_month= np.array(['01','02','03','04','05','06','07','08','09','10','11','12'])
        LatRange        = [-85,-70]
        LonRange        = [160,172]
        ARMGroup        = 'ARM'        
    if (ARMStation=='WAIS'):
        ARMStation_LAT  = -79.468
        ARMStation_LON  = -112.086
        ARMStation_month= np.array(['02','03','04','05','06','07','08','09','10','11','12'])
        LatRange        = [-85,-70]
        LonRange        = [-117,-107]
        ARMGroup        = 'ARM'        
    filename='Antractica'+year+'_ATMS_'+ARMGroup+'.h5' 
    for month in ARMStation_month:
        #------------------------------------------------------------------------    
        ATMS                = {}
        ATMS['TB']          = []
        ATMS['Incid']       = []
        ATMS['Time']        = []
        ATMS['Distance']    = []
        ATMS['Height']      = []
        ATMS['Range']       = []

        time58  = datetime(1958, 1, 1, 0, 0)
        time58  = time58.timestamp()
        
        for ii in np.arange(1,10):
            fname_list  = glob.glob('Data/ATMS/'+year+'/'+month+'/GATMO-SATMS_npp_d'+year+month+'0'+str(ii)+'*_rp2.h5')
    
            if (np.size(fname_list)>0):
                for jj in range(np.size(fname_list)):
                    fname   = fname_list[jj]
                    f       = h5py.File(fname,'r')
    
                    Matrixincid  = np.copy(f['/All_Data/ATMS-SDR-GEO_All/SatelliteZenithAngle'])
                    Matrixlat  = np.copy(f['/All_Data/ATMS-SDR-GEO_All/Latitude'])
                    Matrixlon  = np.copy(f['/All_Data/ATMS-SDR-GEO_All/Longitude'])
                    Matrixtime = np.copy(f['/All_Data/ATMS-SDR_All/BeamTime'])
                    MatrixTB   = np.copy(f['/All_Data/ATMS-SDR_All/BrightnessTemperature'])
                    Matrixfactor = f['/All_Data/ATMS-SDR_All/BrightnessTemperatureFactors'][0]                
                    MatrixHeight  = np.copy(f['/All_Data/ATMS-SDR-GEO_All/Height'])
                    MatrixRange  = np.copy(f['/All_Data/ATMS-SDR-GEO_All/SatelliteRange'])
            
                    select      = (np.copy(Matrixlat)<=LatRange[1])&(np.copy(Matrixlat)>=LatRange[0])&(Matrixlon<=LonRange[1])&(Matrixlon>=LonRange[0])
                                        
                    for i in range(np.size(select.nonzero()[0])):            
                        i1  = select.nonzero()[0][i]
                        i2  = select.nonzero()[1][i]
                        tmplon = Matrixlon[i1,i2]
                        if (tmplon<0): tmplon=tmplon+360.0
                        tmplat = Matrixlat[i1,i2]
                                                
                        tmpdistance = hp.rotator.angdist( [ARMStation_LON,ARMStation_LAT] , [tmplon, tmplat],lonlat=True)*180/math.pi
                        #----------------------                                
                        if (tmpdistance<=1):
                            time = Matrixtime[i1,i2]/1e6
                            time = time+time58 # time in seconds since epoch time
                            ATMS['TB'].append(MatrixTB[i1,i2]*Matrixfactor)
                            ATMS['Incid'].append(Matrixincid[i1,i2])
                            ATMS['Time'].append(time)
                            ATMS['Distance'].append(tmpdistance)
                            ATMS['Height'].append(MatrixHeight[i1,i2])
                            ATMS['Range'].append(MatrixRange[i1,i2])
                            
                                    
        for ii in np.arange(10,32):
            fname_list  = glob.glob('Data/ATMS/'+year+'/'+month+'/GATMO-SATMS_npp_d'+year+month+str(ii)+'*_rp2.h5')
    
            if (np.size(fname_list)>0):
                for jj in range(np.size(fname_list)):
                    fname   = fname_list[jj]
                    f       = h5py.File(fname,'r')
    
                    Matrixincid  = np.copy(f['/All_Data/ATMS-SDR-GEO_All/SatelliteZenithAngle'])
                    Matrixlat  = np.copy(f['/All_Data/ATMS-SDR-GEO_All/Latitude'])
                    Matrixlon  = np.copy(f['/All_Data/ATMS-SDR-GEO_All/Longitude'])
                    Matrixtime = np.copy(f['/All_Data/ATMS-SDR_All/BeamTime'])
                    MatrixTB   = np.copy(f['/All_Data/ATMS-SDR_All/BrightnessTemperature'])
                    Matrixfactor = f['/All_Data/ATMS-SDR_All/BrightnessTemperatureFactors'][0]
                    MatrixHeight  = np.copy(f['/All_Data/ATMS-SDR-GEO_All/Height'])
                    MatrixRange  = np.copy(f['/All_Data/ATMS-SDR-GEO_All/SatelliteRange'])
            
                    select      = (np.copy(Matrixlat)<=LatRange[1])&(np.copy(Matrixlat)>=LatRange[0])&(Matrixlon<=LonRange[1])&(Matrixlon>=LonRange[0])
            
                    for i in range(np.size(select.nonzero()[0])):            
                        i1  = select.nonzero()[0][i]
                        i2  = select.nonzero()[1][i]
                        tmplon = Matrixlon[i1,i2]
                        if (tmplon<0): tmplon=tmplon+360.0
                        tmplat = Matrixlat[i1,i2]
                                                
                        tmpdistance = hp.rotator.angdist( [ARMStation_LON,ARMStation_LAT] , [tmplon, tmplat],lonlat=True)*180/math.pi                                                        
                        if (tmpdistance<=1):
                            time = Matrixtime[i1,i2]/1e6
                            time = time+time58 # time in seconds since epoch time

                            ATMS['TB'].append(MatrixTB[i1,i2]*Matrixfactor)
                            ATMS['Incid'].append(Matrixincid[i1,i2])
                            ATMS['Time'].append(time)
                            ATMS['Distance'].append(tmpdistance)
                            ATMS['Height'].append(MatrixHeight[i1,i2])
                            ATMS['Range'].append(MatrixRange[i1,i2])
                            
                                                                    
        tmp = os.path.isfile(filename)
        if (tmp==True):
            hf = h5py.File(filename, 'r+')    
        else:
            hf = h5py.File(filename, 'w')

        for ii in ATMS.keys():
            try:
                del(hf[ARMStation+'/'+month+'/ATMS_npp_'+ii])
            except:
                print('not exist')
            hf.create_dataset(ARMStation+'/'+month+'/ATMS_npp_'+ii, data=ATMS[ii])

        hf.close()
    
    return        

def Generate_MERRA2(year='2016',ARMStation='McMurdo',LatRange=[-90,-60],LonRange=[-180,180]):
    global ATMS
    global MERRA2
        
    if (ARMStation=='McMurdo'):
        ARMStation_LAT  = -77.85
        ARMStation_LON  = 166.66
        ARMStation_month= np.array(['01','02','03','04','05','06','07','08','09','10','11','12'])
        ARMStation_month= np.array(['10','11','12'])
        ARMGroup        = 'ARM'        
        
        LatRange        = [-85,-70]
        LonRange        = [160,172]
        
    if (ARMStation=='WAIS'):
        ARMStation_LAT  = -79.468
        ARMStation_LON  = -112.086
        ARMStation_month= np.array(['02','03','04','05','06','07','08','09','10','11','12'])
        LatRange        = [-85,-70]
        LonRange        = [-117,-107]
        ARMGroup        = 'ARM'        
    
    filename='/Volumes/Research2/Output/Antractica'+year+'_MERRA2_'+ARMGroup+'.h5' 
    #---------- MERRA2 -------------#    
    for month in ARMStation_month:
        print(month)
        fname1_list  = glob.glob('/Volumes/SARIS/Data/MERRA2/'+year+'/MERRA2_400.tavg3_3d_asm_Nv.'+year+month+'*.nc4')
        fname2_list  = glob.glob('/Volumes/SARIS/Data/MERRA2/'+year+'Surface/MERRA2_400.tavg1_2d_slv_Nx.'+year+month+'*.nc4')    
        if True:
            #--------- MERRA2 ----------------
            MERRA2                     = {}
            MERRA2['3D_height']        = []
            MERRA2['hmin']             = []            
            MERRA2['hmax']             = []            
    
            MERRA2['3D_pressure']      = []
            MERRA2['3D_QV']            = []
            MERRA2['3D_QI']            = []
            MERRA2['3D_QL']            = []
            MERRA2['3D_CLOUD']         = []
            MERRA2['3D_temperature']   = []
            MERRA2['3D_sealevelp']     = []
            MERRA2['3D_surfacep']      = []
            MERRA2['3D_time']          = []      
            
            for ii in range(np.size(fname1_list)):
                print(ii)
                fname   = fname1_list[ii]
                dataset = nc.Dataset(fname)
                date    = fname[(np.size(fname)-7):(np.size(fname)-5)]
                            
                tmp         = np.abs(ARMStation_LAT-dataset['lat'][:])
                select      = tmp==np.min(tmp)
                latindex    = select.nonzero()[0][0]

                tmp         = np.abs(ARMStation_LON-dataset['lon'][:])
                select      = tmp==np.min(tmp)
                lonindex    = select.nonzero()[0][0]   

                #----------------------    
                for jj in range(8):
        
                    time  = datetime(int(year), int(month), int(date), jj*3, 0)
                    time  = time.timestamp()
        
                    MERRA2['3D_pressure'].append(np.reshape(dataset['PL'][jj,:,latindex,lonindex],72))
                    MERRA2['3D_QV'].append(np.reshape(dataset['QV'][jj,:,latindex,lonindex],72))
                    MERRA2['3D_QI'].append(np.reshape(dataset['QI'][jj,:,latindex,lonindex],72))
                    MERRA2['3D_QL'].append(np.reshape(dataset['QL'][jj,:,latindex,lonindex],72))
                    MERRA2['3D_CLOUD'].append(np.reshape(dataset['CLOUD'][jj,:,latindex,lonindex],72))

                    MERRA2['3D_temperature'].append(np.reshape(dataset['T'][jj,:,latindex,lonindex],72))
                    MERRA2['3D_sealevelp'].append(np.reshape(dataset['SLP'][jj,latindex,lonindex],1))
                    MERRA2['3D_surfacep'].append(np.reshape(dataset['PS'][jj,latindex,lonindex],1))
                    MERRA2['3D_time'].append(time)
            
                    MERRA2['3D_height'].append(np.reshape(dataset['H'][jj,:,latindex,lonindex],np.shape(dataset['H'][jj,:,latindex,lonindex])[0]))
                    MERRA2['hmin'].append( np.min(np.reshape(dataset['H'][jj,:,latindex,lonindex],np.shape(dataset['H'][jj,:,latindex,lonindex])[0])) )
                    MERRA2['hmax'].append( np.max(np.reshape(dataset['H'][jj,:,latindex,lonindex],np.shape(dataset['H'][jj,:,latindex,lonindex])[0])) )                    
    
            tmp = os.path.isfile(filename)
            if (tmp==True):
                hf = h5py.File(filename, 'r+')    
            else:
                hf = h5py.File(filename, 'w')
        
            for ii in MERRA2.keys():                
                hf.create_dataset(ARMStation+'/'+month+'/MERRA2_'+ii, data=MERRA2[ii])
            hf.close()
        
        if True:
            MERRA2                      = {}
            MERRA2['2D_PS']             = []
            MERRA2['2D_QV2M']           = []
            MERRA2['2D_T2M']            = []
            MERRA2['2D_TS']             = []
            MERRA2['2D_U10M']           = []
            MERRA2['2D_V10M']           = []

            MERRA2['2D_U2M']           = []
            MERRA2['2D_V2M']           = []

            MERRA2['2D_time']           = []
            
            for ii in range(np.size(fname2_list)):
                print(ii)
                fname   = fname2_list[ii]
                dataset = nc.Dataset(fname)
        
                date = fname[(np.size(fname)-7):(np.size(fname)-5)]
                            
                tmp         = np.abs(ARMStation_LAT-dataset['lat'][:])
                select      = tmp==np.min(tmp)
                latindex    = select.nonzero()[0][0]

                tmp         = np.abs(ARMStation_LON-dataset['lon'][:])
                select      = tmp==np.min(tmp)
                lonindex    = select.nonzero()[0][0]   

                for jj in range(24):
        
                    time  = datetime(int(year), int(month), int(date), jj*1, 0)
                    time  = time.timestamp()
        
                    MERRA2['2D_PS'].append(dataset['PS'][jj,latindex,lonindex])
                    MERRA2['2D_QV2M'].append(dataset['QV2M'][jj,latindex,lonindex])
                    MERRA2['2D_T2M'].append(dataset['T2M'][jj,latindex,lonindex])
                    MERRA2['2D_TS'].append(dataset['TS'][jj,latindex,lonindex])

                    MERRA2['2D_U10M'].append(dataset['U10M'][jj,latindex,lonindex])
                    MERRA2['2D_V10M'].append(dataset['V10M'][jj,latindex,lonindex])

                    MERRA2['2D_U2M'].append(dataset['U2M'][jj,latindex,lonindex])
                    MERRA2['2D_V2M'].append(dataset['V2M'][jj,latindex,lonindex])

                    MERRA2['2D_time'].append(time)
            
            tmp = os.path.isfile(filename)
            if (tmp==True):
                hf = h5py.File(filename, 'r+')    
            else:
                hf = h5py.File(filename, 'w')

            for ii in MERRA2.keys():                
                hf.create_dataset(ARMStation+'/'+month+'/MERRA2_'+ii, data=MERRA2[ii])        
            hf.close()
        
            
    return        
# =================================================================================================================
def forwardRT( X, pressure=None, zenithAngleATMS=0, latitude=0, longitude=0, date=np.array([2015, 8, 1, 0, 0, 0]), surfacepressure=1013., surfacetemperature=0.263178E+03, snow_frac=0.0, surfaceType=np.array([0, 1]), FASTEM='New ice (snow)', temperature2M=None, humidity2M=None, windu2M=None, windv2M=None, skintemperature=None, emiss=None  ):
    global ErrorRTTOV
    
    temperature, humidity = splitTQ(X)
    
    humidity    = 10.00**np.copy(humidity)
    temperature = np.copy(temperature)
    humidity    = humidity[::-1]
    temperature = temperature[::-1]
    
    pressure    = pressure/1e2    
    gas_units = 1 # kg/kg over moist air (default)
    
    # Declare an instance of Profiles
    nlevels = len(pressure)
    nprofiles = 1
    myProfilesATMS  = pyrttov.Profiles(nprofiles, nlevels)

    # Associate the profiles and other data from example_data.h with myProfiles
    # Note that the simplecloud, clwscheme, icecloud and zeeman data are not mandatory and
    # are omitted here
    #------- ATMS
    if True:
        myProfilesATMS.P    = np.empty((1,nlevels))
        myProfilesATMS.T    = np.empty((1,nlevels))
        myProfilesATMS.Q    = np.empty((1,nlevels))
        myProfilesATMS.CO2  = np.zeros((1,nlevels))
        myProfilesATMS.P[0,:] = np.copy(pressure) # from top to bottom
        myProfilesATMS.T[0,:] = np.copy(temperature)
        myProfilesATMS.Q[0,:] = np.copy(humidity)

        myProfilesATMS.Angles       = np.zeros((1,4))
        myProfilesATMS.S2m          = np.zeros((1,6))
        myProfilesATMS.Skin         = np.zeros((1,9))
        myProfilesATMS.SurfType     = np.zeros((1,2))
        myProfilesATMS.SurfGeom     = np.zeros((1,3))
        myProfilesATMS.DateTimes    = np.zeros((1,6))

        # angles[4][nprofiles]: satzen, satazi, sunzen, sunazi
        myProfilesATMS.Angles[0,:]      = np.array([zenithAngleATMS, 0.0, 0.0, 0.0])
        # s2m[6][nprofiles]: 2m p, 2m t, 2m q, 10m wind u, v, wind fetch
        myProfilesATMS.S2m[0,:]         = np.array([surfacepressure/1e2, temperature2M, humidity2M, windu2M, windv2M, 100000.])
    
        # skin[9][nprofiles]: skin T, salinity, snow_frac, foam_frac, fastem_coefsx5
        if (FASTEM=='Default'):
            myProfilesATMS.Skin[0,:]        = np.array([skintemperature, 35., snow_frac, 0., 3.0, 5.0, 15.0, 0.1, 0.3])
        #----- Winter surface type
        if (FASTEM=='Forest'):
            myProfilesATMS.Skin[0,:]        = np.array([skintemperature, 35., snow_frac, 0., 1.7, 1.0, 163.0, 0.0, 0.5])
        if (FASTEM=='Open grass'):
            myProfilesATMS.Skin[0,:]        = np.array([skintemperature, 35., snow_frac, 0., 2.2, 1.3, 138.0, 0.0, 0.42])
        if (FASTEM=='Bare soil'):
            myProfilesATMS.Skin[0,:]        = np.array([skintemperature, 35., snow_frac, 0., 2.3, 1.9, 21.8, 0.0, 0.5])    
        #----- Winter surface type
        if (FASTEM=='Forest and snow'):
            myProfilesATMS.Skin[0,:]        = np.array([skintemperature, 35., snow_frac, 0., 2.9, 3.4, 27.0, 0.0, 0.0])
        if (FASTEM=='Deep dry snow'):
            myProfilesATMS.Skin[0,:]        = np.array([skintemperature, 35., snow_frac, 0., 3.0, 24.0, 60.0, 0.1, 0.15])
        if (FASTEM=='Frozen soil'):
            myProfilesATMS.Skin[0,:]        = np.array([skintemperature, 35., snow_frac, 0., 117.8, 2.0, 0.19, 0.2 ,0.35])    
        #----- Sea ice
        if (FASTEM=='Grease ice'):
            myProfilesATMS.Skin[0,:]        = np.array([skintemperature, 35., snow_frac, 0., 23.7, 7.7, 17.3, 0.0, 0.15])
        if (FASTEM=='Baltic nilas'):
            myProfilesATMS.Skin[0,:]        = np.array([skintemperature, 35., snow_frac, 0., 1.6, 3.3, 2.2, 0.0, 0.0])
        if (FASTEM=='New ice (no snow)'):
            myProfilesATMS.Skin[0,:]        = np.array([skintemperature, 35., snow_frac, 0., 2.9, 3.4, 27.0, 0.0, 0.0])
        if (FASTEM=='New ice (snow)'):
            myProfilesATMS.Skin[0,:]        = np.array([skintemperature, 35., snow_frac, 0., 2.2, 3.7, 122.0, 0.0, 0.15])
        if (FASTEM=='Brash ice'):
            myProfilesATMS.Skin[0,:]        = np.array([skintemperature, 35., snow_frac, 0., 3.0, 5.5, 183.0, 0.0, 0.0])
        if (FASTEM=='Compact pack ice'):
            myProfilesATMS.Skin[0,:]        = np.array([skintemperature, 35., snow_frac, 0., 2.0, 1700000.0, 49000000.0, 0.0, 0.0])
        if (FASTEM=='Fast ice'):
            myProfilesATMS.Skin[0,:]        = np.array([skintemperature, 35., snow_frac, 0., 1.5, 77.8, 703.0, 0.1, 0.35])
        if (FASTEM=='Lake ice + snow'):
            myProfilesATMS.Skin[0,:]        = np.array([skintemperature, 35., snow_frac, 0., 1.8, 67.1, 534.0, 0.1, 0.15])
        if (FASTEM=='Multi-year ice'):
            myProfilesATMS.Skin[0,:]        = np.array([skintemperature, 35., snow_frac, 0., 1.5, 85000.0, 4700000.0, 0.0, 0.0])
    
        # surftype[2][nprofiles]: surftype, watertype
        # surface type: 0=land, 1=sea, 2=sea-ice
        # for water surfaces: 0=fresh water, 1=ocean
        myProfilesATMS.SurfType[0,:]    = surfaceType
        # surfgeom[3][nprofiles]: lat, lon, elev
        myProfilesATMS.SurfGeom[0,:]    = np.array([latitude, longitude, 0.])
        # datetimes[6][nprofiles]: yy, mm, dd, hh, mm, ss
        myProfilesATMS.DateTimes[0,:]   = date
        

    # ------------------------------------------------------------------------
    # Set up Rttov instances for each instrument
    # ------------------------------------------------------------------------
    # Create three Rttov objects for three instruments
    ATMSRttov = pyrttov.Rttov()
    nchan_ATMS = 22

    # Set the options for each Rttov instance:
    # - the path to the coefficient file must always be specified
    # - turn RTTOV interpolation on (because input pressure levels differ from
    #   coefficient file levels)
    # - set the verbose_wrapper flag to true so the wrapper provides more
    #   information
    # - enable solar simulations for SEVIRI
    # - enable CO2 simulations for HIRS (the CO2 profiles are ignored for
    #   the SEVIRI and MHS simulations)
    # - enable the store_trans wrapper option for MHS to provide access to
    #   RTTOV transmission structure
    ATMSRttov.FileCoef = 'rttov13/rtcoef_rttov13/rttov13pred54L/rtcoef_jpss_0_atms.dat'
    ATMSRttov.Options.AddInterp = True
    ATMSRttov.Options.StoreTrans = True
    ATMSRttov.Options.VerboseWrapper = True

    # Load the instruments: for HIRS and MHS do not supply a channel list and
    # so read all channels
        
    try:        
        ATMSRttov.loadInst()
    except pyrttov.RttovError as e:
        sys.stderr.write("Error loading instrument(s): {!s}".format(e))
        ErrorRTTOV = True

    if (ErrorRTTOV==False):    
        # Associate the profiles with each Rttov instance
        ATMSRttov.Profiles = myProfilesATMS

        # ------------------------------------------------------------------------
        # Load the emissivity and BRDF atlases
        # ------------------------------------------------------------------------

        # Load the emissivity and BRDF atlases:
        # - load data for the month in the profile data
        # - load the IR emissivity atlas data for multiple instruments so it can be used for SEVIRI and HIRS
        # - SEVIRI is the only VIS/NIR instrument we can use the single-instrument initialisation for the BRDF atlas
        # TELSEM2 atlas does not require an Rttov object to initialise    
        
        # Set up the surface emissivity/reflectance arrays and associate with the Rttov objects
        surfemisrefl_ATMS = np.zeros((5,nprofiles,nchan_ATMS), dtype=np.float64)
        ATMSRttov.SurfEmisRefl  = surfemisrefl_ATMS

        # ------------------------------------------------------------------------
        # Call RTTOV
        # ------------------------------------------------------------------------

        # Surface emissivity/reflectance arrays must be initialised *before every call to RTTOV*
        # Negative values will cause RTTOV to supply emissivity/BRDF values (i.e. equivalent to
        # calcemis/calcrefl TRUE - see RTTOV user guide)
    
        ATMSRttov.SurfEmisRefl[0,0,:]     = emiss

        # Call the RTTOV direct model for each instrument:
        # no arguments are supplied to runDirect so all loaded channels are
        # simulated
    
        try:
            ATMSRttov.runDirect()
        except pyrttov.RttovError as e:
            sys.stderr.write("Error running RTTOV direct model: {!s}".format(e))
            ErrorRTTOV = True
        # ------------------------------------------------------------------------
        # Print out some of the output
        # ------------------------------------------------------------------------
            
    if (ErrorRTTOV == False):
        TB     = ATMSRttov.BtRefl[0, :]  
    else:
        TB      = np.zeros(22)      
                
    return TB

def read_prior(fname,year,inputmonth,ARMStation,nlevel=30,appendix=40):
    global Reference, Truth, Errorprior
    
    month_list = [inputmonth]
    #----- from hdf5 to nc -----#
    if True:        
        hf = h5py.File(fname, 'r')      
        
        MERRA2['3D_height']        = []
        MERRA2['3D_pressure']      = []
        MERRA2['3D_QV']            = []
        MERRA2['3D_temperature']   = []
        MERRA2['3D_time']          = []
        MERRA2['3D_surfacep']      = []
        
        for month in month_list:
            MERRA2['3D_height']         = MERRA2['3D_height'] + list(np.copy(hf[ARMStation+'/'+month+'/MERRA2_3D_height']))
            MERRA2['3D_pressure']       = MERRA2['3D_pressure'] + list(np.copy(hf[ARMStation+'/'+month+'/MERRA2_3D_pressure']))
            MERRA2['3D_QV']             = MERRA2['3D_QV'] + list(np.copy(hf[ARMStation+'/'+month+'/MERRA2_3D_QV']))
            MERRA2['3D_temperature']    = MERRA2['3D_temperature'] + list(np.copy(hf[ARMStation+'/'+month+'/MERRA2_3D_temperature']))

            MERRA2['3D_time']           = MERRA2['3D_time'] + list(np.copy(hf[ARMStation+'/'+month+'/MERRA2_3D_time']))
            MERRA2['3D_surfacep']       = MERRA2['3D_surfacep'] + list(np.copy(hf[ARMStation+'/'+month+'/MERRA2_3D_surfacep']))
            
        MERRA2['3D_height']        = np.array(MERRA2['3D_height'])
        MERRA2['3D_pressure']      = np.array(MERRA2['3D_pressure'])
        MERRA2['3D_QV']            = np.array(MERRA2['3D_QV'])
        MERRA2['3D_temperature']   = np.array(MERRA2['3D_temperature'])
        MERRA2['3D_time']          = np.array(MERRA2['3D_time'])
        MERRA2['3D_surfacep']      = np.array(MERRA2['3D_surfacep'])
        
        sorta = np.argsort(MERRA2['3D_time'])
        MERRA2['3D_height']         = MERRA2['3D_height'][sorta,:]
        MERRA2['3D_pressure']       = MERRA2['3D_pressure'][sorta,:]
        MERRA2['3D_QV']             = MERRA2['3D_QV'][sorta,:]
        MERRA2['3D_temperature']    = MERRA2['3D_temperature'][sorta,:]
        MERRA2['3D_time']           = MERRA2['3D_time'][sorta]
        MERRA2['3D_surfacep']       = MERRA2['3D_surfacep'][sorta]
        MERRA2['3D_surfaceH']       = np.copy(MERRA2['3D_surfacep'])
        MERRA2['3D_surfaceH2M']     = np.copy(MERRA2['3D_surfacep'])
        MERRA2['3D_surfacep2M']     = np.copy(MERRA2['3D_surfacep'])
        
        for i in range(MERRA2['3D_time'].size):
            func                        = interpolate.interp1d(np.log10(MERRA2['3D_pressure'][i,:]),MERRA2['3D_height'][i,:],fill_value='extrapolate')
            MERRA2['3D_surfaceH'][i]    = func(np.log10(MERRA2['3D_surfacep'][i]))
            
            MERRA2['3D_surfaceH2M'][i]  = MERRA2['3D_surfaceH'][i]+2.0
            func                        = interpolate.interp1d(MERRA2['3D_height'][i,:],np.log10(MERRA2['3D_pressure'][i,:]),fill_value='extrapolate')
            MERRA2['3D_surfacep2M'][i]  = 10.0**func(MERRA2['3D_surfaceH2M'][i])
            
        #----------------------------------------------------------------------------
        #----------------------------------------------------------------------------
        #----------------------------------------------------------------------------
        MERRA2['2D_PS']             = []
        MERRA2['2D_QV2M']           = []
        MERRA2['2D_T2M']            = []
        MERRA2['2D_time']           = []
        for month in month_list:
            
            MERRA2['2D_PS']         = MERRA2['2D_PS'] + list(np.copy(hf[ARMStation+'/'+month+'/MERRA2_2D_PS']))
            MERRA2['2D_QV2M']       = MERRA2['2D_QV2M'] + list(np.copy(hf[ARMStation+'/'+month+'/MERRA2_2D_QV2M']))
            MERRA2['2D_T2M']        = MERRA2['2D_T2M'] + list(np.copy(hf[ARMStation+'/'+month+'/MERRA2_2D_T2M']))
            MERRA2['2D_time']       = MERRA2['2D_time'] + list(np.copy(hf[ARMStation+'/'+month+'/MERRA2_2D_time']))
        
        MERRA2['2D_time']       = np.array(MERRA2['2D_time'])
        MERRA2['2D_PS']         = np.array(MERRA2['2D_PS'])
        MERRA2['2D_QV2M']       = np.array(MERRA2['2D_QV2M'])
        MERRA2['2D_T2M']        = np.array(MERRA2['2D_T2M'])
        
        sorta = np.argsort(MERRA2['2D_time'])
        MERRA2['2D_time']       = MERRA2['2D_time'][sorta]
        MERRA2['2D_PS']         = MERRA2['2D_PS'][sorta]
        MERRA2['2D_QV2M']       = MERRA2['2D_QV2M'][sorta]
        MERRA2['2D_T2M']        = MERRA2['2D_T2M'][sorta]
        #----------------------------------------------------------------------------
        func = interpolate.interp1d(MERRA2['2D_time'],MERRA2['2D_T2M'],fill_value='extrapolate',bounds_error=False)
        MERRA2['3D_surfaceT'] = func(MERRA2['3D_time'])
        func = interpolate.interp1d(MERRA2['2D_time'],MERRA2['2D_QV2M'],fill_value='extrapolate',bounds_error=False)
        MERRA2['3D_surfaceQ'] = func(MERRA2['3D_time'])
        #----------------------------------------------------------------------------        
        nlevel  = 72
        height  = np.zeros(nlevel)    
        for i in range(nlevel):
            height[i] = np.median(MERRA2['3D_height'][:,i])
        height  = list(height) + [np.median(MERRA2['3D_surfaceH2M'])]
        height  = np.array(height)
        nlevel  = nlevel + 1
        #----------------------------------------------------------------------------        
        profileTT = np.ones(MERRA2['3D_time'].size,dtype='datetime64[ns]')
        for ii in range(MERRA2['3D_time'].size):
            profileTT[ii] = np.datetime64(datetime.fromtimestamp(MERRA2['3D_time'][ii]))
        profileTT = np.array(profileTT,dtype='datetime64[ns]')
        #------------------
        prior = xr.open_dataset('pyOptimalEstimation_examples-master/data/radiosonde_climatology_nsa_2002-2020.nc')  
                
        temperature = np.zeros((MERRA2['3D_time'].size,nlevel))
        QV          = np.zeros((MERRA2['3D_time'].size,nlevel))
        pressure    = np.zeros((MERRA2['3D_time'].size,nlevel))
        
        for ii in range(MERRA2['3D_time'].size):
            
            tmpx            = list(np.copy(MERRA2['3D_height'][ii])) + [float(MERRA2['3D_surfaceH2M'][ii])] 
            tmpx            = np.array(tmpx)

            tmpy            = list(np.copy(MERRA2['3D_pressure'][ii])) + [float(MERRA2['3D_surfacep2M'][ii])] 
            tmpy            = np.array(tmpy)            
            
            func            = interpolate.interp1d(tmpx,np.log10(tmpy),fill_value='extrapolate',bounds_error=False)
            pressure[ii]    = 10.0**func(height)
            #--------
            tmpy            = list(np.copy(MERRA2['3D_temperature'][ii])) + [float(MERRA2['3D_surfaceT'][ii])] 
            tmpy            = np.array(tmpy)            
            func            = interpolate.interp1d(tmpx,tmpy,fill_value='extrapolate',bounds_error=False)
            temperature[ii] = func(height)            
            #--------
            tmpy            = list(np.copy(MERRA2['3D_QV'][ii])) + [float(MERRA2['3D_surfaceQ'][ii])] 
            tmpy            = np.array(tmpy)            
            func            = interpolate.interp1d(tmpx,np.log10(tmpy),fill_value='extrapolate',bounds_error=False)
            QV[ii]          = 10.0**func(height)            
                
        
        Reference['temperature'] = np.mean(temperature,axis=0)
        Reference['QV'] = np.mean(QV,axis=0)
        
        ds = xr.Dataset(
             data_vars=dict(
                 temp=(["time", "height"], temperature),
                 q=(["time", "height"], QV),
                 bar_pres=(["time", "height"], pressure),
             ),
             coords=dict(
                 time=profileTT,
                 height=height,
             ),
         )
        
        ds.to_netcdf(path='Prior2020Temp'+str(appendix)+'.nc')
            
    #--------------------------------------------------------------------
    fname='Prior2020Temp'+str(appendix)+'.nc'    
    prior = xr.open_dataset(fname)
    prior['q'] = np.log10(prior['q'])
    
    profileTT = np.array([year+'-'+inputmonth+'-15T00:00:00.000000000'],dtype='datetime64[ns]')
        
    profiles = prior.sel(time=profileTT)
    nLev = len(prior.height)
        
    #--------------------------------------------------------------------
    priors = {'all': prior}
    prior_qts = {}
    for season in priors.keys():
        prior_q = priors[season]['q'].to_pandas()
        prior_q.columns = ['%05i_q' % (i) for i in prior_q.columns]
        prior_t = priors[season]['temp'].to_pandas()
        prior_t.columns = ['%05i_t' % (i) for i in prior_t.columns]
        prior_qts[season] = pn.concat((prior_q, prior_t), axis=1)
        prior_qts[season] = prior_qts[season].reindex(sorted(prior_qts[season].columns), axis=1)
    #--------------------------------------------------------------------
    x_cov = []
    x_mean = []

    for season in ['all']:
        x_cov1 = prior_qts[season].cov().rename_axis('state', axis=0).rename_axis('stateT', axis=1)
        x_mean1 = prior_qts[season].mean().rename_axis('state', axis=0)
                
        x_cov.append(xr.DataArray(x_cov1))
        x_mean.append(xr.DataArray(x_mean1))
            
    x_cov = xr.concat(x_cov, dim='season')
    x_mean = xr.concat(x_mean, dim='season')

    x_cov['season'] = ['all']
    x_mean['season'] = ['all']
    
    Errorprior = True

    for season in x_cov.season:
        if (np.linalg.matrix_rank(x_cov.sel(season=season).to_pandas()) !=  x_cov.shape[-1]):
            Errorprior = False
        
    if (Errorprior==True):
        #--------------------------------------------------------------------
        height = priors['all'].height

        pressure = []
        for k in ['all']:

            pressure.append(priors[k].bar_pres.mean('time'))
        pressure = xr.concat(pressure, dim='season')
        pressure['season'] = ['all'] 
        pressure = pressure.to_pandas()
    
        Reference['x_cov']      = x_cov
        Reference['x_mean']     = x_mean
        Reference['height']     = height
        Reference['pressure']   = pressure
        #--------------------------------------------------------------------
        profiles_t = profiles['temp'].to_pandas()
        profiles_t.columns = ['%05i_t' % (i) for i in profiles_t.columns]
        profiles_q = profiles['q'].to_pandas()
        profiles_q.columns = ['%05i_q' % (i) for i in profiles_q.columns]    
    
        x_truths = pn.concat((profiles_t, profiles_q), 1)
        x_truths = x_truths.reindex(sorted(x_truths.index), axis=0)
        x_truths.columns.name = 'state'
        x_truths.index.name = 'time'

        assert np.all(np.isfinite(x_truths))
    
        season = 'all'  #Spring: March April May
        profile = np.datetime64(year+'-'+inputmonth+'-15T00:00:00.000000000')
        x_truth = x_truths.loc[profile]
        Truth['x_truth'] = x_truth
                        
    return         

def runRetrieval(year='2016',month='06',ARMStation='McMurdo',appendix=1,E0=0.8):
    global ATMS
    global MERRA2
    global forwardKwArgs
    global ErrorRTTOV    
    
    filename = year+'/'+month+'/'+ARMStation    
    tmp = os.path.exists(filename)
    if (tmp==False):
        os.mkdir(year+'/'+month+'/'+ARMStation)
    
    temp_time0  = datetime(int(year),int(month),1,0,0).timestamp()   
    error       = np.array([0.7, 0.8, 0.9, 0.7, 0.7,0.7, 0.7, 0.7, 0.7, 0.75, 1.2, 1.2, 1.5, 2.4, 3.6, 0.5, 0.6, 0.8, 0.8, 0.8, 0.8, 0.9])
         
    #---- Corresponding Station or Feature Location
    if True:
        if (ARMStation=='Ross Ice Shelf'):
            ARMStation_LAT  = -82.6853762107639
            ARMStation_LON  = -153.0
            ARMGroup        = 'IGRA'    
        if (ARMStation=='McMurdo'):
            ARMStation_LAT  = -77.85
            ARMStation_LON  = 166.66
            ARMGroup        = 'ARM'        
        if (ARMStation=='WAIS'):
            ARMStation_LAT  = -79.468
            ARMStation_LON  = -112.086
            ARMGroup        = 'ARM'        
        if (ARMStation=='NEUMAYER'):
            ARMStation_LAT  = -70.6667
            ARMStation_LON  = -8.25
            ARMGroup        = 'IGRA'        
        if (ARMStation=='AMUNDSEN-SCOTT'):
            ARMStation_LAT  = -90.0
            ARMStation_LON  = 0.0
            ARMGroup        = 'IGRA'        
        if (ARMStation=='HALLEY'):
            ARMStation_LAT  = -75.445
            ARMStation_LON  = -26.2181
            ARMGroup        = 'IGRA'        
        if (ARMStation=='BASE MARAMBIO'):
            ARMStation_LAT  = -64.2333
            ARMStation_LON  = -56.7167
            ARMGroup        = 'IGRA'
        if (ARMStation=='ROTHERA'):
            ARMStation_LAT  = -67.5661
            ARMStation_LON  = -68.1297
            ARMGroup        = 'IGRA'
        if (ARMStation=='NOVOLAZAREVSKAJA'):
            ARMStation_LAT  = -70.7678
            ARMStation_LON  = 11.8317
            ARMGroup        = 'IGRA'
        if (ARMStation=='SYOWA'):
            ARMStation_LAT  = -69.0053
            ARMStation_LON  = 39.5811
            ARMGroup        = 'IGRA'
        if (ARMStation=='MAWSON'):
            ARMStation_LAT  = -67.6017
            ARMStation_LON  = 62.8753
            ARMGroup        = 'IGRA'
        if (ARMStation=='DAVIS'):
            ARMStation_LAT  = -68.5744
            ARMStation_LON  = 77.9672
            ARMGroup        = 'IGRA'
        if (ARMStation=='MIRNYJ'):
            ARMStation_LAT  = -66.5519
            ARMStation_LON  = 93.0147
            ARMGroup        = 'IGRA'
        if (ARMStation=='CASEY'):
            ARMStation_LAT  = -66.2825
            ARMStation_LON  = 110.5231
            ARMGroup        = 'IGRA'
        if (ARMStation=='CONCORDIA'):
            ARMStation_LAT  = -75.1017
            ARMStation_LON  = 123.4119
            ARMGroup        = 'IGRA'
        if (ARMStation=='DUMONT D URVILLE'):
            ARMStation_LAT  = -66.6631
            ARMStation_LON  = 140.0011
            ARMGroup        = 'IGRA'
        if (ARMStation=='MARIO ZUCHELLI STATION'):
            ARMStation_LAT  = -74.6958
            ARMStation_LON  = 164.0922
            ARMGroup        = 'IGRA'    
        if (ARMStation=='Ronne_Ice_Shelf'):
            ARMStation_LAT  = -78.3
            ARMStation_LON  = -61.0
            ARMGroup        = 'Features'        
        if (ARMStation=='Filchner_Ice_Shelf'):
            ARMStation_LAT  = -79.0
            ARMStation_LON  = -40.0
            ARMGroup        = 'Features'        
        if (ARMStation=='Larsen_Ice_Shelf'):
            ARMStation_LAT  = -67.5
            ARMStation_LON  = -62.5
            ARMGroup        = 'Features'        
        if (ARMStation=='GeorgeVI_Ice_Shelf'):
            ARMStation_LAT  = -71.75
            ARMStation_LON  = -68
            ARMGroup        = 'Features'        
        if (ARMStation=='Wilkins_Ice_Shelf'):
            ARMStation_LAT  = -70.5
            ARMStation_LON  = -72.5
            ARMGroup        = 'Features'        
        if (ARMStation=='Shackelton_Ice_Shelf'):
            ARMStation_LAT  = -66.0311
            ARMStation_LON  = 100.0934
            ARMGroup        = 'Features'        
        if (ARMStation=='West_Ice_Shelf'):
            ARMStation_LAT  = -66.6667
            ARMStation_LON  = 85
            ARMGroup        = 'Features'        
        if (ARMStation=='Amery_Ice_Shelf'):
            ARMStation_LAT  = -70
            ARMStation_LON  = 71        
            #ARMStation_LAT  = -68
            #ARMStation_LON  = 74
            ARMGroup        = 'Features'        
        if (ARMStation=='Fimbul_Ice_Shelf'):
            ARMStation_LAT  = -71
            ARMStation_LON  = 0.4895        
            #ARMStation_LAT  = -73.1307
            #ARMStation_LON  = 0.4895
            ARMGroup        = 'Features'        
        if (ARMStation=='Riiser_larsen_Ice_Shelf'):
            ARMStation_LAT  = -72.667
            ARMStation_LON  = -16
            ARMGroup        = 'Features'        
        if (ARMStation=='DomeA'):
            ARMStation_LAT  = -80.37
            ARMStation_LON  = 77.37
            ARMGroup        = 'Features'        
        if (ARMStation=='DomeC'):
            ARMStation_LAT  = -75.1
            ARMStation_LON  = 123.38
            ARMGroup        = 'Features'        
        if (ARMStation=='DomeTalos'):
            ARMStation_LAT  = -72.78
            ARMStation_LON  = 159.07
            ARMGroup        = 'Features'        
        if (ARMStation=='DomeTitan'):
            ARMStation_LAT  = -88.499838
            ARMStation_LON  = 165.00406
            ARMGroup        = 'Features'        
        if (ARMStation=='DomeFuji'):
            ARMStation_LAT  = -77.5
            ARMStation_LON  = 37.5
            ARMGroup        = 'Features'        
        if (ARMStation=='LakeVostok'):
            ARMStation_LAT  = -77.5
            ARMStation_LON  = 106
            ARMGroup        = 'Features'    
    #---- Read ATMS Observations    
    if True:
        hf = h5py.File('Antractica'+year+'_ATMS_'+ARMGroup+'.h5', 'r')
        ATMS['j01_Incid']       = np.copy(hf[ARMStation+'/'+month+'/ATMS_npp_Incid'])    
        ATMS['j01_TB']          = np.copy(hf[ARMStation+'/'+month+'/ATMS_npp_TB'])    
        ATMS['j01_Time']        = np.copy(hf[ARMStation+'/'+month+'/ATMS_npp_Time'])    
        ATMS['j01_Distance']    = np.copy(hf[ARMStation+'/'+month+'/ATMS_npp_Distance'])                   
        ATMS['j01_Distance']    = np.copy(ATMS['j01_Distance'][:,0]) 
        ATMS['j01_Time_DOM']= (ATMS['j01_Time'] - temp_time0)/60.0/60.0/24.0
        ATMS['Freq']        = np.array([23.8000,  31.4000,  50.3000,  51.7600,  52.8000 , 53.6000,  54.4000,  54.9500,  55.5000,  57.2900,  57.2900,  57.2900,  57.2900,  57.2900 , 57.2900 , 88.2000, 165.5000 ,183.3100 ,183.3100 ,183.3100, 183.3100, 183.3100])  
        hf.close()

        sorta = np.argsort(ATMS['j01_Time_DOM'])
        ATMS['j01_Time_DOM']    = ATMS['j01_Time_DOM'][sorta]
        ATMS['j01_Incid']       = ATMS['j01_Incid'][sorta]
        ATMS['j01_TB']          = ATMS['j01_TB'][sorta,:]
        ATMS['j01_Time']        = ATMS['j01_Time'][sorta]
        ATMS['j01_Distance']    = ATMS['j01_Distance'][sorta]

        index           = np.arange(ATMS['j01_Time_DOM'].size)
        tmp_i0          = 0
        selected_index  = []

        for i in np.arange(1,ATMS['j01_Time_DOM'].size):
            if ((ATMS['j01_Time_DOM'][i]-ATMS['j01_Time_DOM'][i-1])>1e-4):
                tmp_i1  = i-1
                select = (index>=tmp_i0)&(index<=tmp_i1)&(ATMS['j01_Distance']==np.min(ATMS['j01_Distance'][tmp_i0:(tmp_i1+1)]))
                select = select.nonzero()[0][0]
                selected_index.append(select)
                tmp_i0      = i

        tmp_i1  = ATMS['j01_Time_DOM'].size-1
        select = (index>=tmp_i0)&(index<=tmp_i1)&(ATMS['j01_Distance']==np.min(ATMS['j01_Distance'][tmp_i0:(tmp_i1+1)]))
        select = select.nonzero()[0][0]
        selected_index.append(select)
        selected_index = np.array(selected_index)

        ATMS_Input_Time_DOM    = np.copy(ATMS['j01_Time_DOM'][selected_index])
        ATMS_Input_Incid       = np.copy(ATMS['j01_Incid'][selected_index])
        ATMS_Input_TB          = np.copy(ATMS['j01_TB'][selected_index,:])
        ATMS_Input_Time        = np.copy(ATMS['j01_Time'][selected_index])
        ATMS_Input_Distance    = np.copy(ATMS['j01_Distance'][selected_index])    
    #---- Read MERRA2 Data for Surface Properties 
    if True:
        hf = h5py.File('Antractica'+year+'_MERRA2_'+ARMGroup+'.h5', 'r')   
        
        MERRA2['3D_height']         = np.copy(hf[ARMStation+'/'+month+'/MERRA2_3D_height'])
        MERRA2['3D_pressure']       = np.copy(hf[ARMStation+'/'+month+'/MERRA2_3D_pressure'])
        MERRA2['3D_QV']             = np.copy(hf[ARMStation+'/'+month+'/MERRA2_3D_QV'])
        MERRA2['3D_temperature']    = np.copy(hf[ARMStation+'/'+month+'/MERRA2_3D_temperature'])
        MERRA2['3D_surfacep']       = np.copy(hf[ARMStation+'/'+month+'/MERRA2_3D_surfacep'])
        MERRA2['3D_time']           = np.copy(hf[ARMStation+'/'+month+'/MERRA2_3D_time'])
        MERRA2['3D_time_DOM']       = (MERRA2['3D_time']-temp_time0)/60.0/60.0/24.0
        MERRA2['3D_sealevelp']      = np.copy(hf[ARMStation+'/'+month+'/MERRA2_3D_sealevelp'])
        
        sorta = np.argsort(MERRA2['3D_time'])
        MERRA2['3D_time']           = MERRA2['3D_time'][sorta]
        MERRA2['3D_surfacep']       = MERRA2['3D_surfacep'][sorta]
        MERRA2['3D_sealevelp']      = MERRA2['3D_sealevelp'][sorta]
        MERRA2['3D_height']         = MERRA2['3D_height'][sorta,:]
        MERRA2['3D_pressure']       = MERRA2['3D_pressure'][sorta,:]
        MERRA2['3D_QV']             = MERRA2['3D_QV'][sorta,:]
        MERRA2['3D_temperature']    = MERRA2['3D_temperature'][sorta,:]
        
        MERRA2['3D_surfaceH']       = np.copy(MERRA2['3D_surfacep'])
        MERRA2['3D_sealevelH']      = np.copy(MERRA2['3D_sealevelp'])
        
        for i in range(MERRA2['3D_sealevelp'].size):
            func                        = interpolate.interp1d(np.log10(MERRA2['3D_pressure'][i,:]),MERRA2['3D_height'][i,:],fill_value='extrapolate')
            MERRA2['3D_surfaceH'][i]    = func(np.log10(MERRA2['3D_surfacep'][i]))
            MERRA2['3D_sealevelH'][i]   = func(np.log10(MERRA2['3D_sealevelp'][i]))
                                             
        MERRA2['2D_PS']             = np.copy(hf[ARMStation+'/'+month+'/MERRA2_2D_PS'])
        MERRA2['2D_QV2M']           = np.copy(hf[ARMStation+'/'+month+'/MERRA2_2D_QV2M'])
        MERRA2['2D_T2M']            = np.copy(hf[ARMStation+'/'+month+'/MERRA2_2D_T2M'])
        MERRA2['2D_TS']             = np.copy(hf[ARMStation+'/'+month+'/MERRA2_2D_TS'])
        MERRA2['2D_U10M']           = np.copy(hf[ARMStation+'/'+month+'/MERRA2_2D_U10M'])
        MERRA2['2D_V10M']           = np.copy(hf[ARMStation+'/'+month+'/MERRA2_2D_V10M'])
        MERRA2['2D_U2M']            = np.copy(hf[ARMStation+'/'+month+'/MERRA2_2D_U2M'])
        MERRA2['2D_V2M']            = np.copy(hf[ARMStation+'/'+month+'/MERRA2_2D_V2M'])
        MERRA2['2D_time']           = np.copy(hf[ARMStation+'/'+month+'/MERRA2_2D_time'])
        MERRA2['2D_time_DOM']       = (MERRA2['2D_time']-temp_time0)/60.0/60.0/24.0

        sorta = np.argsort(MERRA2['2D_time'])
        MERRA2['2D_time']           = MERRA2['2D_time'][sorta]
        MERRA2['2D_PS']             = MERRA2['2D_PS'][sorta]
        MERRA2['2D_QV2M']           = MERRA2['2D_QV2M'][sorta]
        MERRA2['2D_T2M']            = MERRA2['2D_T2M'][sorta]
        MERRA2['2D_TS']             = MERRA2['2D_TS'][sorta]
        MERRA2['2D_U10M']           = MERRA2['2D_U10M'][sorta]
        MERRA2['2D_V10M']           = MERRA2['2D_V10M'][sorta]
        MERRA2['2D_U2M']            = MERRA2['2D_U2M'][sorta]
        MERRA2['2D_V2M']            = MERRA2['2D_V2M'][sorta]

        hf.close()   
        
        if True:
            #-----------------------------------
            snow_frac           = np.zeros(np.size(ATMS_Input_Time))
            skintemperature     = np.zeros(np.size(ATMS_Input_Time))
            surfacetemperature  = np.zeros(np.size(ATMS_Input_Time))
            temperature2M       = np.zeros(np.size(ATMS_Input_Time))
            humidity2M          = np.zeros(np.size(ATMS_Input_Time))    
            windu2M             = np.zeros(np.size(ATMS_Input_Time))
            windv2M             = np.zeros(np.size(ATMS_Input_Time))
            surfacep            = np.zeros(np.size(ATMS_Input_Time))
            surfacep_3D         = np.zeros(np.size(ATMS_Input_Time))
            #-----------------------------------
            snow_frac[:]        = 1.0
            
            func                = interpolate.interp1d(MERRA2['2D_time'],MERRA2['2D_TS'],fill_value='extrapolate',bounds_error=False)
            skintemperature     = func(ATMS_Input_Time)
            surfacetemperature  = func(ATMS_Input_Time)

            func                = interpolate.interp1d(MERRA2['2D_time'],MERRA2['2D_T2M'],fill_value='extrapolate',bounds_error=False)
            temperature2M       = func(ATMS_Input_Time)
            func                = interpolate.interp1d(MERRA2['2D_time'],MERRA2['2D_QV2M'],fill_value='extrapolate',bounds_error=False)
            humidity2M          = func(ATMS_Input_Time)
            func                = interpolate.interp1d(MERRA2['2D_time'],MERRA2['2D_U2M'],fill_value='extrapolate',bounds_error=False)
            windu2M             = func(ATMS_Input_Time)
            func                = interpolate.interp1d(MERRA2['2D_time'],MERRA2['2D_V2M'],fill_value='extrapolate',bounds_error=False)
            windv2M             = func(ATMS_Input_Time)
            
            func                = interpolate.interp1d(MERRA2['2D_time'],MERRA2['2D_PS'],fill_value='extrapolate',bounds_error=False)
            surfacep            = func(ATMS_Input_Time)
                            
            func                = interpolate.interp1d(MERRA2['3D_time'],MERRA2['3D_surfacep'][:,0],fill_value='extrapolate',bounds_error=False)
            surfacep_3D         = func(ATMS_Input_Time)
            
            TP_func             = []
            for itime in range(np.size(ATMS_Input_Time)):
                time                = ATMS_Input_Time[itime]
                tmp                 = np.abs(MERRA2['3D_time']-time)
                tmp                 = tmp==np.min(tmp)
                timeindex_MERRA2_3D = tmp.nonzero()[0][0]
                TP_func.append( interpolate.interp1d(MERRA2['3D_height'][timeindex_MERRA2_3D,:],np.log10(MERRA2['3D_pressure'][timeindex_MERRA2_3D,:]),fill_value='extrapolate',bounds_error=False) )                
            
            Input_Day       = np.zeros(np.size(ATMS_Input_Time))
            Input_Hour      = np.zeros(np.size(ATMS_Input_Time))
            Input_Minute    = np.zeros(np.size(ATMS_Input_Time))
            Input_Second    = np.zeros(np.size(ATMS_Input_Time))

            for itime in range(np.size(ATMS_Input_Time)):
                time_DOM            = ATMS_Input_Time_DOM[itime]
                Input_Day[itime]    = int(time_DOM)+1
                tmp                 = (time_DOM - int(time_DOM))*24.0
                Input_Hour[itime]   = int(tmp)
                tmp                 = (tmp - int(tmp))*60.0
                Input_Minute[itime] = int(tmp)
                tmp                 = (tmp - int(tmp))*60.0
                Input_Second[itime] = tmp
                if (tmp>=60):
                    Input_Second[itime] = 59.0
    #--- Run    
    if True:
        Global_Level    = int(72)
        Errorprior      = True
        fname           = 'Antractica'+year+'_MERRA2_'+ARMGroup+'.h5'
        read_prior(fname,year,month,ARMStation,nlevel=Global_Level,appendix=appendix)
                                                            
        if (Errorprior==True):
            for itime in range(np.size(ATMS_Input_Time)):
                keep_iter       = True         
                time            = ATMS_Input_Time[itime]
                time_dayofyear  = ATMS_Input_Time_DOM[itime]
                input_incid     = np.abs(ATMS_Input_Incid[itime])
                input_tb        = np.copy(ATMS_Input_TB[itime,:])
                                                                 
                iter  = 0
                while ( (iter<=10) & (keep_iter==True) ):
                    ErrorRTTOV  = False   
                    Flag        = True
                         
                    if (iter>=1):
                        Emission_Result = np.loadtxt(year+'/'+month+'/'+ARMStation+'/Emission_iter'+str( int(2*(iter-1)+2) )+'_'+str(itime)+'_'+ARMGroup+'_'+str(E0)+'.txt') 
                        Emiss           = np.zeros(22)
                        Emiss[0]        = Emission_Result[0]                        
                        Emiss[1]        = Emission_Result[1]                         
                        Emiss[2]        = Emission_Result[2]                         
                        Emiss[3:15]     = Emission_Result[2] 
                        Emiss[15]       = Emission_Result[15]                                                 
                        Emiss[16]       = Emission_Result[16]                                                 
                        Emiss[17]       = Emission_Result[17]                                                 
                        Emiss[18:22]    = Emission_Result[17]       
                        Emiss_old       = np.copy(Emiss)    
                    else:
                        Emiss           = np.ones(22) * E0                                                           
                        Emiss_old       = np.copy(Emiss)    
                    
                    #----- Step 1 Update Atmosphere Profiles -----#
                    if True:
                        frequencies = np.copy(ATMS['Freq'])
                        y_vars      = np.array(frequencies)
                        x_vars      = Reference['x_mean'].state.values
                        #---- Y uncertainty, covariance matrix Sy -----
                        y_noise = pn.Series(
                            [
                                0.7, 0.8, 0.9, 0.7, 0.7,0.7, 0.7, 0.7, 0.7, 0.75, 1.2, 1.2, 1.5, 2.4, 3.6, 0.5, 0.6, 0.8, 0.8, 0.8, 0.8, 0.9
                            ],
                            index=y_vars
                        )
                        y_obs   = np.array(input_tb)        

                        S_y = pn.DataFrame(
                            np.diag(y_noise.values**2),
                            index=y_vars,
                            columns=y_vars,
                        )
                        #---- Prior -----
                        x_a = Reference['x_mean'].sel(season='all').to_pandas()[x_vars]
                        S_a = Reference['x_cov'].sel(season='all').to_pandas().loc[x_vars, x_vars]

                        inputP = 10.00**TP_func[itime](Reference['height'].values)

                        forwardKwArgs = dict(
                            pressure=inputP,
                            zenithAngleATMS=input_incid,
                            latitude=ARMStation_LAT, 
                            longitude=ARMStation_LON, 
                            date=np.array([int(year), int(month), Input_Day[itime], Input_Hour[itime], Input_Minute[itime], Input_Second[itime]]),                                             
                            surfacepressure=surfacep[itime], 
                            surfacetemperature=surfacetemperature[itime], 
                            snow_frac=snow_frac[itime], 
                            temperature2M=temperature2M[itime], 
                            humidity2M=humidity2M[itime], 
                            windu2M=windu2M[itime], 
                            windv2M=windv2M[itime], 
                            skintemperature=skintemperature[itime], 
                            emiss=Emiss
                        )

                        y_obs = pn.Series(y_obs, index=y_vars)
                        assert np.all(np.isfinite(y_obs))        

                        oe_ref = optimalEstimation2(
                            x_vars, # state variable names
                            x_a,  # a priori
                            S_a, # a priori uncertainty
                            y_vars,  # measurement variable names
                            y_obs, # observations
                            S_y, # observation uncertainty
                            forwardRT, # forward Operator
                            forwardKwArgs=forwardKwArgs, # additonal function arguments
                            x_truth=Truth['x_truth'], # true profile
                        )

                        oe_ref.doRetrieval(Nstep=2)   

                        if (np.isnan(oe_ref.x_op).any()):
                            ErrorRTTOV = False

                            oe_ref.doRetrieval(Nstep=1)    
                            if (np.isnan(oe_ref.x_op).any()):
                                Flag        = False  
                                keep_iter   = False
                                          
                            else:
                                T_optimal, Q_optimal = splitTQ(oe_ref.x_op)
                                T_truth, Q_truth = splitTQ(oe_ref.x_truth)

                                np.savetxt(year+'/'+month+'/'+ARMStation+'/T_optimal_iter'+str( int(2*iter+1) )+'_'+str(itime)+'_'+ARMGroup+'_'+str(E0)+'.txt',T_optimal)
                                np.savetxt(year+'/'+month+'/'+ARMStation+'/Q_optimal_iter'+str( int(2*iter+1) )+'_'+str(itime)+'_'+ARMGroup+'_'+str(E0)+'.txt',Q_optimal)
                                np.savetxt(year+'/'+month+'/'+ARMStation+'/TB_optimal_iter'+str( int(2*iter+1) )+'_'+str(itime)+'_'+ARMGroup+'_'+str(E0)+'.txt',oe_ref.y_op)

                        else:  
                            T_optimal, Q_optimal = splitTQ(oe_ref.x_op)
                            T_truth, Q_truth = splitTQ(oe_ref.x_truth)
                        
                            np.savetxt(year+'/'+month+'/'+ARMStation+'/T_optimal_iter'+str( int(2*iter+1) )+'_'+str(itime)+'_'+ARMGroup+'_'+str(E0)+'.txt',T_optimal)
                            np.savetxt(year+'/'+month+'/'+ARMStation+'/Q_optimal_iter'+str( int(2*iter+1) )+'_'+str(itime)+'_'+ARMGroup+'_'+str(E0)+'.txt',Q_optimal)
                            np.savetxt(year+'/'+month+'/'+ARMStation+'/TB_optimal_iter'+str( int(2*iter+1) )+'_'+str(itime)+'_'+ARMGroup+'_'+str(E0)+'.txt',oe_ref.y_op)
                        
                    #----- Step 2 Update Surface Emission -----#
                    if (Flag):     
                        inputP = 10.00**TP_func[itime](Reference['height'].values)
        
                        ErrorRTTOV==False

                        Emission_list   = np.arange(11)*0.1
                        Model_list      = np.zeros((11,22)) 
                        Emission_Result = np.zeros(22)

                        for i in range(11):
                            Emission = np.ones(22)*Emission_list[i]
                            
                            forwardKwArgs = dict(
                                pressure=inputP,
                                zenithAngleATMS=input_incid,
                                latitude=ARMStation_LAT, 
                                longitude=ARMStation_LON, 
                                date=np.array([int(year), int(month), Input_Day[itime], Input_Hour[itime], Input_Minute[itime], Input_Second[itime]]),
                                surfacepressure=surfacep[itime], 
                                surfacetemperature=surfacetemperature[itime], 
                                snow_frac=snow_frac[itime], 
                                temperature2M=temperature2M[itime], 
                                humidity2M=humidity2M[itime], 
                                windu2M=windu2M[itime], 
                                windv2M=windv2M[itime], 
                                skintemperature=skintemperature[itime], 
                                emiss=Emission
                            )
                            
                            Model_list[i,:] = forwardRT( oe_ref.x_op, **forwardKwArgs)

                        if (ErrorRTTOV==False):
                            for i in range(22):
                                tmpx0 = np.array(Emission_list)
                                tmpy0 = np.copy(Model_list[:,i])
                                tmpx  = np.arange(101)*0.01   
                                tmpfunc = interpolate.interp1d(tmpx0,tmpy0)
                                tmpy    = tmpfunc(tmpx)
                                tmp     = np.abs(y_obs[i] - tmpy)
                                tmp     = tmp == np.min(tmp)
                                Emission_Result[i] = tmpx[tmp.nonzero()[0][0]]

                            np.savetxt(year+'/'+month+'/'+ARMStation+'/Emission_iter'+str( int(2*iter+2) )+'_'+str(itime)+'_'+ARMGroup+'_'+str(E0)+'.txt',Emission_Result)
                            #-------------------------
                            Emiss       = np.zeros(22)
                            Emiss[0]    = Emission_Result[0]                        
                            Emiss[1]    = Emission_Result[1]                         
                            Emiss[2]    = Emission_Result[2]                         
                            Emiss[3:15] = Emission_Result[2] 
                            Emiss[15]   = Emission_Result[15]                                                 
                            Emiss[16]   = Emission_Result[16]                                                 
                            Emiss[17]   = Emission_Result[17]                                                 
                            Emiss[18:22]= Emission_Result[17]       
                        
                            tmp = np.abs(Emiss-Emiss_old)
                            select = tmp>=0.01
                            if (np.sum(select)==0):
                                keep_iter   = False                                                                                                                
                        else:
                            keep_iter   = False   
                                                             
                    iter = iter + 1     
         
                        
    return

def disortcall(listarg):
    month       = listarg[0]
    appendix    = listarg[1]
    E0          = listarg[2]
    ARMStation  = listarg[3]
    runRetrieval(year='2016',month=month,ARMStation=ARMStation,appendix=appendix,E0=E0)
    return
if __name__ == '__main__':
    
    if True:        
        nproc           = 32
        appendix0       = 0                     
        multip          = Pool(nproc)
        
        month_vec        = []  
        appendix_vec     = []      
        E0_vec           = []
        ARMStation_vec   = []
        
        for month in ['01','02','03','04','05','06','07','08','09','10','11','12']:
            for E0 in [0.8]:
                for ARMStation in ['McMurdo']:                    
                    
                    month_vec.append(month)
                    ARMStation_vec.append(ARMStation)
                    appendix_vec.append(appendix0)
                    appendix0 = appendix0+1
                    E0_vec.append(E0)

                                        
        listarg = [[a1, a2, a3, a4]
                   for a1, a2, a3, a4 in
                   zip( month_vec, appendix_vec, E0_vec, ARMStation_vec )]
        multip.map(disortcall, listarg)
        multip.close()  
            
        pdb.set_trace()
    
    
    
    
    
    pdb.set_trace()













