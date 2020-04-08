#! /bin/python3
# vim: set expandtab:
# -------------------------------------------------------------------------
import sys
import pandas as pd
import numpy as np 
import curvefit
from curvefit.functions import *
from matplotlib import pyplot as plt 
#
# -------------------------------------------------------------------------
# load NYC data
df = pd.read_csv('../../data/ny_google.csv')
df['day'] = pd.Series([i+1 for i in range(len(df))])
# set vars
n_data       = len(df) # 29
num_params   = 3 # alpha beta p
alpha_true   = 2.0
beta_true    = 3.0
p_true       = 4.0
rel_tol      = 1e-6
func = erf
fname = "NY"
xcol = 'day'
ycol = 'cases'
groupcol = 'fips'
# -------------------------------------------------------------------------
# model for the mean of the data
def generalized_logistic(t, params) :
    alpha = params[0]
    beta  = params[1]
    p     = params[2]
    return p / ( 1.0 + np.exp( - alpha * ( t - beta ) ) )
#
# link function used for beta
def identity_fun(x) :
    return x
#
# link function used for alpha, p
def exp_fun(x) :
    return np.exp(x)
#
# params_true
params_true       = np.array( [ alpha_true, beta_true, p_true ] )
#


# data_frame
independent_var   = df[xcol]
measurement_value = df[ycol] # generalized_logistic(independent_var, params_true)
measurement_std   = n_data * [ 0.1 ] # NEED TO DEFINE, SET TO NONE FOR NOW BELOW

# covariates
covs            = n_data * [ 1.0 ]

data_group        = df[groupcol]

data_dict         = {
    'independent_var'   : independent_var   ,
    'measurement_value' : measurement_value ,
    'measurement_std'   : measurement_std   ,
    'covs'              : covs      ,
    'region'            : data_group        ,
}
data_frame        = pd.DataFrame(data_dict)
#
# curve_model

col_t        = 'independent_var'
col_obs      = 'measurement_value'
col_covs     = num_params *[ [ 'covs' ] ]
col_group    = 'region'
param_names  = [ 'alpha', 'beta',       'p'     ]
link_fun     = [ exp_fun, identity_fun, exp_fun ]
var_link_fun = link_fun
fun          = func
col_obs_se   = None # 'measurement_std'
#
curve_model = curvefit.CurveModel(
    data_frame,
    col_t,
    col_obs,
    col_covs,
    col_group,
    param_names,
    link_fun,
    var_link_fun,
    fun,
    col_obs_se
)
#
# fit_params
# TODO: what do do about the below line when we don't have params true?
fe_init         = params_true / 3.0
curve_model.fit_params(fe_init)
params_estimate = curve_model.params
#
# for i in range(num_params) :
#     rel_error = params_estimate[i] / params_true[i] - 1.0
#     assert abs(rel_error) < rel_tol
#
print(params_estimate)
print('get_started.py: OK')
# sys.exit(0)

plt.yscale("log")
plt.grid()
plt.xlabel("Number of Days")
plt.ylabel("Cases")
plt.plot(independent_var, measurement_value, 'b-', label='data')
plt.plot(independent_var, func(independent_var, params_estimate), 'r-', label='fit: {}: {}'.format(func.__name__, params_estimate))
plt.title("{} Cases fit to {}".format(fname, func.__name__))
# plt.plot(x, func(x, params_estimate + std), 'g--', label='fit: {}: {}'.format(func.__name__, params_estimate + std))
# plt.plot(x, func(x, params_estimate - std), 'g--', label='fit: {}: {}'.format(func.__name__, params_estimate - std))
# plt.plot([], [], ' ', label="std dev {}".format(std))

plt.legend() 
plt.savefig('output/{}_fitting_{}.png'.format(fname, func.__name__))
# plt.show() 
plt.clf() 

