#! /bin/python3
# vim: set expandtab:
# -------------------------------------------------------------------------
import sys
import pandas as pd
import numpy as np 
import curvefit
from curvefit.core.functions import *
from curvefit.core.model import *
from matplotlib import pyplot as plt 
from sklearn.metrics import r2_score, mean_squared_log_error
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter

# -------------------------------------------------------------------------
# load data
df = pd.read_csv('../../data/data/bbmp.csv')
df['date'] = df['Date'].apply(lambda x: datetime.strptime(x, "%d.%m.%Y").date())
df['day'] = pd.Series([i+1 for i in range(len(df))])
df.rename(columns = {'Cumulative Deaths til Date':'deaths'}, inplace = True) 
df.rename(columns = {'Cumulative Cases Til Date':'cases'}, inplace = True) 
df['group'] = pd.Series([1 for i in range(len(df))])
test_size = 5
seed = 'last{}'.format(test_size)
# data, test = train_test_split(df, train_size=.8, shuffle=True, random_state=seed)
data = df[:-test_size]
test = df[-test_size:]
print ('seed: {}'.format(seed))

# set vars
n_data       = len(data) # 29
num_params   = 3 # alpha beta p
alpha_true   = 2.0
beta_true    = 3.0
p_true       = 4.0
rel_tol      = 1e-6
# model for the mean of the data
def generalized_logistic(t, params) :
    alpha = params[0]
    beta  = params[1]
    p     = params[2]
    return p / ( 1.0 + np.exp( - alpha * ( t - beta ) ) )
# erf, derf, expit, dderf
fname = "BBMP"
xcol = 'day'
date = 'date'
ycols = {
    'cases' : generalized_logistic,
    'deaths': generalized_logistic,
}
groupcol = 'group'
# -------------------------------------------------------------------------
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

def fit_predict_plot(curve_model, xcol, ycol, data, test, func, predictdate):
    # fit_params
    # TODO: what do do about the below line when we don't have params true?
    fe_init         = params_true / 3.0
    curve_model.fit_params(fe_init)
    params_estimate = curve_model.params
    print(params_estimate)
    
    # for i in range(num_params) :
    #     rel_error = params_estimate[i] / params_true[i] - 1.0
    #     assert abs(rel_error) < rel_tol
    
    predictx = np.array([x+1 for x in range(len(predictdate))])
    predictions = curve_model.predict(t=predictx, group_name='all')

    xtest, ytest = test[xcol], test[ycol]
    predtest = curve_model.predict(t=xtest, group_name='all')
    r2, msle = r2_score(ytest, predtest), mean_squared_log_error(ytest, predtest)
    print ('test set - r2: {} msle: {}'.format(r2, msle))

    r2, msle = r2_score(data[ycol], predictions[:len(data[xcol])]), \
        mean_squared_log_error(data[ycol], predictions[:len(data[xcol])])
    print ('overall - r2: {} msle: {}'.format(r2, msle))

    # sys.exit(0)

    plt.yscale("log")
    ax = plt.gca()
    formatter = DateFormatter("%d.%m")
    ax.xaxis.set_major_formatter(formatter)
    plt.grid()
    plt.xlabel("Date")
    plt.ylabel("Cases")
    plt.plot(data[date], data[ycol], 'b+', label='data')
    plt.plot(test[date], test[ycol], 'g+', label='data (test)')
    plt.plot(predictdate, predictions, 'r-', label='fit: {}: {}'.format(func.__name__, params_estimate))
    plt.title("{} Cases fit to {}".format(fname, func.__name__))
    # plt.plot(x, func(x, params_estimate + std), 'g--', label='fit: {}: {}'.format(func.__name__, params_estimate + std))
    # plt.plot(x, func(x, params_estimate - std), 'g--', label='fit: {}: {}'.format(func.__name__, params_estimate - std))
    # plt.plot([], [], ' ', label="std dev {}".format(std))

    plt.legend() 
    plt.savefig('output/{}_{}_fitting_{}_{}.png'.format(fname, ycol, func.__name__, seed))
    # plt.show() 
    plt.clf()

    return predictions 

predictions = {}
for ycol, func in ycols.items():

    # data_frame
    independent_var   = data[xcol]
    measurement_value = data[ycol] # generalized_logistic(independent_var, params_true)
    measurement_std   = n_data * [ 0.1 ] # NEED TO DEFINE, SET TO NONE FOR NOW BELOW

    # covariates
    covs            = n_data * [ 1.0 ]

    data_group        = data[groupcol]

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
    link_fun     = [ identity_fun, exp_fun, exp_fun ]
    var_link_fun = link_fun
    fun          = func
    col_obs_se   = None # 'measurement_std'
    #
    curve_model = CurveModel(
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

    predictdate = pd.to_datetime(pd.Series([timedelta(days=x)+data[date].iloc[0] for x in range(90)]))
    predictions[ycol] = fit_predict_plot(curve_model, xcol, ycol, data, test, func, predictdate)

# datetime | confidence | infections lower bound | infections most likely


results = pd.concat([df[date], pd.Series(predictdate[len(df):], name=date)], axis=0)

for ycol, preds in predictions.items():
    col_results = pd.concat([pd.Series(df[ycol]), pd.Series(preds[len(df):], name=ycol)], axis=0)
    results = pd.concat([results, col_results], axis=1)
    temp = results[ycol].shift(1)
    results["new {}".format(ycol)] = results[ycol] - temp

results.to_csv("output/{}_{}_{}.csv".format(fname, func.__name__, seed), index=False)


