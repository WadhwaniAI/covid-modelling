#! /bin/python3
# vim: set expandtab:
# -------------------------------------------------------------------------
import sys
import pandas as pd
import numpy as np 
import curvefit
from curvefit.core.functions import *
from curvefit.pipelines.basic_model import *
from matplotlib import pyplot as plt 
from sklearn.metrics import r2_score, mean_squared_log_error
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter

# -------------------------------------------------------------------------
# load data
df = pd.read_csv('../../data/data/bbmp.csv')
df['date'] = df['Date'].apply(lambda x: datetime.strptime(x, "%d.%m.%Y"))
df.drop("Date", axis=1, inplace=True)
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
    'cases' : erf,
    'deaths': erf,
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
    
    predictx = np.array([x+1 for x in range(len(predictdate))])
    
    pipeline.setup_pipeline()
    pipeline.run_predictive_validity(theta=1) # TODO: determine theta
    pipeline.fit(data)
    params_estimate = pipeline.mod.params
    print(params_estimate)
    # print (pipeline.pv.all_residuals)
    # TODO: all params for below. the column names for covariates need to be fixed below, and need to be in pipeline.pv.all_residuals
    pipeline.fit_residuals(smoothed_radius=[3,3], num_smooths=4, covariates=['num_data', 'data_index'], exclude_below=0, exclude_groups=[])
    # TODO: determine num_draws
    pipeline.create_draws(num_draws=5, prediction_times=predictx)
    predictions = pipeline.predict(times=predictx, predict_space=func, predict_group='all')

    pipeline.plot_results(prediction_times=predictx)
    plt.savefig('output/pipeline/draws_{}_{}_{}_{}.png'.format(fname, ycol, func.__name__, seed))
    plt.clf()
    
    xtest, ytest = test[xcol], test[ycol]
    predtest = pipeline.predict(times=xtest, predict_space=func, predict_group='all')
    r2, msle = r2_score(ytest, predtest), mean_squared_log_error(ytest, predtest)
    print ('test set - r2: {} msle: {}'.format(r2, msle))

    r2, msle = r2_score(data[ycol], predictions[:len(data[xcol])]), \
        mean_squared_log_error(data[ycol], predictions[:len(data[xcol])])
    print ('overall - r2: {} msle: {}'.format(r2, msle))

    # sys.exit(0)

    plt.yscale("log")
    plt.gca().xaxis.set_major_formatter(DateFormatter("%d.%m"))
    plt.grid()
    plt.xlabel("Date")
    plt.ylabel("Cases")
    plt.plot(data[date], data[ycol], 'b+', label='data')
    plt.plot(test[date], test[ycol], 'g+', label='data (test)')
    plt.plot(predictdate, predictions, 'r-', label='fit: {}: {}'.format(func.__name__, params_estimate))
    plt.title("{} Cases fit to {}".format(fname, func.__name__))
    
    plt.legend() 
    plt.savefig('output/pipeline/{}_{}_{}_{}.png'.format(fname, ycol, func.__name__, seed))
    # plt.show() 
    plt.clf()

    return predictions 

predictions = {}
for ycol, func in ycols.items():
    
    data['covs']            = n_data * [ 1.0 ]

    param_names  = [ 'alpha', 'beta',       'p'     ]
    link_fun     = [ identity_fun, exp_fun, exp_fun ]
    var_link_fun = link_fun
    covs = ['covs']
    #
    pipeline = BasicModel(
        all_data=data, #: (pd.DataFrame) of *all* the data that will go into this modeling pipeline
        col_t=xcol, #: (str) name of the column with time
        col_group=groupcol, #: (str) name of the column with the group in it
        col_obs=ycol, #: (str) the name of the column with observations for fitting the model
        col_obs_compare=ycol, #TODO: (str) the name of the column that will be used for predictive validity comparison
        all_cov_names=covs, #TODO: List[str] list of name(s) of covariate(s). Not the same as the covariate specifications
            # that are required by CurveModel in order of parameters. You should exclude intercept from this list.
        fun=func, #: (callable) the space to fit in, one of curvefit.functions
        predict_space=func, #TODO confirm: (callable) the space to do predictive validity in, one of curvefit.functions
        obs_se_func=None, #TODO if we wanna specify: (optional) function to get observation standard error from col_t
        # predict_group='all', #: (str) which group to make predictions for
        fit_dict={ # TODO: add priors here
            'fe_init': params_true / 3.0,
        }, #: keyword arguments to CurveModel.fit_params()
        basic_model_dict= { #: additional keyword arguments to the CurveModel class
            'col_obs_se': None,#(str) of observation standard error
            'col_covs': num_params*[covs],#TODO: List[str] list of names of covariates to put on the parameters
            'param_names': param_names,#(list{str}):
                # Names of the parameters in the specific functional form.
            'link_fun': link_fun,#(list{function}):
                # List of link functions for each parameter.
            'var_link_fun': var_link_fun,#(list{function}):
                # List of link functions for the variables including fixed effects
                # and random effects.
        },
    )

    predictdate = pd.to_datetime(pd.Series([timedelta(days=x)+data[date].iloc[0] for x in range(90)]))
    predictions[ycol] = fit_predict_plot(pipeline, xcol, ycol, data, test, func, predictdate)

# datetime | confidence | infections lower bound | infections most likely


# results = pd.concat([df[date], pd.Series(predictdate[len(df):], name=date)], axis=0)

# for ycol, preds in predictions.items():
#     col_results = pd.concat([pd.Series(df[ycol]), pd.Series(preds[len(df):], name=ycol)], axis=0)
#     results = pd.concat([results, col_results], axis=1)
#     temp = results[ycol].shift(1)
#     results["new {}".format(ycol)] = results[ycol] - temp

# results.to_csv("output/{}_{}_{}.csv".format(fname, func.__name__, seed), index=False)



