#! /bin/python3
# vim: set expandtab:
# -------------------------------------------------------------------------
import sys
import json
import argparse
import pandas as pd
import numpy as np 
from datetime import datetime, timedelta
from matplotlib import pyplot as plt 
from matplotlib.dates import DateFormatter

import curvefit
from curvefit.core.functions import *
from curvefit.pipelines.basic_model import *

from sklearn.metrics import r2_score, mean_squared_log_error
from sklearn.model_selection import train_test_split

from data import *

# -------------------------------------------------------------------------
# load params



params_group = "karnataka"
with open('params.json', "r") as paramsfile:
  pargs = json.load(paramsfile)[params_group]

# load data

data_func = getattr(sys.modules[__name__], pargs['data_func'])
if 'data_func_args' in pargs:
    df = data_func(pargs['data_func_args'])
else:
    df = data_func()

test_size = pargs['test_size']
data, test = df[:-test_size], df[-test_size:]
seed = 'last{}'.format(test_size)
print ('seed: {}'.format(seed))
# data, test = train_test_split(df, train_size=.8, shuffle=True, random_state=seed)

# set vars
n_data       = len(data)
num_params   = 3 # alpha beta p
alpha_true   = pargs['alpha_true'] # TODO
beta_true    = pargs['beta_true'] # TODO
p_true       = pargs['p_true'] # TODO

fname = params_group
date, groupcol = pargs['date'], pargs['groupcol']
xcol, ycols = pargs['xcol'], pargs['ycols']
for (k,v) in ycols.items():
    ycols[k] = getattr(sys.modules[__name__], v)

daysforward, daysback = pargs['daysforward'], pargs['daysback']
# -------------------------------------------------------------------------

# link functions
identity_fun = lambda x: x
exp_fun = lambda x : np.exp(x)

params_true       = np.array( [ alpha_true, beta_true, p_true ] )

def fit_predict_plot(curve_model, xcol, ycol, data, test, func, predictdate, pargs=None):
    p_args = {
        "n_draws": 5,
        "cv_threshold": 1e-2,
        "smoothed_radius": [3,3], 
        "num_smooths": 3, 
        "exclude_groups": [], 
        "exclude_below": 0,
        "exp_smoothing": None, 
        "max_last": None
    }
    p_args.update(pargs)

    predictx = np.array([x+1 for x in range(-daysback,daysforward)])
    
    # pipeline
    pipeline.setup_pipeline()
    # TODO: all params for below. the column names for covariates need to be fixed below, and need to be in pipeline.pv.all_residuals
    pipeline.run(n_draws=p_args['n_draws'], prediction_times=predictx, 
        cv_threshold=p_args['cv_threshold'], smoothed_radius=p_args['smoothed_radius'], 
        num_smooths=p_args['num_smooths'], exclude_groups=p_args['exclude_groups'], 
        exclude_below=p_args['exclude_below'], exp_smoothing=p_args['exp_smoothing'], 
        max_last=p_args['max_last']
    )
    params_estimate = pipeline.mod.params
    print(params_estimate)
    
    # plot draws
    pipeline.plot_results(prediction_times=predictx)
    plt.savefig('output/pipeline/draws_{}_{}_{}_{}.png'.format(fname, ycol, func.__name__, seed))
    plt.clf()
    
    predictions = pipeline.predict(times=predictx, predict_space=func, predict_group='all')
    
    # evaluate against test set
    xtest, ytest = test[xcol], test[ycol]
    predtest = pipeline.predict(times=xtest, predict_space=func, predict_group='all')
    r2, msle = r2_score(ytest, predtest), mean_squared_log_error(ytest, predtest)
    print ('test set - r2: {} msle: {}'.format(r2, msle))

    # evaluate overall
    r2, msle = r2_score(data[ycol], predictions[daysback:daysback+len(data[xcol])]), \
        mean_squared_log_error(data[ycol], predictions[daysback:daysback+len(data[xcol])])
    print ('overall - r2: {} msle: {}'.format(r2, msle))


    # plot predictions against actual
    plt.yscale("log")
    plt.gca().xaxis.set_major_formatter(DateFormatter("%d.%m"))
    plt.grid()
    plt.xlabel("Date")
    plt.ylabel(ycol)
    plt.plot(data[date], data[ycol], 'b+', label='data')
    plt.plot(test[date], test[ycol], 'g+', label='data (test)')
    plt.plot(predictdate, predictions, 'r-', label='fit: {}: {}'.format(func.__name__, params_estimate))
    plt.title("{} {} fit to {}".format(fname, ycol, func.__name__))
    
    plt.legend() 
    plt.savefig('output/pipeline/{}_{}_{}_{}.png'.format(fname, ycol, func.__name__, seed))
    # plt.show() 
    plt.clf()


    for i, group in enumerate(pipeline.groups):
        # x = prediction_times = predictx
        draws = pipeline.draws[group].copy()
        # draws = data_translator(
        #     data=draws,
        #     input_space=pipeline.predict_space,
        #     output_space=pipeline.predict_space
        # )
        mean_fit = pipeline.mean_predictions[group].copy() # predictions
        # mean_fit = data_translator(
        #     data=mean_fit,
        #     input_space=pipeline.predict_space,
        #     output_space=pipeline.predict_space
        # )
        
        mean = draws.mean(axis=0)

        # uncertainty
        lower = np.quantile(draws, axis=0, q=0.025)
        upper = np.quantile(draws, axis=0, q=0.975)

    return mean_fit, lower, mean, upper 

predictions = {}
for ycol, func in ycols.items():
    
    data.loc[:,'covs']            = n_data * [ 1.0 ]
    data.loc[:,'deaths_normalized']            = data['deaths']/data['deaths'].max()

    param_names  = [ 'alpha', 'beta',       'p'     ]
    covs = ['covs', 'covs', 'covs']
    link_fun     = [ identity_fun, exp_fun, exp_fun ]
    var_link_fun = link_fun

    # # think this could work with more death data:
    # link_fun     = [ identity_fun, identity_fun, exp_fun ]
    # covs = ['covs', 'deaths_normalized', 'covs']
    
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
            # 'col_covs': num_params*[covs],#TODO: List[str] list of names of covariates to put on the parameters
            'col_covs': [[cov] for cov in covs],#TODO: List[str] list of names of covariates to put on the parameters
            'param_names': param_names,#(list{str}):
                # Names of the parameters in the specific functional form.
            'link_fun': link_fun,#(list{function}):
                # List of link functions for each parameter.
            'var_link_fun': var_link_fun,#(list{function}):
                # List of link functions for the variables including fixed effects
                # and random effects.
        },
    )

    predictdate = pd.to_datetime(pd.Series([timedelta(days=x)+data[date].iloc[0] for x in range(-daysback,daysforward)]))
    predictions[ycol] = fit_predict_plot(pipeline, xcol, ycol, data, test, func, predictdate, pargs=pargs)

# datetime | confidence | infections lower bound | infections most likely


results = pd.concat([df[date], pd.Series(predictdate[len(df):], name=date)], axis=0)

for ycol, (preds, lower, mean, upper) in predictions.items():
    dummy = pd.Series(np.full((len(preds)-len(df),), fill_value=np.nan), name='observed {}'.format(ycol))
    observed = pd.concat(
        [pd.Series(df[ycol], name='observed {}'.format(ycol)), dummy],axis=0
    ).reset_index(drop=True)
    results = pd.concat([results, pd.Series(observed)], axis=1)
    results = pd.concat([results, pd.Series(preds, name='preds {}'.format(ycol))], axis=1)
    results = pd.concat([results, pd.Series(lower, name='lower {}'.format(ycol))], axis=1)
    results = pd.concat([results, pd.Series(mean, name='mean {}'.format(ycol))], axis=1)
    results = pd.concat([results, pd.Series(upper, name='upper {}'.format(ycol))], axis=1)

results.to_csv("output/pipeline/{}_{}_{}.csv".format(fname, func.__name__, seed), index=False)



