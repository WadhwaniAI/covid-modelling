#! /bin/python3
# vim: set expandtab:
# -------------------------------------------------------------------------
import sys
import os
import json
import argparse
import pandas as pd
import numpy as np 
from datetime import datetime, timedelta
from matplotlib import pyplot as plt 

from matplotlib.dates import DateFormatter

import curvefit
from curvefit.core.functions import *
from curvefit.core.utils import data_translator
from curvefit.pipelines.basic_model import *

from sklearn.metrics import r2_score, mean_squared_log_error
from sklearn.model_selection import train_test_split

from data import *
from util import plot_draws_deriv, mape, smooth, setup_plt, Params

# -------------------------------------------------------------------------
# load params

# seed = 'last{}'.format(self.test_size)
# print ('seed: {}'.format(seed))
# # data, test = train_test_split(df, train_size=.8, shuffle=True, random_state=seed)

# n_data = len(data)

# fname = label
# output_folder = f'output/pipeline/{fname}'
# if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

derivs = {
    erf: derf,
    # gaussian_cdf: gaussian_pdf,
    log_erf: log_derf,
}
parser = argparse.ArgumentParser() 
parser.add_argument("-p", "--params", help="name of entry in params.json", required=True)
parser.add_argument("-d", "--daily", help="whether or not to plot daily", required=False, action='store_true')
parser.add_argument("-s", "--smoothing", help="how much to smooth, else no smoothing", required=False)
args = parser.parse_args() 
daily, smoothing_window = args.daily, args.smoothing

with open('params.json', "r") as paramsfile:
    pargs = json.load(paramsfile)
    if args.params not in pargs:
        print("entry not found in params.json")
        sys.exit(0)
    pargs = pargs[args.params]

# set vars
num_params   = 3 # alpha beta p
alpha_true   = pargs['alpha_true'] # TODO
beta_true    = pargs['beta_true'] # TODO
p_true       = pargs['p_true'] # TODO
params_true       = np.array( [ alpha_true, beta_true, p_true ] )

fname = args.params
output_folder = f'output/pipeline/{fname}'
if not os.path.exists(output_folder):
        os.makedirs(output_folder)
date, groupcol = pargs['date'], pargs['groupcol']
xcol, ycols = pargs['xcol'], pargs['ycols']
for (k,v) in ycols.items():
    ycols[k] = getattr(sys.modules[__name__], v)
daycol = 'day'

aparams = Params(args.params)

# assign data
adf, aagg_df = aparams.df, aparams.agg_df

# load data
data_func = getattr(sys.modules[__name__], pargs['data_func'])
if 'data_func_args' in pargs:
    df = data_func(pargs['data_func_args'])
else:
    df = data_func()

daily = args.daily
if daily:
    for ycol in ycols.keys():
        dailycol = f"daily_{ycol}"
        dailycol_vals = pd.Series()
        df.sort_values(groupcol)
        for grp in df[groupcol].unique():
            dailycol_vals = pd.concat((dailycol_vals, df[df[groupcol] == grp][ycol] - df[df[groupcol] == grp][ycol].shift(1)))

        df[dailycol] = dailycol_vals

if smoothing_window:
    new_ycols = {}
    for ycol in ycols.keys():
        df[f'{ycol}_smooth'] = smooth(df[ycol], 5)
        new_ycols[f'{ycol}_smooth'] = ycols[ycol]
    orig_ycols = ycols
    ycols = new_ycols 


test_size = pargs['test_size'] # in num days

max_date = df[date].max()
df[daycol] = (df[date] - df[date].min()).dt.days
threshold = max_date - timedelta(days=test_size)

data, test = df[df[date] < threshold], df[df[date] >= threshold]
agg_df = df.groupby(date).sum().reset_index(col_fill=date)
agg_df[daycol] = (agg_df[date] - agg_df[date].min()).dt.days
agg_data, agg_test = agg_df[agg_df[date] < threshold], agg_df[agg_df[date] >= threshold]

seed = 'last{}'.format(test_size)
print ('seed: {}'.format(seed))
# data, test = train_test_split(df, train_size=.8, shuffle=True, random_state=seed)

n_data = len(data)

daysforward, daysback = pargs['daysforward'], pargs['daysback']
smart_init = pargs['smart_init'] if 'smart_init' in pargs else False

# link functions
identity_fun = lambda x: x
exp_fun = lambda x : np.exp(x)

# -------------------------------------------------------------------------
def fit_predict_plot(curve_model, xcol, ycol, data, test, func, predictdate, pargs=None, orig_ycol=None):
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
    plt.savefig(f'{output_folder}/draws_{fname}_{ycol}_{func.__name__}_{seed}.png')
    plt.clf()

    if daily:
        # plot draws - daily data/preds
        plot_draws_deriv(pipeline, predictx, derivs[func], dailycol)
        plt.savefig(f'{output_folder}/draws_{fname}_{ycol}_{derivs[func].__name__}_{seed}.png')
        plt.clf()


    # pipeline.predict
    if len(data[groupcol].unique()) > 1:
        # for each groups if multiple groups, then aggregate
        # TODO: add an option for plotting per group; then we can just predict group='all'
            # and set a boolean check below to not plot group_predictions data when not wanted
        group_predictions = pd.DataFrame()
        for grp in data[groupcol].unique():
            grp_df = pd.DataFrame(columns=[daycol, date, groupcol, f'{ycol}_pred'])
            grp_df[f'{ycol}_pred'] = pd.Series(pipeline.predict(times=predictx, predict_space=func, predict_group=grp))
            grp_df[groupcol] = grp
            grp_df[daycol] = predictx
            grp_df[date] = predictdate
            group_predictions = group_predictions.append(grp_df)
        predictions = group_predictions.groupby(daycol).sum()[f'{ycol}_pred']
    else:
        # otherwise just call predict once on all
        predictions = pipeline.predict(times=predictx, predict_space=func, predict_group='all')

    # evaluate against test set - this is only done overall, not per group
    xtest, ytest = test[xcol], test[ycol]
    predtest = pipeline.predict(times=xtest, predict_space=func, predict_group='all')
    r2, msle = r2_score(ytest, predtest), None
    # this throws an error otherwise, hence the check
    if 'log' not in func.__name__ :
        msle = mean_squared_log_error(ytest, predtest)
    maperr = mape(ytest, predtest)
    print ('test set - mape: {} r2: {} msle: {}'.format(maperr, r2, msle))

    # evaluate overall - this is only done overall, not per group
    r2, msle = r2_score(agg_data[ycol], predictions[daysback:daysback+len(agg_data[ycol])]), None
    # this throws an error otherwise, hence the check
    if 'log' not in func.__name__ :
        msle = mean_squared_log_error(agg_data[ycol], predictions[daysback:daysback+len(agg_data[ycol])])
    maperr = mape(agg_data[ycol], predictions[daysback:daysback+len(agg_data[ycol])])
    print ('overall - mape: {} r2: {} msle: {}'.format(maperr, r2, msle))

    # plot predictions against actual
    # set up the canvas
    setup_plt(ycol)
    plt.title("{} {} fit to {}".format(fname, ycol, func.__name__))
    # actually plot the data
    if smoothing_window:
        # plot original data
        plt.plot(agg_data[date], agg_data[orig_ycol], 'k+', label='data (test)')
        plt.plot(agg_test[date], agg_test[orig_ycol], 'k+', label='data (test)')
    # plot data we fit on (smoothed if -s)
    plt.plot(agg_data[date], agg_data[ycol], 'r+', label='data')
    plt.plot(agg_test[date], agg_test[ycol], 'g+', label='data (test)')
    # plot predictions
    plt.plot(predictdate, predictions, 'r-', label='fit: {}: {}'.format(func.__name__, params_estimate))
    # plot error bars based on MAPE
    plt.errorbar(predictdate[df[date].nunique():], predictions[df[date].nunique():], yerr=predictions[df[date].nunique():]*maperr, color='black', barsabove='False')
    # plot each group's curve
    clrs = ['c', 'm', 'y', 'k']
    if len(data[groupcol].unique()) > 1:
        for i, grp in enumerate(data[groupcol].unique()):
            # plot each group's predictions
            plt.plot(predictdate, group_predictions[group_predictions[groupcol] == grp][f'{ycol}_pred'], f'{clrs[i]}-', label=grp)
            # plot each group's actual data
            plt.plot(data[data[groupcol] == grp][date], data[data[groupcol] == grp][ycol], f'{clrs[i]}+', label='data')
            plt.plot(test[test[groupcol] == grp][date], test[test[groupcol] == grp][ycol], f'{clrs[i]}+')
    
    plt.legend() 
    plt.savefig(f'{output_folder}/{fname}_{ycol}_{func.__name__}_{seed}.png')
    plt.clf()

    if daily:
        # also predict daily numbers
        if len(data[groupcol].unique()) > 1:
            # per group, and then aggregate
            daily_group_predictions = pd.DataFrame()
            for grp in data[groupcol].unique():
                grp_df = pd.DataFrame(columns=[daycol, date, groupcol, f'{ycol}_pred'])
                grp_df[f'{ycol}_pred'] = pd.Series(pipeline.predict(times=predictx, predict_space=derivs[func], predict_group=grp))
                grp_df[groupcol] = grp
                grp_df[daycol] = predictx
                grp_df[date] = predictdate
                daily_group_predictions = daily_group_predictions.append(grp_df)
            daily_predictions = daily_group_predictions.groupby(daycol).sum()[f'{ycol}_pred']
        else:
            # just overall
            daily_predictions = pipeline.predict(times=predictx, predict_space=derivs[func], predict_group='all')

        # calculate mape for error bars
        maperr = mape(agg_data[dailycol], daily_predictions[daysback:daysback+len(agg_data[dailycol])])
        print(f"Daily MAPE: {maperr}")
        
        # plot daily predictions against actual
        setup_plt(ycol)
        plt.title("{} {} fit to {}".format(fname, ycol, derivs[func].__name__))
        # plot daily deaths
        plt.plot(agg_data[date], agg_data[dailycol], 'b+', label='data')
        plt.plot(agg_test[date], agg_test[dailycol], 'g+', label='data (test)')
        # against predicted daily deaths
        plt.plot(predictdate, daily_predictions, 'r-', label='fit: {}: {}'.format(derivs[func].__name__, params_estimate))
        # with error bars
        plt.errorbar(predictdate[df[date].nunique():], daily_predictions[df[date].nunique():], yerr=daily_predictions[df[date].nunique():]*maperr, color='black', barsabove='False')
        # including per group predictions if multiple
        # TODO: add per group observed data here
        if len(data[groupcol].unique()) > 1:
            for i, grp in enumerate(data[groupcol].unique()):
                plt.plot(predictdate, daily_predictions[daily_predictions[groupcol] == grp][f'{ycol}_pred'], f'{clrs[i]}-', label=grp)
        
        plt.legend() 
        plt.savefig(f'{output_folder}/{fname}_{ycol}_{derivs[func].__name__}_{seed}.png')
        # plt.show() 
        plt.clf()

    # Now, all plotting is complete. Re-acquire detailed draws information for output (csv)
    # Reliability of these numbers are questionable. Uncertainty metric evalutation ongoing.
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

# -------------------------------------------------------------------------
predictions = {}
for i, (ycol, func) in enumerate(ycols.items()):
    
    data.loc[:,'covs']            = n_data * [ 1.0 ]
    data.loc[:,'deaths_normalized']            = data['deaths']/data['deaths'].max()

    param_names  = [ 'alpha', 'beta',       'p'     ]
    covs = ['covs', 'covs', 'covs']
    link_fun     = [ identity_fun, exp_fun, exp_fun ]
    # link_fun     = [ exp_fun, identity_fun, exp_fun ] # According to their methods should be
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
            'smart_initialize': smart_init,
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
    orig_ycol = list(orig_ycols.keys())[i] if smoothing_window else None
    predictions[ycol] = fit_predict_plot(pipeline, xcol, ycol, data, test, func, predictdate, pargs=pargs, orig_ycol=orig_ycol)

results = pd.concat([df[date], pd.Series(predictdate[len(df):], name=date)], axis=0)

for ycol, (preds, lower, mean, upper) in predictions.items():
    dummy = pd.Series(np.full((len(preds)-len(df[date].unique()),), fill_value=np.nan), name='observed {}'.format(ycol))
    observed = pd.concat(
        [pd.Series(df[ycol], name='observed {}'.format(ycol)), dummy],axis=0
    ).reset_index(drop=True)
    results = pd.concat([results, pd.Series(observed)], axis=1)
    results = pd.concat([results, pd.Series(preds, name='preds {}'.format(ycol))], axis=1)
    results = pd.concat([results, pd.Series(lower, name='lower {}'.format(ycol))], axis=1)
    results = pd.concat([results, pd.Series(mean, name='mean {}'.format(ycol))], axis=1)
    results = pd.concat([results, pd.Series(upper, name='upper {}'.format(ycol))], axis=1)

results.to_csv(f"{output_folder}/{fname}_{func.__name__}_{seed}.csv", index=False)



