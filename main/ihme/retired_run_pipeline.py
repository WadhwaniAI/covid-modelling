#! /bin/python3
# vim: set expandtab:
# -------------------------------------------------------------------------
import os
import sys
import argparse
import pandas as pd
import numpy as np 
from datetime import timedelta, datetime

import curvefit
from curvefit.core.utils import data_translator
from curvefit.pipelines.basic_model import BasicModel

sys.path.append('../..')
from models.ihme.util import Params
from models.ihme.plotting import Plotter

# -------------------------------------------------------------------------
parser = argparse.ArgumentParser() 
parser.add_argument("-p", "--params", help="name of entry in params.json", required=True)
parser.add_argument("-d", "--daily", help="whether or not to plot daily", required=False, action='store_true')
parser.add_argument("-s", "--smoothing", help="how much to smooth, else no smoothing", required=False)
args = parser.parse_args() 

# load params
daily, smoothing_window = args.daily, args.smoothing
params = Params.fromjson(args.params)

# assign data
df, agg_df = params.df, params.agg_df
if smoothing_window: 
    orig_ycols = params.smoothing_init()
if daily: 
    dailycol = params.daily_init()

data, test = params.train_test(df)
multigroup = params.multigroup
agg_data, agg_test = params.train_test(agg_df)

seed = f'last{params.test_size}'
print (f'seed: {seed}')

# set vars
date, groupcol = params.date, params.groupcol
xcol, ycols = params.xcol, params.ycols
daysforward, daysback = params.daysforward, params.daysback
pipeline_run_args = params.pipeline_run_args

# output
fname = args.params
today = datetime.today()
output_folder = f'output/pipeline/{fname}/{today}'
if not os.path.exists(output_folder):
        os.makedirs(output_folder)

predictdate = pd.to_datetime(pd.Series([timedelta(days=x)+data[date].iloc[0] for x in range(-daysback,daysforward)]))
predictx = np.array([x+1 for x in range(-daysback,daysforward)])

# link functions
identity_fun = lambda x: x
exp_fun = lambda x : np.exp(x)

# -------------------------------------------------------------------------
def fit_predict_plot(curve_model, xcol, ycol, data, test, func, pargs={}, orig_ycol=None):
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
    
    # pipeline
    pipeline.setup_pipeline()
    pipeline.run(n_draws=p_args['n_draws'], prediction_times=predictx, 
        cv_threshold=p_args['cv_threshold'], smoothed_radius=p_args['smoothed_radius'], 
        num_smooths=p_args['num_smooths'], exclude_groups=p_args['exclude_groups'], 
        exclude_below=p_args['exclude_below'], exp_smoothing=p_args['exp_smoothing'], 
        max_last=p_args['max_last']
    )
    params_estimate = pipeline.mod.params
    print(params_estimate)
    dailycolname = dailycol.format(ycol=ycol) if daily else None

    plotter = Plotter(pipeline, params, predictdate, predictx, f'{fname}_{seed}', output_folder, ycol, func)
    plotter.plot_draws(dailycolname=dailycolname)

    # plot_prediction calls these functions:
        # group_predictions, predictions = predict(func, multigroup)
        # calc_error(test, predictions, agg_data, daysback)
    plotter.plot_predictions(df, agg_data, agg_test, orig_ycol, test, daysback, smoothing_window, multigroup, dailycolname)

    # Now, all plotting is complete. Re-acquire detailed draws information for output (csv)
    # Reliability of these numbers are questionable. Uncertainty metric evalutation ongoing.
    for group in pipeline.groups:
        # x = prediction_times = predictx
        draws = pipeline.draws[group].copy()
        draws = data_translator(
            data=draws,
            input_space=pipeline.predict_space,
            output_space=pipeline.predict_space
        )
        mean_fit = pipeline.mean_predictions[group].copy() # predictions
        mean_fit = data_translator(
            data=mean_fit,
            input_space=pipeline.predict_space,
            output_space=pipeline.predict_space
        )
        mean = draws.mean(axis=0)
        # uncertainty
        lower = np.quantile(draws, axis=0, q=0.025)
        upper = np.quantile(draws, axis=0, q=0.975)

    return mean_fit, lower, mean, upper 

# -------------------------------------------------------------------------
predictions = {}
for i, (ycol, func) in enumerate(ycols.items()):
    
    data.loc[:,'covs'] = len(data) * [ 1.0 ]
    data.loc[:,'sd'] = data[date].apply(lambda x: [1.0 if x >= datetime(2020, 3, 24) else 0.0]).tolist()
    data.loc[:,f'{ycol}_normalized'] = data[ycol]/data[ycol].max()

    param_names  = [ 'alpha', 'beta', 'p' ]
    covs = ['covs', 'covs', 'covs']
    # link_fun = [ identity_fun, exp_fun, exp_fun ]
    link_fun = [ exp_fun, identity_fun, exp_fun ] # According to their methods should be
    var_link_fun = [ identity_fun, identity_fun, identity_fun ]
    # var_link_fun = link_fun

    # # think this could work with more death data:
    # link_fun     = [ identity_fun, identity_fun, exp_fun ]
    # covs = ['covs', 'deaths_normalized', 'covs']

    # python3 run_pipeline.py -p pune -d -s 3;python3 run_pipeline.py -p amd -d -s 3;python3 run_pipeline.py -p mumbai -d;python3 run_pipeline.py -p jaipur -d -s 3;python3 run_pipeline.py -p bengaluru -d -s 3;python3 run_pipeline.py -p delhi -d -s 3

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
        fit_dict=params.priors, #: keyword arguments to CurveModel.fit_params()
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

    orig_ycol = list(orig_ycols.keys())[i] if smoothing_window else None
    predictions[ycol] = fit_predict_plot(pipeline, xcol, ycol, data, test, func, pargs=pipeline_run_args, orig_ycol=orig_ycol)

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



