import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

import curvefit
from curvefit.core.functions import *
from curvefit.pipelines.basic_model import BasicModel

sys.path.append('../..')
from utils.age_standardisation import standardise_age, census_age
from models.ihme.pipeline import WAIPipeline
from models.ihme.util import Params
from models.ihme.data import get_district_timeseries_cached

from sklearn.metrics import r2_score, mean_squared_log_error
from models.ihme.util import mape
from models.ihme.plotting import setup_plt

parser = argparse.ArgumentParser() 
parser.add_argument("-l", "--log", help="fit on log", required=False, action='store_true')
parser.add_argument("-a", "--age", help="perform age-standardisation", required=False, action='store_true')
parser.add_argument("-sd", "--sd", help="use social distance covariate", required=False, action='store_true')
args = parser.parse_args()

districts = ['Mumbai', ['Bengaluru Urban', 'Bengaluru Rural'], 'Ahmadabad', 'Jaipur', 'Pune', 'all']
states = ['Maharashtra', 'Karnataka', 'Gujarat', 'Rajasthan', 'Maharashtra', 'Delhi']

age_data = census_age()

amd = 'District - Ahmadabad (07)'
mumbai = 'District - Mumbai (23)'
mumbai2 = 'District - Mumbai Suburban (22)'
pune = 'District - Pune (25)'
delhi ='District - New Delhi (05)'
jaipur = 'District - Jaipur (12)'
bengaluru = 'District - Bangalore (18)'

# -------------------

district_timeseries = get_district_timeseries_cached('Mumbai', 'Maharashtra')
df = standardise_age(district_timeseries, age_data, 'Mumbai', 'Maharashtra', [mumbai, mumbai2]).reset_index(col_fill='date')

ycol = 'age_std' if args.age else 'non_std'
if args.log:
    obs = 'log_non_std'
    func = log_erf
    ycol = f'log_{ycol}' 
    label = 'log_age_std'
else:
    obs = 'non_std'
    func = erf
    label = 'age_std'

df = df.loc[df['age_std'].gt(1e-15).idxmax():,:]
df['day'] = [x for x in range(len(df))]
df['date']= pd.to_datetime(df['date']) 
df['log_age_std'] = df['age_std'].apply(np.log)
df['log_non_std'] = df['non_std'].apply(np.log)
df.loc[:,'group'] = len(df) * [ 1.0 ]
df.loc[:,'covs'] = len(df) * [ 1.0 ]
df.loc[:,'sd'] = df['date'].apply(lambda x: [1.0 if x >= datetime(2020, 3, 24) else 0.0]).tolist()
df.loc[:,f'{ycol}_normalized'] = df[ycol]/df[ycol].max()

params = Params.fromdefault(df, {
    'groupcol': 'group',
    'xcol': 'day',
    'ycols': {
        ycol: func,
    }
}, default_label=label)

# set vars
fname = 'age_std_mumbai_deaths'
dtp = 3069834 # mumbai district population

priors = {
    "fe_init": [0.01, 1.4, -500],
    "fe_bounds": [[0, 1], [1, 100], [-20, -1]]
}

pipeline_run_args = {
    "n_draws": 20,
    "cv_threshold": 1e-2,
    "smoothed_radius": [7,7],
    "num_smooths": 3,
    "exclude_groups": [],
    "exclude_below": 0,
    "exp_smoothing": None,
    "max_last": None,
}

# output
today = datetime.today()
output_folder = f'output/pipeline/{fname}/{today}'
if not os.path.exists(output_folder):
        os.makedirs(output_folder)

param_names  = [ 'alpha', 'beta', 'p' ]
# link functions
identity_fun = lambda x: x
exp_fun = lambda x : np.exp(x)
link_fun = [ exp_fun, identity_fun, exp_fun ]
var_link_fun = [ identity_fun, identity_fun, identity_fun ]

xcol, groupcol = params.xcol, params.groupcol
daysback, daysforward = params.daysback, params.daysforward
predictdate = pd.to_datetime(pd.Series([timedelta(days=x)+df['date'].iloc[0] for x in range(-daysback,daysforward)]))
predictx = np.array([x+1 for x in range(-daysback,daysforward)])


if args.sd:
    covs = ['covs', 'sd', 'covs']
else:
    covs = ['covs', 'covs', 'covs']

pipeline = BasicModel(
    all_data=df, #: (pd.DataFrame) of *all* the data that will go into this modeling pipeline
    col_t=xcol, #: (str) name of the column with time
    col_group=groupcol, #: (str) name of the column with the group in it
    col_obs=ycol, #: (str) the name of the column with observations for fitting the model
    col_obs_compare=ycol, #TODO: (str) the name of the column that will be used for predictive validity comparison
    all_cov_names=covs, #TODO: List[str] list of name(s) of covariate(s). Not the same as the covariate specifications
    fun=func, #: (callable) the space to fit in, one of curvefit.functions
    predict_space=func, #TODO confirm: (callable) the space to do predictive validity in, one of curvefit.functions
    obs_se_func=None, #TODO if we wanna specify: (optional) function to get observation standard error from col_t
    fit_dict=priors, #: keyword arguments to CurveModel.fit_params()
    basic_model_dict= { #: additional keyword arguments to the CurveModel class
        'col_obs_se': None,#(str) of observation standard error
        'col_covs': [[cov] for cov in covs],#TODO: List[str] list of names of covariates to put on the parameters
        'param_names': param_names,#(list{str}):
        'link_fun': link_fun,#(list{function}):
        'var_link_fun': var_link_fun,#(list{function}):
    },
)

def fit_predict_plot(curve_model, xcol, ycol, data, func, pargs={}, orig_ycol=None):
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

    # plot draws
    pipeline.plot_results(prediction_times=predictx)
    plt.legend()
    plt.savefig(f'{output_folder}/{fname}_draws_{ycol}_{func.__name__}.png')
    plt.clf()

    predictions = pipeline.predict(times=predictx, predict_space=func, predict_group='all')
    # evaluate overall - this is only done overall, not per group
    r2, msle = r2_score(df[ycol].values, predictions[daysback:daysback+len(df[ycol])]), None
    # this throws an error otherwise, hence the check
    # if 'log' not in func.__name__ :
    #     msle = mean_squared_log_error(df[ycol].values, predictions[daysback:daysback+len(df[ycol])])
    maperr_overall = mape(df[ycol].values, predictions[daysback:daysback+len(df[ycol])])
    print ('overall - mape: {} r2: {} msle: {}'.format(maperr_overall, r2, msle))
    
    title = f'{fname} {ycol}' +  ' fit to {}'
    # plot predictions against actual
    # set up the canvas
    setup_plt(ycol)
    plt.yscale("linear")
    plt.title(title.format(func.__name__))
    # actually plot the data
    plt.plot(df['date'], df[ycol], 'r+', label='age standardized')
    plt.plot(df['date'], df[obs], 'b+', label='observed (non-standardized)')
    # plot predictions
    plt.plot(predictdate, predictions, 'r-', label='fit: {}: {}'.format(func.__name__, pipeline.mod.params))
    plt.legend()
    plt.savefig(f'{output_folder}/{fname}_{ycol}_{func.__name__}.png')
    plt.clf()

fit_predict_plot(pipeline, xcol, ycol, df, func, pargs=pipeline_run_args)