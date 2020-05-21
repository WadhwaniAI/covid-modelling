import os
import sys
import argparse
import pandas as pd
import numpy as np 
from datetime import timedelta, datetime
from copy import copy

import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from matplotlib.dates import DateFormatter
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_log_error

import curvefit
from curvefit.pipelines.basic_model import BasicModel
from curvefit.core.functions import *
from curvefit.core.utils import data_translator

# from models.ihme.util import smooth, get_daily_vals
from models.ihme.util import get_daily_vals
from utils.util import smooth, rollingavg

# class IHME(ModelWrapperBase):
class IHME():

    def __init__(self, model_parameters: dict):
        self.model_parameters = model_parameters
        self.xcol = model_parameters.get('xcol')
        self.ycol = model_parameters.get('ycol')
        self.func = model_parameters.get('func')
        self.date = model_parameters.get('date')
        self.groupcol = model_parameters.get('groupcol')
        self.priors = model_parameters.get('priors')
        self.pipeline_args = model_parameters.get('pipeline_args')
        self.smoothing = model_parameters.get('smoothing')
        # self.daysback, self.daysforward = model_parameters.get('daysback'), model_parameters.get('daysforward')
        self.covs = model_parameters.get('covs')

        self.param_names  = [ 'alpha', 'beta', 'p' ]
        self.predict_space = model_parameters.get('predict_space') if model_parameters.get('predict_space') else self.func

        # link functions
        identity_fun = lambda x: x
        exp_fun = lambda x : np.exp(x)
        self.link_fun = [ exp_fun, identity_fun, exp_fun ]
        self.var_link_fun = [ identity_fun, identity_fun, identity_fun ]

        # self.data = df
        # self.agg_data = self.data.groupby(self.date).sum().reset_index(col_fill=self.date)
        
        self.deriv_col = f'daily_{self.ycol}'
        # self._daily_init()
        
        # if self.smoothing:
        #     self.orig_ycol = self.ycol
        #     self.df[:, self.orig_ycol] = self.df[self.ycol]
            
        #     self.ycol = f'{self.ycol}_smooth'
        #     self.df[:, self.ycol] = rollingavg(self.df[self.ycol], self.smoothing)
        
        # self.predictdate = pd.to_datetime(pd.Series([timedelta(days=x)+self.data[self.date].iloc[0] for x in range(-self.daysback,self.daysforward)]))
        # self.predictx = np.array([x+1 for x in range(-self.daysback,self.daysforward)])

        self.pipeline = None

        self.derivs = {
            erf: derf,
            # gaussian_cdf: gaussian_pdf,
            log_erf: log_derf,
        }

    # def supported_forecast_variables(self):
    #     return [ForecastVariable.deceased]

    def predict(self, start_date: str, end_date: str, **kwargs):
        # vars
        n_days = (end_date - start_date).days + 1
        self.run(n_days)
        # n_days = (datetime.strptime(end_date, "%m/%d/%y") - datetime.strptime(start_date, "%m/%d/%y")).days + 1
        predictx = np.array([x+1 for x in range(n_days)])
        return self.pipeline.predict(times=predictx, predict_space=self.predict_space, predict_group='all')

    def fit(self, data: pd.DataFrame):
        self.pipeline = BasicModel(
            all_data=data, #: (pd.DataFrame) of *all* the data that will go into this modeling pipeline
            col_t=self.xcol, #: (str) name of the column with time
            col_group=self.groupcol, #: (str) name of the column with the group in it
            col_obs=self.ycol, #: (str) the name of the column with observations for fitting the model
            col_obs_compare=self.ycol, #TODO: (str) the name of the column that will be used for predictive validity comparison
            all_cov_names=self.covs, #TODO: List[str] list of name(s) of covariate(s). Not the same as the covariate specifications
            fun=self.func, #: (callable) the space to fit in, one of curvefit.functions
            predict_space=self.predict_space, #TODO confirm: (callable) the space to do predictive validity in, one of curvefit.functions
            obs_se_func=None, #TODO if we wanna specify: (optional) function to get observation standard error from col_t
            fit_dict=self.priors, #: keyword arguments to CurveModel.fit_params()
            basic_model_dict= { #: additional keyword arguments to the CurveModel class
                'col_obs_se': None,#(str) of observation standard error
                'col_covs': [[cov] for cov in self.covs],#TODO: List[str] list of names of covariates to put on the parameters
                'param_names': self.param_names,#(list{str}):
                'link_fun': self.link_fun,#(list{function}):
                'var_link_fun': self.var_link_fun,#(list{function}):
            },
        )
      
    def is_black_box(self):
        # what is this supposed to mean
        return True

    def run(self, n_days, pipeline_args=None):
        p_args = self.pipeline_args
        if pipeline_args is not None:
            p_args.update(pipeline_args)
        
        # pipeline
        self.pipeline.setup_pipeline()
        predictx = pd.Series(range(1, 1 + n_days))
        self.pipeline.run(n_draws=p_args['n_draws'], prediction_times=predictx, 
            cv_threshold=p_args['cv_threshold'], smoothed_radius=p_args['smoothed_radius'], 
            num_smooths=p_args['num_smooths'], exclude_groups=p_args['exclude_groups'], 
            exclude_below=p_args['exclude_below'], exp_smoothing=p_args['exp_smoothing'], 
            max_last=p_args['max_last']
        )
        return self.pipeline

    # def _daily_init(self):
    #     self.df.sort_values(self.groupcol)
    #     self.agg_df[self.deriv_col] = get_daily_vals(self.agg_df, self.ycol)
    #     dailycol_dfs = [get_daily_vals(self.df[self.df[self.groupcol] == grp], self.ycol) for grp in self.df[self.groupcol].unique()]
    #     self.df.loc[:, self.deriv_col] = pd.concat(dailycol_dfs)

    def calc_draws(self):
        draws_dict = {}
        for group in self.pipeline.groups:
            draws = self.pipeline.draws[group].copy()
            draws = data_translator(
                data=draws,
                input_space=self.pipeline.predict_space,
                output_space=self.predict_space
            )
            mean_fit = self.pipeline.mean_predictions[group].copy()
            mean_fit = data_translator(
                data=mean_fit,
                input_space=self.pipeline.predict_space,
                output_space=self.predict_space
            )
            # mean = draws.mean(axis=0)

            lower = np.quantile(draws, axis=0, q=0.025)
            upper = np.quantile(draws, axis=0, q=0.975)
            draws_dict[group] =  {
                'lower': lower, 
                'upper': upper
            }
        return draws_dict

    def generate(self):
        out = IHME(self.model_parameters)
        out.xcol = self.xcol
        out.ycol = self.ycol
        out.func = self.func
        out.date = self.date
        out.groupcol = self.groupcol
        out.priors = copy(self.priors)
        out.pipeline_args = copy(self.pipeline_args)
        out.smoothing = self.smoothing
        out.covs = copy(self.covs)

        out.param_names  = copy(self.param_names)
        out.predict_space = self.predict_space

        out.link_fun = copy(self.link_fun)
        out.var_link_fun = copy(self.var_link_fun)
        
        out.deriv_col = self.deriv_col
        out.derivs = copy(self.derivs)
        
        out.pipeline = None
        return out
