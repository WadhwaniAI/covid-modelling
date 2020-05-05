import os
import sys
import argparse
import pandas as pd
import numpy as np 
from datetime import timedelta, datetime

import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from matplotlib.dates import DateFormatter
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_log_error

import curvefit
from curvefit.pipelines.basic_model import BasicModel
from curvefit.core.functions import *
from curvefit.core.utils import data_translator

from models.ihme.util import mape, smooth, get_daily_vals
from models.ihme.plotting import setup_plt

class WAIPipeline():
    def __init__(self, df, ycol, params, covs, file_prefix, smoothing=None, predict_space=None):
        # get from params object
        self.file_prefix = file_prefix
        self.xcol, self.ycol, self.func = params.xcol, ycol, params.ycols[ycol]
        self.date, self.groupcol = params.date, params.groupcol
        test_size =  params.test_size
        priors, self.pipeline_args = params.priors, params.pipeline_run_args
        self.smoothing = smoothing
        param_names  = [ 'alpha', 'beta', 'p' ]
        predict_space = predict_space if predict_space else self.func
        
        # link functions
        identity_fun = lambda x: x
        exp_fun = lambda x : np.exp(x)
        link_fun = [ exp_fun, identity_fun, exp_fun ]
        var_link_fun = [ identity_fun, identity_fun, identity_fun ]

        self.df = df
        self.agg_df = self.df.groupby(self.date).sum().reset_index(col_fill=self.date)
        threshold = self.df[self.date].max() - timedelta(days=test_size)
        
        self.deriv_col = f'daily_{self.ycol}'
        self._daily_init()
        
        if self.smoothing:
            self.orig_ycol = self.ycol
            self.df[:, self.orig_ycol] = self.df[self.ycol]
            
            self.ycol = f'{self.ycol}_smooth'
            self.df[:, self.ycol] = smooth(self.df[self.ycol], smoothing)
        
        
        self.data, self.test = self._train_test_split(self.df, threshold)
        self.agg_data, self.agg_test = self._train_test_split(self.agg_df, threshold)
        self.multigroup = len(df[self.groupcol].unique()) > 1
        self.daysback, self.daysforward = params.daysback, params.daysforward
        self.predictdate = pd.to_datetime(pd.Series([timedelta(days=x)+self.data[self.date].iloc[0] for x in range(-self.daysback,self.daysforward)]))
        self.predictx = np.array([x+1 for x in range(-self.daysback,self.daysforward)])

        self.pipeline = BasicModel(
            all_data=self.data, #: (pd.DataFrame) of *all* the data that will go into this modeling pipeline
            col_t=self.xcol, #: (str) name of the column with time
            col_group=self.groupcol, #: (str) name of the column with the group in it
            col_obs=self.ycol, #: (str) the name of the column with observations for fitting the model
            col_obs_compare=self.ycol, #TODO: (str) the name of the column that will be used for predictive validity comparison
            all_cov_names=covs, #TODO: List[str] list of name(s) of covariate(s). Not the same as the covariate specifications
            fun=self.func, #: (callable) the space to fit in, one of curvefit.functions
            predict_space=predict_space, #TODO confirm: (callable) the space to do predictive validity in, one of curvefit.functions
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

        self.derivs = {
            erf: derf,
            # gaussian_cdf: gaussian_pdf,
            log_erf: log_derf,
        }
    
    def run(self, pipeline_args=None):
        p_args = self.pipeline_args
        if pipeline_args is not None:
            p_args.update(pipeline_args)
        
        # pipeline
        self.pipeline.setup_pipeline()
        self.pipeline.run(n_draws=p_args['n_draws'], prediction_times=self.predictx, 
            cv_threshold=p_args['cv_threshold'], smoothed_radius=p_args['smoothed_radius'], 
            num_smooths=p_args['num_smooths'], exclude_groups=p_args['exclude_groups'], 
            exclude_below=p_args['exclude_below'], exp_smoothing=p_args['exp_smoothing'], 
            max_last=p_args['max_last']
        )
        return self.pipeline

    def predict(self, predict_space=None):
        predict_space = predict_space if predict_space else self.pipeline.predict_space
        # pipeline.predict
        if self.multigroup:
            # for each groups if multiple groups, then aggregate
            # TODO: add an option for plotting per group; then we can just predict group='all'
                # and set a boolean check below to not plot group_predictions data when not wanted
            group_predictions = pd.DataFrame()
            for grp in self.data[self.groupcol].unique():
                grp_df = pd.DataFrame(columns=[self.xcol, self.date, self.groupcol, f'{self.ycol}_pred'])
                grp_df[f'{self.ycol}_pred'] = pd.Series(self.pipeline.predict(times=self.predictx, predict_space=predict_space, predict_group=grp))
                grp_df[self.groupcol] = grp
                grp_df[self.xcol] = self.predictx
                grp_df[self.date] = self.predictdate
                group_predictions = group_predictions.append(grp_df)
            predictions = group_predictions.groupby(self.xcol).sum()[f'{self.ycol}_pred']
        else:
            # otherwise just call predict once on all
            group_predictions = None
            predictions = self.pipeline.predict(times=self.predictx, predict_space=predict_space, predict_group='all')

        return predictions, group_predictions
    
    def calc_error(self, predictions):
        # evaluate against test set - this is only done overall, not per group
        ytest = self.test[self.pipeline.col_obs]
        ypred = self._predict_test(self.pipeline.fun)
        r2_test, msle_test, maperr_test = self._calc_error(ytest, ypred)
        print ('test set - mape: {} r2: {} msle: {}'.format(maperr_test, r2_test, msle_test))

        # evaluate overall - this is only done overall, not per group
        pred = predictions[self.daysback:self.daysback+len(self.agg_data[self.pipeline.col_obs])]
        r2_overall, msle_overall, maperr_overall = self._calc_error(self.agg_data[self.ycol], pred)
        print ('overall - mape: {} r2: {} msle: {}'.format(maperr_overall, r2_overall, msle_overall))
        
        return {
            "test": {
                "r2": r2_test,
                "mape": maperr_test,
                "msle": msle_test
            },
            "overall": {
                "r2": r2_overall,
                "mape": maperr_overall,
                "msle": msle_overall
            }
        }

    def plot_results(self, predictions, group_predictions=None, ycol=None):
        ycol = ycol if ycol else self.ycol
        maperr = self.calc_error(predictions)['test']['mape']
        title = f'{self.file_prefix} {ycol}' +  ' fit to {}'
        # plot predictions against actual
        setup_plt(ycol)
        plt.title(title.format(self.func.__name__))
        # plot data we fit on
        plt.plot(self.agg_data[self.date], self.agg_data[ycol], 'r+', label='data')
        plt.plot(self.agg_test[self.date], self.agg_test[ycol], 'g+', label='data (test)')
        if self.smoothing:
            plt.plot(self.agg_df[self.date], self.agg_df[self.orig_ycol], 'k+', label='unsmoothed data')
        # plot predictions
        plt.plot(self.predictdate, predictions, 'r-', 
            label='fit: {}: {}'.format(self.func.__name__, self.pipeline.mod.params))
        # plot error bars based on MAPE
        plt.errorbar(self.predictdate[self.df[self.date].nunique():], 
            predictions[self.df[self.date].nunique():], 
            yerr=predictions[self.df[self.date].nunique():]*maperr, color='black', barsabove='False')
        
        # plot per group
        clrs = ['c', 'm', 'y', 'k']
        if self.multigroup and group_predictions is not None:
            for i, grp in enumerate(self.data[self.groupcol].unique()):
                # plot each group's predictions
                plt.plot(self.predictdate, group_predictions[group_predictions[self.groupcol] == grp][f'{ycol}_pred'], f'{clrs[i]}-', label=grp)
                # plot each group's actual data
                grpdata = self.data[self.data[self.groupcol] == grp]
                grptest = self.test[self.test[self.groupcol] == grp]
                plt.plot(grpdata[self.date], grpdata[ycol], f'{clrs[i]}+', label=f'{grp}_data')
                plt.plot(grptest[self.date], grptest[ycol], f'{clrs[i]}+', label=f'{grp}_test')
        
        plt.legend()
        return

    def plot_derivative(self):
        predictions, group_predictions = self.predict(predict_space=self.derivs[self.pipeline.fun])
        self.plot_results(predictions, group_predictions=group_predictions, ycol=self.deriv_col)
        return     
    
    def all_plots(self, output_folder, predictions, 
                ycol=None, group_predictions=None, deriv=False):
        self._plot_draws(self.pipeline.fun)
        plt.savefig(f'{output_folder}/{self.file_prefix}_{self.pipeline.col_obs}_{self.func.__name__}_draws.png')
        plt.clf()
        self.plot_results(predictions, group_predictions=None, ycol=ycol)
        plt.savefig(f'{output_folder}/{self.file_prefix}_{self.pipeline.col_obs}_{self.func.__name__}.png')
        plt.clf()
        if deriv:
            self._plot_draws(self.derivs[self.pipeline.fun])
            plt.savefig(f'{output_folder}/{self.file_prefix}_{self.pipeline.col_obs}_deriv_{self.derivs[self.func].__name__}_draws.png')
            plt.clf()
            self.plot_derivative()
            plt.savefig(f'{output_folder}/{self.file_prefix}_{self.pipeline.col_obs}_deriv_{self.derivs[self.func].__name__}.png')
            plt.clf()

    def lograte_to_cumulative(self, predictions, population, output_folder, 
                group_predictions=None):
        predicted_cumulative_deaths = np.exp(predictions) * population
        self.df.loc[:, 'cumulative_deaths'] = np.exp(self.df[self.ycol]) * population
        self.data.loc[:, 'cumulative_deaths'] = np.exp(self.data[self.ycol]) * population
        self.agg_data.loc[:, 'cumulative_deaths'] = np.exp(self.agg_data[self.ycol]) * population
        self.test.loc[:, 'cumulative_deaths'] = np.exp(self.test[self.ycol]) * population
        self.agg_test.loc[:, 'cumulative_deaths'] = np.exp(self.agg_test[self.ycol]) * population
        # if group_predictions is not None:
        #     cumulative_group_predictions = np.exp(group_predictions) * population
        # predicted_daily_deaths = pd.DataFrame(get_daily_vals(predicted_cumulative_deaths, predicted_cumulative_deaths.columns[0]))
        self.all_plots(output_folder, predicted_cumulative_deaths, ycol='cumulative_deaths', group_predictions=group_predictions)
        # self.plot_results(predicted_cumulative_deaths, group_predictions=cumulative_group_predictions, ycol=predicted_cumulative_deaths.name)
        # self.plot_results(predicted_daily_deaths, group_predictions=None, ycol=predicted_daily_deaths.name)

    def rate_to_cumulative(self, predictions, population, output_folder, 
                group_predictions=None):
        predicted_cumulative_deaths = np.exp(predictions) * population
        self.df.loc[:, 'cumulative_deaths'] = self.df[self.ycol] * population
        self.data.loc[:, 'cumulative_deaths'] = self.data[self.ycol] * population
        self.agg_data.loc[:, 'cumulative_deaths'] = self.agg_data[self.ycol] * population
        self.test.loc[:, 'cumulative_deaths'] = self.test[self.ycol] * population
        self.agg_test.loc[:, 'cumulative_deaths'] = self.agg_test[self.ycol] * population
        # if group_predictions is not None:
        #     cumulative_group_predictions = np.exp(group_predictions) * population
        # predicted_daily_deaths = pd.DataFrame(get_daily_vals(predicted_cumulative_deaths, predicted_cumulative_deaths.columns[0]))
        self.all_plots(output_folder, predicted_cumulative_deaths, ycol='cumulative_deaths', group_predictions=group_predictions)
        # self.plot_results(predicted_cumulative_deaths, group_predictions=cumulative_group_predictions, ycol=predicted_cumulative_deaths.name)
        # self.plot_results(predicted_daily_deaths, group_predictions=None, ycol=predicted_daily_deaths.name)

    def _plot_draws(self, draw_space):
        _, ax = plt.subplots(len(self.pipeline.groups), 1, figsize=(8, 4 * len(self.pipeline.groups)),
                                sharex=True, sharey=True)
        setup_plt(self.pipeline.col_obs)
        plt.yscale("linear")
        if len(self.pipeline.groups) == 1:
            ax = [ax]
        for i, group in enumerate(self.pipeline.groups):
            draws = self.pipeline.draws[group].copy()
            draws = data_translator(
                data=draws,
                input_space=self.pipeline.predict_space,
                output_space=draw_space
            )
            mean_fit = self.pipeline.mean_predictions[group].copy()
            mean_fit = data_translator(
                data=mean_fit,
                input_space=self.pipeline.predict_space,
                output_space=draw_space
            )
            mean = draws.mean(axis=0)
            ax[i].plot(self.predictdate, mean, c='red', linestyle=':')
            ax[i].plot(self.predictdate, mean_fit, c='black')

            lower = np.quantile(draws, axis=0, q=0.025)
            upper = np.quantile(draws, axis=0, q=0.975)
            lower_nonneg = lower.copy()
            lower_nonneg[lower_nonneg < 0] = 0
            ax[i].plot(self.predictdate, lower_nonneg, c='red', linestyle=':')
            ax[i].plot(self.predictdate, upper, c='red', linestyle=':')

            df_data = self.pipeline.all_data.loc[self.pipeline.all_data[self.pipeline.col_group] == group].copy()
            ax[i].scatter(df_data[self.date], df_data[self.pipeline.col_obs])

            ax[i].set_title(f"{group} predictions")

    def _daily_init(self):
        self.df.sort_values(self.groupcol)
        self.agg_df[self.deriv_col] = get_daily_vals(self.agg_df, self.ycol)
        dailycol_dfs = [get_daily_vals(self.df[self.df[self.groupcol] == grp], self.ycol) for grp in self.df[self.groupcol].unique()]
        self.df[self.deriv_col] = pd.concat(dailycol_dfs)

    def _train_test_split(self, df, threshold):
            return df[df[self.date] < threshold], df[df[self.date] >= threshold]

    def _predict_test(self, predict_space=None):
            predict_space = predict_space if predict_space else self.pipeline.predict_space
            return self.pipeline.predict(times=self.test[self.pipeline.col_t], predict_space=predict_space, predict_group='all')        

    def _calc_error(self, actual, pred):
        r2 = r2_score(actual, pred)
        try:
            msle = mean_squared_log_error(actual, pred)
        except Exception as e:
            msle = None
            print("Could not compute MSLE: {}".format(e))
        maperr = mape(actual, pred)
        return r2, msle, maperr
        