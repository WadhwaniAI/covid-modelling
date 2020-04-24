from curvefit.core.utils import data_translator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
from matplotlib.dates import DateFormatter
import curvefit
import data as dataloader
from datetime import datetime, timedelta
import json
import sys


def plot_draws_deriv(generator, prediction_times, draw_space, plot_obs, sharex=True, sharey=True, plot_uncertainty=True):
    _, ax = plt.subplots(len(generator.groups), 1, figsize=(8, 4 * len(generator.groups)),
                               sharex=sharex, sharey=sharey)
    if len(generator.groups) == 1:
        ax = [ax]
    for i, group in enumerate(generator.groups):
        draws = generator.draws[group].copy()
        draws = data_translator(
            data=draws,
            input_space=generator.predict_space,
            output_space=draw_space
        )
        mean_fit = generator.mean_predictions[group].copy()
        mean_fit = data_translator(
            data=mean_fit,
            input_space=generator.predict_space,
            output_space=draw_space
        )
        mean = draws.mean(axis=0)
        ax[i].plot(prediction_times, mean, c='red', linestyle=':')
        ax[i].plot(prediction_times, mean_fit, c='black')

        if plot_uncertainty:
            lower = np.quantile(draws, axis=0, q=0.025)
            upper = np.quantile(draws, axis=0, q=0.975)
            ax[i].plot(prediction_times, lower, c='red', linestyle=':')
            ax[i].plot(prediction_times, upper, c='red', linestyle=':')

        if plot_obs is not None:
            df_data = generator.all_data.loc[generator.all_data[generator.col_group] == group].copy()
            ax[i].scatter(df_data[generator.col_t], df_data[plot_obs])

        ax[i].set_title(f"{group} predictions")

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))

def smooth(y, smoothing_window):
    box = np.ones(smoothing_window)/smoothing_window
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

# def sma(y, smoothing_window):
#     return y.rolling(window=smoothing_window)

# def ema(y, smoothing_window):
#     return y.ewm(span=smoothing_window, adjust=False)

def setup_plt(ycol):
    register_matplotlib_converters()
    plt.yscale("log")
    plt.gca().xaxis.set_major_formatter(DateFormatter("%d.%m"))
    plt.grid()
    plt.xlabel("Date")
    plt.ylabel(ycol)

class Params():
    def __init__(self, label):
        with open('params.json', "r") as paramsfile:
            pargs = json.load(paramsfile)
            if label not in pargs:
                print("entry not found in params.json")
                sys.exit(0)
        pargs = pargs[label]

        # set vars
        self.params_true = np.array( [ pargs['alpha_true'], pargs['beta_true'], pargs['p_true'] ] )
        
        self.date, self.groupcol, self.xcol = pargs['date'], pargs['groupcol'], 'day'
        self.xcol, self.ycols = pargs['xcol'], pargs['ycols']
        self.test_size=pargs['test_size']

        self.pipeline_run_args = {
            "n_draws": pargs["n_draws"],
            "cv_threshold": pargs["cv_threshold"],
            "smoothed_radius": pargs["smoothed_radius"], 
            "num_smooths": pargs["num_smooths"],
            "exclude_groups": pargs["exclude_groups"],
            "exclude_below": pargs["exclude_below"],
            "exp_smoothing": pargs["exp_smoothing"],
            "max_last": pargs["max_last"],
        }

        self.daysforward, self.daysback = pargs['daysforward'], pargs['daysback']
        self.smart_init = pargs['smart_init'] if 'smart_init' in pargs else False

        # convert str to functions
        for (k,v) in self.ycols.items():
            self.ycols[k] = getattr(curvefit.core.functions, v)

        # load data
        data_func = getattr(dataloader, pargs['data_func'])
        if 'data_func_args' in pargs:
            self.df = data_func(pargs['data_func_args'])
        else:
            self.df = data_func()
        
        self.multigroup = len(self.df[self.groupcol].unique()) > 1
        self.agg_df = self.df.groupby(self.date).sum().reset_index(col_fill=self.date)

        self.threshold = self.df[self.date].max() - timedelta(days=self.test_size)
         
    def train_test(self, df):
        data, test = df[df[self.date] < self.threshold], df[df[self.date] >= self.threshold]
        return data, test

    def smoothing_init(self):
        new_ycols = {}
        for ycol in self.ycols.keys():
            self.df[f'{ycol}_smooth'] = smooth(self.df[ycol], 5)
            self.agg_df[f'{ycol}_smooth'] = smooth(self.agg_df[ycol], 5)
            new_ycols[f'{ycol}_smooth'] = self.ycols[ycol]
        orig_ycols = self.ycols
        self.ycols = new_ycols 
        return orig_ycols

    def daily_init(self):
        self.df.sort_values(self.groupcol)
        dailycol = "daily_{ycol}"
        for ycol in self.ycols.keys():
            self.agg_df[dailycol.format(ycol=ycol)] = self.agg_df[ycol] - self.agg_df[ycol].shift(1)
            dailycol_dfs = [self.df[self.df[self.groupcol] == grp][ycol] - self.df[self.df[self.groupcol] == grp][ycol].shift(1) for grp in self.df[self.groupcol].unique()]
            self.df[dailycol.format(ycol=ycol)] = pd.concat(dailycol_dfs)
        return dailycol
