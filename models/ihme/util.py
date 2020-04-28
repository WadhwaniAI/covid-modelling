import sys
import json
import numpy as np
import pandas as pd

from datetime import timedelta

import curvefit
from . import data as dataloader

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

class Params():
    def __init__(self, label):
        with open('params.json', "r") as paramsfile:
            params = json.load(paramsfile)
            if label not in params:
                print("entry not found in params.json")
                sys.exit(0)
        pargs = params['default']
        pargs.update(params[label])
        

        # set vars
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
        self.priors = pargs['priors']
        # "priors": {
        #     "fe_init": [2.0, 3.0, 4.0],
		# 	  "smart_initialize": true
		# 	  "re_init": null,
		# 	  "fe_bounds": null,
		# 	  "re_bounds": null,
		# 	  "fe_gprior": null,
		# 	  "re_gprior": null,
		# 	  "fun_gprior": null,
		# 	  "fixed_params": null,
		# 	  "smart_initialize": false,
		# 	  "fixed_params_initialize": null,
		# 	  "options": null,
		# 	  "smart_init_options": null
		# }

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
