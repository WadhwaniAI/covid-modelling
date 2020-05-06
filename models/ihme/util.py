import sys
import json
import numpy as np
import pandas as pd

from datetime import timedelta
from copy import deepcopy
import curvefit
from . import data as dataloader

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))

def smooth(y, smoothing_window):
    box = np.ones(smoothing_window)/smoothing_window
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def get_daily_vals(df, col):
        return df[col] - df[col].shift(1)

class Params():
    def __init__(self, df, pargs):
        self.pargs = deepcopy(pargs)
        
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

        # # convert str to functions
        # for (k,v) in self.ycols.items():
        #     self.ycols[k] = getattr(curvefit.core.functions, v)

        self.df = df
        self.multigroup = len(self.df[self.groupcol].unique()) > 1
        self.agg_df = self.df.groupby(self.date).sum().reset_index(col_fill=self.date)
        self.threshold = self.df[self.date].max() - timedelta(days=self.test_size)

    @classmethod
    def fromjson(cls, label):
        with open('params.json', "r") as paramsfile:
            params = json.load(paramsfile)
            if label not in params:
                print("entry not found in params.json")
                sys.exit(0)
        pargs = params['default']
        pargs.update(params[label])
        
        # load data
        data_func = getattr(dataloader, pargs['data_func'])
        if 'data_func_args' in pargs:
            df = data_func(pargs['data_func_args'])
        else:
            df = data_func()
        
        return cls(df, pargs)

    @classmethod
    def fromdefault(cls, df, custom, default_label='default'):
        with open('params.json', "r") as paramsfile:
            params = json.load(paramsfile)
        pargs = params['default']
        pargs.update(params[default_label])
        pargs.update(custom)
        return cls(df, pargs)

    # def train_test(self, df):
    #     data, test = df[df[self.date] < self.threshold], df[df[self.date] >= self.threshold]
    #     return data, test

    # def smoothing_init(self):
    #     new_ycols = {}
    #     for ycol in self.ycols.keys():
    #         self.df[f'{ycol}_smooth'] = smooth(self.df[ycol], 5)
    #         self.agg_df[f'{ycol}_smooth'] = smooth(self.agg_df[ycol], 5)
    #         new_ycols[f'{ycol}_smooth'] = self.ycols[ycol]
    #     orig_ycols = self.ycols
    #     self.ycols = new_ycols 
    #     return orig_ycols

    # def daily_init(self):
    #     self.df.sort_values(self.groupcol)
    #     dailycol = "daily_{ycol}"
    #     for ycol in self.ycols.keys():
    #         self.agg_df[dailycol.format(ycol=ycol)] = self.agg_df[ycol] - self.agg_df[ycol].shift(1)
    #         dailycol_dfs = [self.df[self.df[self.groupcol] == grp][ycol] - self.df[self.df[self.groupcol] == grp][ycol].shift(1) for grp in self.df[self.groupcol].unique()]
    #         self.df[dailycol.format(ycol=ycol)] = pd.concat(dailycol_dfs)
    #     return dailycol
