import sys
import json
import pandas as pd

from datetime import timedelta
from copy import deepcopy
import curvefit
from . import dataloader

class Params():
    def __init__(self, df, pargs):
        self.pargs = deepcopy(pargs)
        
        # set vars
        self.date, self.groupcol, self.xcol = pargs['date'], pargs['groupcol'], 'day'
        self.xcol, self.ycols = pargs['xcol'], pargs['ycols']
        self.test_size=pargs['test_size']

        self.pipeline_run_args = pargs['pipeline_args']
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