from utils.loss import Loss_Calculator
from copy import copy
from models.ihme.model import IHME
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time

from pathos.multiprocessing import ProcessingPool as Pool

import matplotlib.pyplot as plt
import matplotlib as mpl

import sys
sys.path.append('../..')
from utils.util import HidePrints
from viz import setup_plt
from main.ihme.fitting import run_cycle

class IHMEBacktest:
    def __init__(self, model: IHME, data: pd.DataFrame, district, state):
        self.model = model.generate()
        self.data = copy(data)
        self.district = district
        self.state = state

    def test(self, increment=5, future_days=10, 
        hyperopt_val_size=7, max_evals=100, xform_func=None,
        dtp=None, min_days=7, scoring='mape'):
        runtime_s = time.time()
        start = self.data[self.model.date].min()
        end =  self.data[self.model.date].max()
        n_days = (end - start).days + 1 - future_days
        results = {}
        seed = datetime.today().timestamp()
        pool = Pool(processes=10)
        
        args = []
        for run_day in range(min_days + hyperopt_val_size, n_days, increment):
            kwargs = {
                'model': self.model.generate(),
            }
            fit_data = self.data[(self.data[self.model.date] <= start + timedelta(days=run_day))]
            val_data = self.data[(self.data[self.model.date] > start + timedelta(days=run_day)) \
                & (self.data[self.model.date] <= start + timedelta(days=run_day+future_days))]
            for arg in ['fit_data', 'val_data', 'run_day', 'max_evals', 
                'hyperopt_val_size', 'min_days', 'xform_func', 'dtp', 'scoring']:
                kwargs[arg] = eval(arg)
            
            args.append(kwargs)
        for run_day, result_dict in pool.map(run_model_unpack, args):
            results[run_day] = result_dict
    
        runtime = time.time() - runtime_s
        print (runtime)
        self.results = {
            'results': results,
            'seed': seed,
            'df': self.data,
            'dtp': dtp,
            'future_days': future_days,
            'runtime': runtime,
            'model': self.model,
        }
        return self.results

    def plot_results(self, file_prefix, scoring='mape', results=None, transform_y=None, dtp=None, axis_name=None, savepath=None):
        results = self.results['results'] if results is None else results
        ycol = self.model.ycol
        title = f'{file_prefix} {ycol}' +  ' backtesting'
        # plot predictions against actual
        if axis_name is not None:
            setup_plt(axis_name)
        else:
            setup_plt(ycol)
        plt.yscale("linear")
        plt.title(title.format(self.model.func.__name__))

        if transform_y is not None:
            self.data[self.model.ycol] = transform_y(self.data[self.model.ycol], dtp)
        errkey = 'xform_error' if transform_y is not None else 'error'

        # plot predictions
        cmap = mpl.cm.get_cmap('winter')
        for i, run_day in enumerate(results.keys()):
            pred_dict = results[run_day]['predictions']
            if transform_y is not None:
                preds = transform_y(pred_dict['predictions'], dtp)
            else:
                preds = pred_dict['predictions']
            val_dates = pred_dict['val_dates']
            fit_dates = pred_dict['fit_dates']
            
            color = cmap(i/len(results.keys()))
            plt.plot(val_dates, preds.loc[val_dates, ycol], ls='dashed', c=color,
                label=f'run day: {run_day}')
            plt.plot(fit_dates, preds.loc[fit_dates, ycol], ls='solid', c=color,
                label=f'run day: {run_day}')
            plt.errorbar(val_dates, preds.loc[val_dates, ycol],
                yerr=preds.loc[val_dates, ycol]*(results[run_day][errkey]['test'][scoring]/100), lw=0.5,
                color='lightcoral', barsabove='False', label=scoring)
            plt.errorbar(fit_dates, preds.loc[fit_dates, ycol],
                yerr=preds.loc[fit_dates, ycol]*(results[run_day][errkey]['test'][scoring]/100), lw=0.5,
                color='lightcoral', barsabove='False', label=scoring)

        # plot data we fit on
        plt.scatter(self.data[self.model.date], self.data[ycol], c='crimson', marker='+', label='data')

        # plt.legend()
        if savepath is not None:
            plt.savefig(savepath)
            plt.clf()
        return

    def plot_errors(self, file_prefix, scoring='mape', use_xform=True, axis_name=None, results=None, savepath=None):
        results = self.results['results'] if results is None else results
        start = self.data[self.model.date].min()
        ycol = self.model.ycol
        
        title = f'{file_prefix} {ycol}' +  ' backtesting errors'
        errkey = 'xform_error' if use_xform else 'error'

        setup_plt(scoring)
        plt.yscale("linear")
        plt.title(title)

        # plot error
        dates = [start + timedelta(days=run_day) for run_day in results.keys()]
        errs = [results[run_day][errkey]['test'][scoring] for run_day in results.keys()]
        plt.plot(dates, errs, ls='-', c='crimson',
            label=scoring)
        plt.legend()
        if savepath is not None:
            plt.savefig(savepath)
            plt.clf()
        return

def run_model_unpack(kwargs):
    return run_model(**kwargs)

def run_model(model, run_day, fit_data, val_data, max_evals, hyperopt_val_size, min_days, xform_func, dtp, scoring):
    print ("\rbacktesting for", run_day, end="")
    dataframes = {'train': fit_data, 'test': val_data}
    result_dict = run_cycle(dataframes, copy(model.model_parameters), 
        dtp=dtp, max_evals=max_evals, min_days=min_days, scoring=scoring, 
        val_size=hyperopt_val_size, xform_func=xform_func, predict_days=0)
    return run_day, result_dict
