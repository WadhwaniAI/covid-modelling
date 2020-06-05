from utils.loss import evaluate
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
from utils.util import HidePrints, train_test_split
from viz import setup_plt
from main.ihme.optimiser import Optimiser

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

    def plot_results(self, file_prefix, results=None, transform_y=None, dtp=None, axis_name=None, savepath=None):
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
                val_preds = transform_y(pred_dict['val_preds'], dtp)
                fit_preds = transform_y(pred_dict['fit_preds'], dtp)
                # fit_preds = transform_y(pred_dict['fit_preds'][-14:], dtp)
            else:
                val_preds = pred_dict['val_preds']
                fit_preds = pred_dict['fit_preds']#[-14:]
            val_dates = pred_dict['val_dates']
            fit_dates = pred_dict['fit_dates']#[-14:]
            
            color = cmap(i/len(results.keys()))
            plt.plot(val_dates, val_preds, ls='dashed', c=color,
                label=f'run day: {run_day}')
            plt.plot(fit_dates, fit_preds, ls='solid', c=color,
                label=f'run day: {run_day}')
            plt.errorbar(val_dates, val_preds,
                yerr=val_preds*results[run_day][errkey]['mape']/100, lw=0.5,
                color='lightcoral', barsabove='False', label='MAPE')
            plt.errorbar(fit_dates, fit_preds,
                yerr=fit_preds*results[run_day][errkey]['mape']/100, lw=0.5,
                color='lightcoral', barsabove='False', label='MAPE')

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
        errs = [results[run_day][errkey][scoring] for run_day in results.keys()]
        plt.plot(dates, errs, ls='-', c='crimson',
            label=scoring)
        plt.legend()
        if savepath is not None:
            plt.savefig(savepath)
            plt.clf()
        return

def run_model_unpack(kwargs):
    return run_model(**kwargs)

def run_model(model, run_day, fit_data, val_data, max_evals, hyperopt_val_size,min_days, xform_func, dtp, scoring):
    print ("\rbacktesting for", run_day, end="")
    incremental_model = model.generate()
    
    # # OPTIMIZE HYPERPARAMS
    kwargs = {
        'bounds': copy(incremental_model.priors['fe_bounds']), 
        'iterations': max_evals,
        'scoring': scoring, 
        'val_size': hyperopt_val_size,
        'min_days': min_days,
    }
    o = Optimiser(incremental_model, fit_data, kwargs)
    ((best_init, n_days), err, trials) = o.optimisestar(0)
    
    fit_data = fit_data[-n_days:]
    fit_data.loc[:, 'day'] = (fit_data['date'] - np.min(fit_data['date'])).apply(lambda x: x.days)
    val_data.loc[:, 'day'] = (val_data['date'] - np.min(fit_data['date'])).apply(lambda x: x.days)
    incremental_model.priors['fe_init'] = best_init
    
    # FIT/PREDICT
    incremental_model.fit(fit_data)
    predictions = incremental_model.predict(fit_data[model.date].min(),
        val_data[model.date].max())
    # print (predictions)
    err = evaluate(val_data[model.ycol], predictions[len(fit_data):])
    xform_err = None
    if xform_func is not None:
        xform_err = evaluate(xform_func(val_data[model.ycol], dtp),
            xform_func(predictions[len(fit_data):], dtp))
    result_dict = {
        'fe_init': best_init,
        'n_days': n_days,
        'error': err,
        'xform_error': xform_err,
        'predictions': {
            'start': fit_data[model.date].min(),
            'fit_dates': fit_data[model.date],
            'val_dates': val_data[model.date],
            'fit_preds': predictions[:len(fit_data)],
            'val_preds': predictions[len(fit_data):],
        },
        'trials': trials,
    }
    return run_day, result_dict