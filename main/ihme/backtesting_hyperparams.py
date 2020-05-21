from models.ihme.util import evaluate
from copy import copy
from models.ihme.new_model import IHME
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from copy import copy
import multiprocessing
import time

import dill
from pathos.multiprocessing import ProcessingPool as Pool

def run_dill_encoded(payload):
    fun, args = dill.loads(payload)
    return fun(*args)

def apply_async(pool, fun, args):
    payload = dill.dumps((fun, args))
    return pool.apply_async(run_dill_encoded, (payload,))

import os, sys
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def backtesting(model: IHME, data, start, end, increment=5, future_days=10, 
        hyperopt_val_size=7, optimize_runs=3, max_evals=100, xform_func=None, dtp=None, min_days=14):
    runtime_s = time.time()
    n_days = (end - start).days + 1 - future_days
    model = model.generate()
    results = {}
    seed = datetime.today().timestamp()
    for run_day in range(min_days + hyperopt_val_size, n_days, increment):
        print ("\rbacktesting for", run_day, end="")
        incremental_model = model.generate()
        fit_data = data[(data[model.date] <= start + timedelta(days=run_day))]
        val_data = data[(data[model.date] > start + timedelta(days=run_day)) \
            & (data[model.date] <= start + timedelta(days=run_day+future_days))]
        
        # # OPTIMIZE HYPERPARAMS
        if optimize_runs > 0:
            # n_days, best_init = optimize_hyperparameters(incremental_model, fit_data,
            #     incremental_model.priors['fe_bounds'], (0.1, 2, 0.5), iterations=optimize_runs, val_size=5, seed=seed)
            #     # incremental_model.priors['fe_bounds'], (0.1, 2, 0.5), iterations=optimize, val_size=5)
            
            hyperopt_runs = {}
            pool = Pool(processes=4)
            o = Optimize((incremental_model, fit_data,
                    incremental_model.priors['fe_bounds'], max_evals, 'mape', 
                    hyperopt_val_size, min_days))
            for (best_init, n_days), err in pool.map(o.optimizestar, list(range(optimize_runs))):
                hyperopt_runs[err] = (best_init, n_days)
            best_init, n_days = hyperopt_runs[min(hyperopt_runs.keys())]
            
            # results = [apply_async(pool, optimize,
            #     (incremental_model, fit_data,
            #         incremental_model.priors['fe_bounds'], max_evals, 'mape', 
            #         hyperopt_val_size, min_days))
            #     for _ in range(optimize_runs)]
            # for r in results:
            #     out = r.get()
            #     outputs.append(out)
            # print(outputs)
            # for (best_init, n_days), err in outputs:
            #     hyperopt_runs[err] = (best_init, n_days)
            # best_init, n_days = hyperopt_runs[min(hyperopt_runs.keys())]

            # for _ in range(optimize_runs):
            #     (best_init, n_days), err = optimize(incremental_model, fit_data,
            #         incremental_model.priors['fe_bounds'],
            #         iterations=max_evals, val_size=5, min_days=14)
            #     hyperopt_runs[err] = (best_init, n_days)
            # best_init, n_days = hyperopt_runs[min(hyperopt_runs.keys())]
            
            fit_data = fit_data[-n_days:]
            fit_data.loc[:, 'day'] = (fit_data['date'] - np.min(fit_data['date'])).apply(lambda x: x.days)
            val_data.loc[:, 'day'] = (val_data['date'] - np.min(fit_data['date'])).apply(lambda x: x.days)
            incremental_model.priors['fe_init'] = best_init
        else:
            n_days, best_init = len(fit_data), incremental_model.priors['fe_init']
        
        # # PRINT DATES (ENSURE CONTINUITY)
        # print (fit_data[model.date].min(), fit_data[model.date].max())
        # print (val_data[model.date].min(), val_data[model.date].max())
        
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
        results[run_day] = {
            'fe_init': best_init,
            'n_days': n_days,
            'seed': seed,
            'error': err,
            'xform_error': xform_err,
            'predictions': {
                'start': start,
                'fit_dates': fit_data[model.date],
                'val_dates': val_data[model.date],
                'fit_preds': predictions[:len(fit_data)],
                'val_preds': predictions[len(fit_data):],
            }
        }
    runtime = time.time() - runtime_s
    print (runtime)
    out = {
        'results': results,
        'df': data,
        'future_days': future_days,
        'runtime': runtime,
    }
    return out

def train_test_split(df, threshold, threshold_col='date'):
            return df[df[threshold_col] <= threshold], df[df[threshold_col] > threshold]

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
def optimize(model: IHME, data: pd.DataFrame, bounds: list, 
        iterations: int, scoring='mape', val_size=7, min_days=7):
    if len(data) - val_size < min_days:
        raise Exception(f'len(data) - val_size must be >= {min_days}')
    model = model.generate()
    data = copy(data)
    threshold = data[model.date].max() - timedelta(days=val_size)
    train, val = train_test_split(data, threshold, threshold_col=model.date)

    def objective(params):
        test_model = model.generate()
        test_model.priors.update({
            'fe_init': [params['alpha'], params['beta'], params['p']],
        })
        train_cut = train[-params['n']:]
        val_cut = val[:]
        train_cut.loc[:, 'day'] = (train_cut['date'] - np.min(train_cut['date'])).apply(lambda x: x.days)
        val_cut.loc[:, 'day'] = (val_cut['date'] - np.min(train_cut['date'])).apply(lambda x: x.days)
        
        with HiddenPrints():
            test_model.fit(train_cut)
            predictions = test_model.predict(val_cut[model.date].min(), val_cut[model.date].max())
        err = evaluate(val_cut[model.ycol], predictions)
        return {
            'loss': err[scoring],
            'status': STATUS_OK,
            # -- store other results like this
            'error': err,
            'predictions': predictions,
        }

    space = {}
    for i, bound in enumerate(bounds):
        space[model.param_names[i]] = hp.uniform(model.param_names[i], bound[0], bound[1])
    # fmin returns index for hp.choice
    n_days_range = np.arange(min_days, 1 + len(data) - val_size, dtype=int)
    space['n'] = hp.choice('n', n_days_range)

    trials = Trials()
    best = fmin(objective,
        space=space,
        algo=tpe.suggest,
        max_evals=iterations,
        trials=trials)
    
    fe_init = []
    for i, param in enumerate(model.param_names):
        fe_init.append(best[param])
    best['n'] = n_days_range[best['n']]
    
    min_loss = min(trials.losses())
    # print (best, min_loss)
    return (fe_init, best['n']), min_loss

class Optimize():
    def __init__(self, args):
        self.args = args
    def optimizestar(self, _):
        return optimize(*self.args)

def random_search(model: IHME, data: pd.DataFrame, bounds: list,
        steps: list, iterations: int, scoring='mape', seed=None, val_size=7, min_days=7):
    '''
    bounds: [(l1, u1), (l2, u2), (l3, u3)]
    steps: [s1, s2, s3]
    '''
    if len(data) - val_size < min_days:
        raise Exception(f'len(data) - val_size must be >= {min_days}')
    data = copy(data)
    model = model.generate()
    fe_inits = [[float(i),float(j),float(k)] for i in np.arange(bounds[0][0], bounds[0][1], steps[0])
                            for j in np.arange(bounds[1][0], bounds[1][1], steps[1])
                            for k in np.arange(bounds[2][0], bounds[2][1], steps[2])]
    all_inits = [[fe_init, n] for fe_init in fe_inits
                            for n in range(min_days, max([1 + len(data) - val_size, 28]))]
    if seed is None:
        seed = datetime.today().timestamp()
    random.seed(seed)
    indices = random.sample(range(len(all_inits)), iterations)
    
    threshold = data[model.date].max() - timedelta(days=val_size)
    train, val = train_test_split(data, threshold, threshold_col=model.date)

    # set baseline
    model.fit(train)
    predictions = model.predict(val[model.date].min(), val[model.date].max())
    err = evaluate(val[model.ycol], predictions)
    min_err = err[scoring]
    best_init = [model.priors['fe_init'], len(train)]

    for idx in indices:
        test_model = model.generate()
        test_model.priors.update({
            'fe_init': all_inits[idx][0],
            'fe_bounds': bounds
        })
        train_cut = train[-all_inits[idx][1]:]
        val_cut = val[:]
        train_cut.loc[:, 'day'] = (train_cut['date'] - np.min(train_cut['date'])).apply(lambda x: x.days)
        val_cut.loc[:, 'day'] = (val_cut['date'] - np.min(train_cut['date'])).apply(lambda x: x.days)
            
        test_model.fit(train_cut)
        predictions = test_model.predict(val_cut[model.date].min(), val_cut[model.date].max())
        err = evaluate(val_cut[model.ycol], predictions)
        
        if err[scoring] < min_err:
            min_err = err[scoring]
            best_init = all_inits[idx]
    return min_err, best_init

def optimize_hyperparameters(model: IHME, data: pd.DataFrame, bounds: list, 
        steps: list, iterations: int, scoring='mape', seed=None, val_size=7):
    model = model.generate()
    data = copy(data)
    _, (fe_init, n_days) = random_search(model, data, bounds, steps, iterations,
        scoring=scoring, seed=seed, val_size=val_size)
    # model.priors['fe_init'] = fe_init
    # _, n_days = best_train_set(model, data, scoring=scoring, val_size=val_size)
    return n_days, fe_init

import numpy as np
def lograte_to_cumulative(to_transform, population):
    cumulative = np.exp(to_transform) * population
    return cumulative

def rate_to_cumulative(to_transform, population):
    cumulative = to_transform * population
    return cumulative

from models.ihme.util import setup_plt
import matplotlib.pyplot as plt
import matplotlib as mpl
from pandas.plotting import register_matplotlib_converters
from matplotlib.dates import DateFormatter
import pandas as pd

def plot_results(model, train, test, predictions, predictdate, testerr,
        file_prefix, draws=None):
    ycol = model.ycol
    maperr = testerr['mape']
    title = f'{file_prefix} {ycol}' +  ' fit to {}'
    # plot predictions against actual
    setup_plt(ycol)
    plt.yscale("linear")
    plt.title(title.format(model.func.__name__))
    n_data = len(train) + len(test)
    # plot predictions
    plt.plot(predictdate, predictions, ls='-', c='dodgerblue', 
        label='fit: {}: {}'.format(model.func.__name__, model.pipeline.mod.params))
    # plot error bars based on MAPE
    future_x = predictdate[n_data:]
    future_y = predictions[n_data:]
    plt.errorbar(future_x, 
        future_y,
        yerr=future_y*maperr, lw=0.5, color='palegoldenrod', barsabove='False', label='mape')
    if draws is not None:
        # plot error bars based on draws
        plt.errorbar(future_x, 
            future_y, 
            yerr=draws[:,n_data:], lw=0.5, color='lightcoral', barsabove='False', label='draws')
    # plot train test boundary
    plt.axvline(train[model.date].max(), ls=':', c='slategrey', label='train/test boundary')
    # plot data we fit on
    plt.scatter(train[model.date], train[ycol], c='crimson', marker='+', label='data')
    plt.scatter(test[model.date], test[ycol], c='crimson', marker='+')
    if model.smoothing:
        plt.plot(train[model.date], train[model.ycol.split('_smoothed')[0]], c='k', marker='+', label='unsmoothed data')
        plt.plot(test[model.date], test[model.ycol.split('_smoothed')[0]], c='k', marker='+')
    
    plt.legend()
    return

def plot_backtesting_results(model, df, results, future_days, file_prefix, transform_y=None, dtp=None, axis_name=None):
    ycol = model.ycol
    title = f'{file_prefix} {ycol}' +  ' backtesting'
    # plot predictions against actual
    if axis_name is not None:
        setup_plt(axis_name)
    else:
        setup_plt(ycol)
    plt.yscale("linear")
    plt.title(title.format(model.func.__name__))

    if transform_y is not None:
        df[model.ycol] = transform_y(df[model.ycol], dtp)
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
        plt.plot(val_dates, val_preds, ls='-', c=color,
            label=f'run day: {run_day}')
        plt.plot(fit_dates, fit_preds, ls='-', c=color,
            label=f'run day: {run_day}')
        plt.errorbar(val_dates, val_preds,
            yerr=val_preds*results[run_day][errkey]['mape'], lw=0.5,
            color='lightcoral', barsabove='False', label='MAPE')
        plt.errorbar(fit_dates, fit_preds,
            yerr=fit_preds*results[run_day][errkey]['mape'], lw=0.5,
            color='lightcoral', barsabove='False', label='MAPE')

    # plot data we fit on
    plt.scatter(df[model.date], df[ycol], c='crimson', marker='+', label='data')

    # plt.legend()
    return

def plot_backtesting_errors(model, df, start_date, results, file_prefix,
                            scoring='mape', use_xform=True, axis_name=None):
    ycol = model.ycol
    title = f'{file_prefix} {ycol}' +  ' backtesting errors'
    errkey = 'xform_error' if use_xform else 'error'

    setup_plt(scoring)
    plt.yscale("linear")
    plt.title(title)

    # plot error
    dates = [start_date + timedelta(days=run_day) for run_day in results.keys()]
    errs = [results[run_day][errkey][scoring] for run_day in results.keys()]
    plt.plot(dates, errs, ls='-', c='crimson',
        label=scoring)
    plt.legend()
    return