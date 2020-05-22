from utils.loss import evaluate
from copy import copy
from models.ihme.new_model import IHME
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from copy import copy
import multiprocessing
import time

import dill
from pathos.multiprocessing import ProcessingPool as Pool

import sys
sys.path.append('../..')
from utils.util import HidePrints, train_test_split

def backtesting(model: IHME, data, start, end, increment=5, future_days=10, 
        hyperopt_val_size=7, optimize_runs=3, max_evals=100, xform_func=None,
        dtp=None, min_days=14):
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
            hyperopt_runs = {}
            trials_dict = {}
            pool = Pool(processes=5)
            o = Optimize((incremental_model, fit_data,
                    incremental_model.priors['fe_bounds'], max_evals, 'mape', 
                    hyperopt_val_size, min_days))
            for i, ((best_init, n_days), err, trials) in enumerate(pool.map(o.optimizestar, list(range(optimize_runs)))):
                hyperopt_runs[err] = (best_init, n_days)
                trials_dict[i] = trials
            best_init, n_days = hyperopt_runs[min(hyperopt_runs.keys())]
            
            fit_data = fit_data[-n_days:]
            fit_data.loc[:, 'day'] = (fit_data['date'] - np.min(fit_data['date'])).apply(lambda x: x.days)
            val_data.loc[:, 'day'] = (val_data['date'] - np.min(fit_data['date'])).apply(lambda x: x.days)
            incremental_model.priors['fe_init'] = best_init
        else:
            n_days, best_init = len(fit_data), incremental_model.priors['fe_init']
            trials_dict = None
        
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
        'dtp': dtp,
        'future_days': future_days,
        'runtime': runtime,
        'model': model,
        'trials': trials_dict,
    }
    return out

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
        
        with HidePrints():
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
    
    # force only min_days for n
    # n_days_range = np.arange(min_days, 1 + min_days, dtype=int)
    
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
    return (fe_init, best['n']), min_loss, trials

# to make pool.map work
class Optimize():
    def __init__(self, args):
        self.args = args
    def optimizestar(self, _):
        return optimize(*self.args)
