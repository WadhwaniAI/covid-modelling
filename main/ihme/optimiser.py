from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from utils.loss import evaluate
from copy import copy
from models.ihme.new_model import IHME
import numpy as np
import pandas as pd
from datetime import timedelta

# from pathos.multiprocessing import ProcessingPool as Pool
# from pathos.multiprocessing import ThreadPool

import sys
sys.path.append('../..')
from utils.util import HidePrints, train_test_split

class Optimiser():
    def __init__(self, model: IHME, data: pd.DataFrame, args):
        self.model = model
        self.data = data
        self.args = args
    def optimisestar(self, _):
        return self.optimise(**self.args)
    def optimise(self, bounds: list, 
            iterations: int, scoring='mape', val_size=7, min_days=7):
        if len(self.data) - val_size < min_days:
            raise Exception(f'len(data) - val_size must be >= {min_days}')
        model = self.model.generate()
        data = copy(self.data)
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
        # n_days_range = np.arange(min_days, 1 + len(data) - val_size, dtype=int)
        
        # force only min_days for n
        n_days_range = np.arange(min_days, 1 + min_days, dtype=int)
        
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