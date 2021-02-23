import sys

from copy import copy
from datetime import timedelta

import numpy as np
import pandas as pd
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from models.ihme.model import IHME
from utils.fitting.loss import Loss_Calculator

sys.path.append('../..')
from utils.fitting.util import HidePrints, train_test_split


class Optimiser():
    def __init__(self, model: IHME, data: pd.DataFrame, args=None):
        """
        Initalises the optimiser for finding best fe_init for IHME model

        Args:
            model (IHME): untrained model
            data (pd.DataFrame): fit + val data
            args ([type], optional): args for self.optimisestar(). Defaults to None.
        """
        self.model = model
        self.data = data
        self.args = args

    def optimisestar(self, _):
        """
        wrapper function for self.optimise

        Args:
            _ ([type]): anything, scrapped

        Returns:
            tuple: fe_init, min_loss, trials object
        """
        return self.optimise(**self.args)

    def optimise(self, bounds: list,
                 iterations: int, scoring='mape', val_size=7, min_days=7):
        """
        optimise function to find best fe_init and n_days_train

        Args:
            bounds (list): fe_bounds; searchspace for fe_init
            iterations (int): number of evals to search for optimum
            scoring (str, optional): mape, rmse, rmsle. Defaults to 'mape'.
            val_size (int, optional): to withold as val set. Defaults to 7.
            min_days (int, optional): min_days to train on. Defaults to 7.

        Raises:
            Exception: raised if data doesn't have min_days + val_size rows

        Returns:
            tuple: fe_init, min_loss, trials object
        """
        # if len(self.data) - val_size < min_days:
        #     raise Exception(f'len(data) - val_size must be >= {min_days}')
        model = self.model.generate()
        data = copy(self.data)
        threshold = data[model.date].max() - timedelta(days=val_size)
        train, val = train_test_split(data, threshold, threshold_col=model.date)

        def objective(params):
            test_model = model.generate()
            test_model.priors.update({
                'fe_init': [params['alpha'], params['beta'], params['p']],
            })
            train_cut = train[:]
            val_cut = val[:]
            train_cut.loc[:, 'day'] = (train_cut['date'] - np.min(train_cut['date'])).apply(lambda x: x.days)
            val_cut.loc[:, 'day'] = (val_cut['date'] - np.min(train_cut['date'])).apply(lambda x: x.days)

            with HidePrints():
                test_model.fit(train_cut)
                predictions = test_model.predict(val_cut[model.date].min(), val_cut[model.date].max())
            lc = Loss_Calculator()
            err = lc.evaluate(val_cut[model.ycol], predictions)
            return {
                'loss': err[scoring],
                'status': STATUS_OK,
                'error': err,
                'predictions': predictions,
            }

        space = {}
        for i, bound in enumerate(bounds):
            space[model.param_names[i]] = hp.uniform(model.param_names[i], bound[0], bound[1])
        # fmin returns index for hp.choice
        # n_days_range = np.arange(min_days, 1 + len(data) - val_size, dtype=int)

        # TODO: decide, and remove n from hyperopt search if it remains consistent
        # force only min_days for n
        # n_days_range = np.arange(min_days, 1 + min_days, dtype=int)
        # space['n'] = hp.choice('n', n_days_range)

        trials = Trials()
        best = fmin(objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=iterations,
                    trials=trials)

        fe_init = []
        for i, param in enumerate(model.param_names):
            fe_init.append(best[param])

        # returns index of range provided to hp.choice for n
        # best['n'] = n_days_range[best['n']]

        min_loss = min(trials.losses())
        # print (best, min_loss)
        return fe_init, min_loss, trials
