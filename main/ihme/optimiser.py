import sys
from pathos.multiprocessing import ProcessingPool as Pool
import multiprocessing

import numpy as np
import pandas as pd
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from models.ihme.model import IHME
from utils.fitting.loss import Loss_Calculator

sys.path.append('../..')

from utils.fitting.util import HidePrints


class Optimiser():

    def __init__(self, model: IHME, train: pd.DataFrame, val: pd.DataFrame, args=None):
        """
        Initalises the optimiser for finding best fe_init for IHME model

        Args:
            model (IHME): untrained model
            train (pd.DataFrame): train data
            val (pd.DataFrame): val data
            args ([type], optional): args for self.optimisestar(). Defaults to None.
        """
        self.model = model
        self.train = train
        self.val = val
        self.args = args

    def _optimise(self, bounds: list, iterations: int, scoring='mape', **kwargs):
        """
        optimise function to find best fe_init and n_days_train

        Args:
            bounds (list): fe_bounds; searchspace for fe_init
            iterations (int): number of evals to search for optimum
            scoring (str, optional): mape, rmse, rmsle. Defaults to 'mape'.

        Raises:
            Exception: raised if data doesn't have min_days + val_size rows

        Returns:
            tuple: fe_init, min_loss, trials object
        """
        model = self.model.generate()

        def objective(params):
            test_model = model.generate()
            test_model.priors.update({
                'fe_init': [params['alpha'], params['beta'], params['p']],
            })
            train_cut = self.train[:]
            val_cut = self.val[:]
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

        trials = Trials()
        best = fmin(objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=iterations,
                    trials=trials)

        fe_init = []
        for i, param in enumerate(model.param_names):
            fe_init.append(best[param])

        min_loss = min(trials.losses())
        return fe_init, min_loss, trials

    def optimise_helper(self, _):
        """
        wrapper function for self.optimise

        Args:
            _ ([type]): anything, scrapped

        Returns:
            tuple: fe_init, min_loss, trials object
        """
        return self._optimise(**self.args)

    def optimise(self):
        """

        Returns:

        """
        hyperopt_runs = {}
        trials_dict = {}
        num_processes = multiprocessing.cpu_count() if self.args['num_trials'] > 1 else 1
        pool = Pool(processes=num_processes)
        for i, (best_init, err, trials) in enumerate(pool.map(self.optimise_helper,
                                                              list(range(self.args['num_trials'])))):
            hyperopt_runs[err] = best_init
            trials_dict[i] = trials
        pool.close()
        best_index = np.argmin(hyperopt_runs.keys())
        best_err = min(hyperopt_runs.keys())
        return hyperopt_runs[best_err], trials_dict[best_index]
