import sys
import time
from copy import copy
from datetime import datetime, timedelta

import pandas as pd
from pathos.multiprocessing import ProcessingPool as Pool

from models.ihme.model import IHME

sys.path.append('../..')
from utils.enums import Columns
from main.ihme.fitting import run_cycle_compartments


class IHMEBacktest:
    def __init__(self, model: IHME, data: pd.DataFrame, district, state):
        """
        initialises a backtesting class for the IHME model

        Args:
            model (IHME): untrained model to base backtesting off of
            data (pd.DataFrame): all historical data for the location
            district (str): district name
            state (str): state name
        """
        self.model = model.generate()
        self.data = copy(data)
        self.district = district
        self.state = state

    def test(self, increment=5, future_days=10,
             hyperopt_val_size=7, max_evals=100, xform_func=None,
             dtp=None, min_days=7, scoring='mape', which_compartments=Columns.curve_fit_compartments()):
        """
        Runs the backtesting at the specified increment frequency, with parellelisation
        Up to 10 threads, per run_day

        Args:
            increment (int, optional): how frequently to rerun the model. Defaults to 5.
            future_days (int, optional): how far to predict. Defaults to 10.
            hyperopt_val_size (int, optional): val size for the model training. Defaults to 7.
            max_evals (int, optional): num evals in hyperparam optimisation. Defaults to 100.
            xform_func ([type], optional): function to transform the data back to # cases. Defaults to None.
            dtp ([type], optional): district total population. Defaults to None.
            min_days (int, optional): train_period minimum. Defaults to 7.
            scoring (str, optional): 'mape', 'rmse', or 'rmsle. Defaults to 'mape'.
            which_compartments ([type], optional): members of Columns to backtest. Defaults to Columns.curve_fit_compartments().

        Returns:
            [dict]: dict['results'] contains the results from each run
        """
        runtime_s = time.time()
        start = self.data[self.model.date].min()
        end = self.data[self.model.date].max()
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
                                 & (self.data[self.model.date] <= start + timedelta(days=run_day + future_days))]
            for arg in ['fit_data', 'val_data', 'run_day', 'max_evals', 'which_compartments',
                        'hyperopt_val_size', 'min_days', 'xform_func', 'dtp', 'scoring']:
                kwargs[arg] = eval(arg)

            args.append(kwargs)
        for run_day, result_dict in pool.map(run_model_unpack, args):
            results[run_day] = result_dict

        runtime = time.time() - runtime_s
        print(runtime)
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


def run_model_unpack(kwargs):
    """
    wrapper function to use in multithreading

    Args:
        kwargs ([type]): args to pass into run_model

    Returns:
        dict: result_dict
    """
    return run_model(**kwargs)


def run_model(model, run_day, fit_data, val_data, max_evals, hyperopt_val_size, min_days, xform_func, dtp, scoring,
              which_compartments):
    """
    wrapper function to use in multithreading that returns run_day along with results_dict

    Args:
        model (IHME): model to train
        run_day (int): run_day (just gets returned as-is)
        fit_data (pd.DataFrame): passed to run_cycle
        val_data (pd.DataFrame): passed to run_cycle
        max_evals (int): passed to run_cycle
        hyperopt_val_size (int): passed to run_cycle
        min_days (int): passed to run_cycle
        xform_func (func): passed to run_cycle
        dtp (int): passed to run_cycle
        scoring (str): passed to run_cycle
        which_compartments (list): passed to run_cycle

    Returns:
        dict: result_dict
    """
    print("\rbacktesting for", run_day, end="")
    df = pd.concat([fit_data, val_data], axis=1)
    dataframes = {
        'train': fit_data,
        'test': val_data,
        'df': df,
        'df_nora': pd.DataFrame(columns=df.columns, index=df.index),
        'train_nora': pd.DataFrame(columns=fit_data.columns, index=fit_data.index),
        'test_nora': pd.DataFrame(columns=val_data.columns, index=val_data.index)
    }
    result_dict = run_cycle_compartments(dataframes, copy(model.model_parameters),
                                         dtp=dtp, max_evals=max_evals, min_days=min_days, scoring=scoring,
                                         val_size=hyperopt_val_size, xform_func=xform_func, forecast_days=0)
    return run_day, result_dict
