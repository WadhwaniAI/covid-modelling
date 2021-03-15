"""
forecast.py
"""
import numpy as np


def get_forecast():
    """

    Returns:

    """
    pass


def get_uncertainty_draws(model, model_params):
    """

    Args:
        model ():
        model_params ():

    Returns:

    """
    draws = None
    if model_params['pipeline_args']['n_draws'] > 0:
        draws_dict = model.calc_draws()
        for k in draws_dict.keys():
            low = draws_dict[k]['lower']
            up = draws_dict[k]['upper']
            # TODO: group handling
            draws = np.vstack((low, up))
    return draws
