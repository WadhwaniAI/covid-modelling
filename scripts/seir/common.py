import copy
import datetime
import sys

sys.path.append('../..')

from main.seir.main import single_fitting_cycle
from main.seir.forecast import get_forecast
from viz import plot_forecast, plot_top_k_trials, plot_ptiles
from viz.uncertainty import plot_beta_loss


def fitting(config):
    predictions_dict = single_fitting_cycle(
        **copy.deepcopy(config['fitting']))

    return predictions_dict

def fit_beta(predictions_dict, config):
    uncertainty_args = {'predictions_dict': predictions_dict,
                        'fitting_config': config['fitting'],
                        'forecast_config': config['forecast'],
                        **config['uncertainty']['uncertainty_params']}

    uncertainty = config['uncertainty']['method'](**uncertainty_args)
    return uncertainty


def process_uncertainty_fitting(predictions_dict, config, uncertainty):
    predictions_dict['plots']['beta_loss'], _ = plot_beta_loss(
        uncertainty.dict_of_trials)
    uncertainty_forecasts = uncertainty.get_forecasts()
    for key in uncertainty_forecasts.keys():
        predictions_dict['forecasts'][key] = uncertainty_forecasts[key]['df_prediction']

    predictions_dict['forecasts']['ensemble_mean'] = uncertainty.ensemble_mean_forecast

    predictions_dict['beta'] = uncertainty.beta
    predictions_dict['beta_loss'] = uncertainty.beta_loss
    predictions_dict['deciles'] = uncertainty_forecasts


def process_ensemble(predictions_dict, uncertainty):
    predictions_dict['plots']['beta_loss'], _ = plot_beta_loss(
        uncertainty.dict_of_trials)

    predictions_dict['forecasts']['ensemble_mean'] = uncertainty.ensemble_mean_forecast

    predictions_dict['beta'] = uncertainty.beta
    predictions_dict['beta_loss'] = uncertainty.beta_loss


def process_uncertainty_forecasts(predictions_dict, uncertainty):
    uncertainty_forecasts = uncertainty.get_forecasts()
    for key in uncertainty_forecasts.keys():
        predictions_dict['forecasts'][key] = uncertainty_forecasts[key]['df_prediction']
    predictions_dict['deciles'] = uncertainty_forecasts


def forecast_best(predictions_dict, config):
    predictions_dict['forecasts'] = {}
    predictions_dict['forecasts'] = {}
    predictions_dict['forecasts']['best'] = predictions_dict['trials']['predictions'][0]

    predictions_dict['plots']['forecast_best'] = plot_forecast(
        predictions_dict,
        config['fitting']['data']['dataloading_params']['location_description'],
        error_bars=False,
        which_compartments=config['fitting']['loss']['loss_compartments']
    )


def plot_forecasts_top_k_trials(predictions_dict, config):
    kforecasts = plot_top_k_trials(
        predictions_dict, k=config['forecast']['num_trials_to_plot'],
        which_compartments=config['forecast']['plot_topk_trials_for_columns']
    )

    predictions_dict['plots']['forecasts_topk'] = {}
    for column in config['forecast']['plot_topk_trials_for_columns']:
        predictions_dict['plots']['forecasts_topk'][column.name] = kforecasts[column]


def plot_ensemble_forecasts(predictions_dict, config):
    predictions_dict['plots']['forecast_ensemble_mean'] = plot_forecast(
        predictions_dict,
        config['fitting']['data']['dataloading_params']['location_description'],
        which_compartments=config['fitting']['loss']['loss_compartments'],
        fits_to_plot=['ensemble_mean'], error_bars=False
    )


def plot_forecasts_ptiles(predictions_dict, config):
    ptiles_plots = plot_ptiles(predictions_dict,
                               which_compartments=config['forecast']['plot_ptiles_for_columns'])
    predictions_dict['plots']['forecasts_ptiles'] = {}
    for column in config['forecast']['plot_ptiles_for_columns']:
        predictions_dict['plots']['forecasts_ptiles'][column.name] = ptiles_plots[column]
