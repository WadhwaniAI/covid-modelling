import copy
import datetime

from main.seir.fitting import single_fitting_cycle
from main.seir.forecast import get_forecast
from main.seir.sensitivity import calculate_sensitivity_and_plot
from viz import plot_forecast, plot_top_k_trials, plot_ptiles
from viz.uncertainty import plot_beta_loss


def fitting(predictions_dict, config):
    predictions_dict['m1'] = single_fitting_cycle(
        **copy.deepcopy(config['fitting']))

    m2_params = copy.deepcopy(config['fitting'])
    m2_params['split']['val_period'] = 0
    predictions_dict['m2'] = single_fitting_cycle(**m2_params)

    predictions_dict['fitting_date'] = datetime.datetime.now().strftime(
        "%Y-%m-%d")


def sensitivity(predictions_dict, config):
    predictions_dict['m1']['plots']['sensitivity'], _, _ = calculate_sensitivity_and_plot(
        predictions_dict, config, which_fit='m1')
    predictions_dict['m2']['plots']['sensitivity'], _, _ = calculate_sensitivity_and_plot(
        predictions_dict, config, which_fit='m2')


def fit_beta(predictions_dict, config):
    uncertainty_args = {'predictions_dict': predictions_dict,
                        'fitting_config': config['fitting'],
                        'forecast_config': config['forecast'],
                        **config['uncertainty']['uncertainty_params']}

    uncertainty = config['uncertainty']['method'](**uncertainty_args)
    return uncertainty


def process_beta_fitting(predictions_dict, uncertainty):
    predictions_dict['m2']['plots']['beta_loss'], _ = plot_beta_loss(
        uncertainty.dict_of_trials)

    predictions_dict['m2']['forecasts']['ensemble_mean'] = uncertainty.ensemble_mean_forecast

    predictions_dict['m2']['beta'] = uncertainty.beta
    predictions_dict['m2']['beta_loss'] = uncertainty.beta_loss


def process_uncertainty_fitting(predictions_dict, uncertainty):
    uncertainty_forecasts = uncertainty.get_forecasts()
    for key in uncertainty_forecasts.keys():
        predictions_dict['m2']['forecasts'][key] = uncertainty_forecasts[key]['df_prediction']
    predictions_dict['m2']['deciles'] = uncertainty_forecasts


def forecast_best(predictions_dict, config):
    predictions_dict['m1']['forecasts'] = {}
    predictions_dict['m2']['forecasts'] = {}
    for fit in ['m1', 'm2']:
        predictions_dict[fit]['forecasts']['best'] = get_forecast(
            predictions_dict, train_fit=fit,
            model=config['fitting']['model'],
            train_end_date=config['fitting']['split']['end_date'],
            forecast_days=config['forecast']['forecast_days']
        )

        predictions_dict[fit]['plots']['forecast_best'] = plot_forecast(
            predictions_dict,
            config['fitting']['data']['dataloading_params']['location_description'],
            which_fit=fit, error_bars=False,
            which_compartments=config['fitting']['loss']['loss_compartments']
        )


def plot_forecasts_top_k_trials(predictions_dict, config):
    kforecasts = plot_top_k_trials(
        predictions_dict, train_fit='m2',
        k=config['forecast']['num_trials_to_plot'],
        which_compartments=config['forecast']['plot_topk_trials_for_columns']
    )

    predictions_dict['m2']['plots']['forecasts_topk'] = {}
    for column in config['forecast']['plot_topk_trials_for_columns']:
        predictions_dict['m2']['plots']['forecasts_topk'][column.name] = kforecasts[column]


def plot_ensemble_forecasts(predictions_dict, config):
    predictions_dict['m2']['plots']['forecast_ensemble_mean'] = plot_forecast(
        predictions_dict,
        config['fitting']['data']['dataloading_params']['location_description'],
        which_compartments=config['fitting']['loss']['loss_compartments'],
        fits_to_plot=['ensemble_mean'], error_bars=False
    )


def plot_forecasts_ptiles(predictions_dict, config):
    ptiles_plots = plot_ptiles(predictions_dict,
                               which_compartments=config['forecast']['plot_ptiles_for_columns'])
    predictions_dict['m2']['plots']['forecasts_ptiles'] = {}
    for column in config['forecast']['plot_ptiles_for_columns']:
        predictions_dict['m2']['plots']['forecasts_ptiles'][column.name] = ptiles_plots[column]
