import copy
import json
import os
import pickle
from pprint import pformat

import numpy as np
import pandas as pd
import pypandoc
import yaml
from mdutils.mdutils import MdUtils

from utils.fitting.util import CustomEncoder
from utils.generic.config import make_date_str


def create_output(predictions_dict, output_folder, tag):
    """Custom output generation function"""
    directory = f'{output_folder}/{tag}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    d = {}
    for inner in ['variable_param_ranges', 'best_params', 'beta_loss']:
        if inner in predictions_dict:
            with open(f'{directory}/{inner}.json', 'w') as f:
                json.dump(predictions_dict[inner], f, indent=4)
    for inner in ['df_prediction', 'df_district', 'df_train', 'df_val', 'df_loss', 'df_district_unsmoothed']:
        if inner in predictions_dict and predictions_dict[inner] is not None:
            predictions_dict[inner].to_csv(f'{directory}/{inner}.csv')
    for inner in ['trials', 'run_params', 'plots', 'smoothing_description', 'default_params']:
        with open(f'{directory}/{inner}.pkl', 'wb') as f:
            pickle.dump(predictions_dict[inner], f)
    if 'ensemble_mean' in predictions_dict['forecasts']:
        predictions_dict['forecasts']['ensemble_mean'].to_csv(
            f'{directory}/ensemble_mean_forecast.csv')
    predictions_dict['trials']['predictions'][0].to_csv(
        f'{directory}/trials_predictions.csv')
    np.save(f'{directory}/trials_params.npy',
            predictions_dict['trials']['params'])
    np.save(f'{directory}/trials_losses.npy',
            predictions_dict['trials']['losses'])
    d[f'data_last_date'] = predictions_dict['data_last_date']
    d['fitting_date'] = predictions_dict['fitting_date']
    np.save(f'{directory}/beta.npy', predictions_dict['beta'])
    with open(f'{directory}/other.json', 'w') as f:
        json.dump(d, f, indent=4)
    with open(f'{directory}/config.json', 'w') as f:
        json.dump(make_date_str(
            predictions_dict['config']), f, indent=4, cls=CustomEncoder)
    with open(f'{directory}/config.yaml', 'w') as f:
        yaml.dump(make_date_str(predictions_dict['config']), f)


def _dump_predictions_dict(predictions_dict, ROOT_DIR):
    filepath = os.path.join(ROOT_DIR, 'predictions_dict.pkl')
    with open(filepath, 'wb+') as dump:
        pickle.dump(predictions_dict, dump)


def _dump_params(predictions_dict, ROOT_DIR):
    filepath = os.path.join(ROOT_DIR, 'params.json')
    with open(filepath, 'w+') as dump:
        run_params = {
            'm1': copy.copy(predictions_dict['run_params']),
        }
        del run_params['model_class']
        del run_params['variable_param_ranges']

        json.dump(run_params, dump, indent=4)

def _save_trials(predictions_dict, ROOT_DIR):
    predictions_dict['all_trials'].to_csv(os.path.join(ROOT_DIR, 'm1-trials.csv'))


def _create_md_file(predictions_dict, config, ROOT_DIR):
    fitting_date = predictions_dict['fitting_date']
    data_last_date = predictions_dict['data_last_date']
    ld = config['fitting']['data']['dataloading_params']['location_description']
    filename = os.path.join(ROOT_DIR, f'{ld}_report_{fitting_date}')
    mdFile = MdUtils(file_name=filename, title=f'{ld} Fit [Based on data until {data_last_date}]')
    return mdFile, filename


def _log_hyperparams(mdFile, predictions_dict):
    mdFile.new_paragraph("---")
    mdFile.new_paragraph(f"Data available till: {predictions_dict['data_last_date']}")
    mdFile.new_paragraph(f"Fitting Date: {predictions_dict['fitting_date']}")


def _log_plots_util(mdFile, ROOT_DIR, plot_filename, figure, fig_text):
    os.makedirs(os.path.join(os.path.abspath(ROOT_DIR), 'plots'), exist_ok=True)
    plot_filepath = os.path.join(os.path.abspath(ROOT_DIR), 'plots', plot_filename)
    figure.savefig(plot_filepath)
    mdFile.new_line(mdFile.new_inline_image(text=fig_text, path=plot_filepath))
    mdFile.new_paragraph("")


def _log_smoothing(mdFile, ROOT_DIR, fit_dict):
    mdFile.new_header(level=1, title=f'SMOOTHING')
    _log_plots_util(mdFile, ROOT_DIR, 'smoothing.png',
                    fit_dict['plots']['smoothing'], 'Smoothing Plot')
    for sentence in fit_dict['smoothing_description'].split('\n'):
        mdFile.new_paragraph(sentence)
    mdFile.new_paragraph("")


def _log_fits(mdFile, ROOT_DIR, fit_dict):
    mdFile.new_header(level=1, title=f'FIT')
    mdFile.new_header(level=2, title=f'Optimal Parameters')
    mdFile.insert_code(pformat(fit_dict['trials']['params'][0]))
    mdFile.new_header(level=2, title=f'MAPE Loss Values')
    mdFile.new_paragraph(fit_dict['df_loss'].to_markdown())

    mdFile.new_header(level=2, title=f'Fit Curves')
    _log_plots_util(mdFile, ROOT_DIR, f'fit.png',
                    fit_dict['plots']['fit'], f'Fit Curve')


def _log_uncertainty_fit(mdFile, fit_dict):
    mdFile.new_paragraph(f"beta - {fit_dict['beta']}")
    mdFile.new_paragraph(f"beta loss")
    mdFile.insert_code(pformat(fit_dict['beta_loss']))


def _log_forecasts(mdFile, ROOT_DIR, fit_dict):
    _log_plots_util(mdFile, ROOT_DIR, 'forecast-best.png',
                    fit_dict['plots']['forecast_best'], 'Forecast using M2 Best Params')
    _log_plots_util(mdFile, ROOT_DIR, 'forecast-best-50.png',
                    fit_dict['plots']['forecast_best_50'], 'Forecast using best fit, 50th decile params')
    _log_plots_util(mdFile, ROOT_DIR, 'forecast-best-80.png',
                    fit_dict['plots']['forecast_best_80'], 'Forecast using best fit, 80th decile params')
    _log_plots_util(mdFile, ROOT_DIR, 'forecast-ensemble-mean-50.png',
                    fit_dict['plots']['forecast_ensemble_mean_50'], 'Forecast of ensemble fit, 50th decile')

    for column, figure in fit_dict['plots']['forecasts_topk'].items():
        _log_plots_util(mdFile, ROOT_DIR, f'forecast-topk-{column}.png',
                        figure, f'Forecast of top k trials for column {column}')

    for column, figure in fit_dict['plots']['forecasts_ptiles'].items():
        _log_plots_util(mdFile, ROOT_DIR, f'forecast-ptiles-{column}.png',
                        figure, f'Forecast of all ptiles for column {column}')

    if 'scenarios' in fit_dict['plots'].keys():
        mdFile.new_header(level=1, title="What if Scenarios")
        for column, figure in fit_dict['plots']['scenarios'].items():
            _log_plots_util(mdFile, ROOT_DIR, f'forecast-scenarios-{column}.png',
                            figure, '')

    mdFile.new_paragraph("---")


def _log_tables(mdFile, fit_dict):
    trials_processed = copy.deepcopy(fit_dict['trials'])
    trials_processed['losses'] = np.around(trials_processed['losses'], 2)
    trials_processed['params'] = [{key: np.around(value, 2) for key, value in params_dict.items()}
                                  for params_dict in trials_processed['params']]
    mdFile.new_header(level=2, title="Top 10 Trials")
    df = pd.DataFrame.from_dict({(i+1, trials_processed['losses'][i]): trials_processed['params'][i] 
                                 for i in range(10)})
    tbl = df.to_markdown()
    mdFile.new_paragraph(tbl)

    deciles = copy.deepcopy(fit_dict['deciles'])
    for key in deciles.keys():
        deciles[key]['params'] = {param: np.around(value, 2)
                                for param, value in deciles[key]['params'].items()}

    for key in deciles.keys():
        deciles[key]['df_loss'] = deciles[key]['df_loss'].astype(float).round(2)

    mdFile.new_header(level=2, title="Decile Params")
    df = pd.DataFrame.from_dict({np.around(key, 1) : deciles[key]['params'] for key in deciles.keys()})
    tbl = df.to_markdown()
    mdFile.new_paragraph(tbl)

    mdFile.new_header(level=2, title="Decile Loss")
    df = pd.DataFrame.from_dict({np.around(key, 1) : deciles[key]['df_loss'].to_dict()['train'] 
                                 for key in deciles.keys()})
    tbl = df.to_markdown()
    mdFile.new_paragraph(tbl)


def save_dict_and_create_report(predictions_dict, config, ROOT_DIR='../../misc/reports/', 
                                config_filename='default.yaml', config_ROOT_DIR='../../configs/seir'):
    """Creates report (BOTH MD and DOCX) for an input of a dict of predictions for a particular district/region
    The DOCX file can directly be uploaded to Google Drive and shared with the people who have to review

    Arguments:
        predictions_dict {dict} -- Dict of predictions for a particual district/region [NOT ALL Districts]

    Keyword Arguments:
        ROOT_DIR {str} -- the path where the plots and the report would be saved (default: {'../../misc/reports/'})
    """

    if not os.path.exists(ROOT_DIR):
        os.makedirs(ROOT_DIR)

    _dump_predictions_dict(predictions_dict, ROOT_DIR)

    os.system(f'cp {config_ROOT_DIR}/{config_filename} {ROOT_DIR}/{config_filename}')
    
    mdFile, filename = _create_md_file(predictions_dict, config, ROOT_DIR)
    _log_hyperparams(mdFile, predictions_dict)

    with open(f'{config_ROOT_DIR}/{config_filename}') as configfile:
        config = yaml.load(configfile, Loader=yaml.SafeLoader)

    if 'smoothing' in predictions_dict and predictions_dict['plots']['smoothing'] is not None:
        _log_smoothing(mdFile, ROOT_DIR, predictions_dict)

    mdFile.new_header(level=1, title=f'FIT')
    _log_fits(mdFile, ROOT_DIR, predictions_dict)

    mdFile.new_header(level=2, title=f'Uncertainty Fitting')
    _log_uncertainty_fit(mdFile, predictions_dict)

    mdFile.new_header(level=1, title=f'FORECASTS')
    _log_forecasts(mdFile, ROOT_DIR, predictions_dict)
    
    mdFile.new_header(level=1, title="Tables")
    _log_tables(mdFile, predictions_dict)

    # Create a table of contents
    mdFile.new_table_of_contents(table_title='Contents', depth=2)
    mdFile.create_md_file()

    pypandoc.convert_file("{}.md".format(filename), 'docx', outputfile="{}.docx".format(filename))
    # TODO: pdf conversion has some issues with order of images, low priority
