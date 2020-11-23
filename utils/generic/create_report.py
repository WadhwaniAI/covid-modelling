import os
import datetime
import pypandoc
from mdutils.mdutils import MdUtils
from mdutils import Html
from pprint import pformat
import pandas as pd
import numpy as np
import pickle
import yaml
import copy
import json

def _dump_predictions_dict(predictions_dict, ROOT_DIR):
    filepath = os.path.join(ROOT_DIR, 'predictions_dict.pkl')
    with open(filepath, 'wb+') as dump:
        pickle.dump(predictions_dict, dump)


def _dump_params(m0_dict, m1_dict, m2_dict, ROOT_DIR):
    filepath = os.path.join(ROOT_DIR, 'params.json')
    with open(filepath, 'w+') as dump:
        run_params = {
            'm0': copy.copy(m0_dict['run_params']),
            'm1': copy.copy(m1_dict['run_params']),
            'm2': copy.copy(m2_dict['run_params'])
        }
        del run_params['m0']['model_class']
        del run_params['m0']['variable_param_ranges']
        del run_params['m1']['model_class']
        del run_params['m1']['variable_param_ranges']
        del run_params['m2']['model_class']
        del run_params['m2']['variable_param_ranges']

        json.dump(run_params, dump, indent=4)

def _save_trials(m1_dict, m2_dict, ROOT_DIR):
    m1_dict['all_trials'].to_csv(os.path.join(ROOT_DIR, 'm1-trials.csv'))
    m2_dict['all_trials'].to_csv(os.path.join(ROOT_DIR, 'm2-trials.csv'))


def _create_md_file(predictions_dict, config, ROOT_DIR):
    fitting_date = predictions_dict['fitting_date']
    data_last_date = predictions_dict['m1']['data_last_date']
    state = config['fitting']['data']['dataloading_params']['state']
    dist = config['fitting']['data']['dataloading_params']['district']
    dist = '' if dist is None else dist
    filename = os.path.join(ROOT_DIR, f'{state.title()}-{dist.title()}_report_{fitting_date}')
    mdFile = MdUtils(file_name=filename, title=f'{dist.title()} Fits [Based on data until {data_last_date}]')
    return mdFile, filename

def _log_hyperparams(mdFile, predictions_dict, config):
    mdFile.new_paragraph("---")
    mdFile.new_paragraph(f"Data available till: {predictions_dict['m2']['data_last_date']}")
    mdFile.new_paragraph(f"Fitting Date: {predictions_dict['fitting_date']}")


def _log_plots_util(mdFile, ROOT_DIR, plot_filename, figure, fig_text):
    os.makedirs(os.path.join(os.path.abspath(ROOT_DIR), 'plots'), exist_ok=True)
    plot_filepath = os.path.join(os.path.abspath(ROOT_DIR), 'plots', plot_filename)
    figure.savefig(plot_filepath)
    mdFile.new_line(mdFile.new_inline_image(text=fig_text, path=plot_filepath))
    mdFile.new_paragraph("")

def _log_daily_trends(mdFile, ROOT_DIR, fit_dict):
    mdFile.new_header(level=1, title=f'DAILY TRENDS')
    for trend, trend_plot in fit_dict['plots']['trends'].items():
        _log_plots_util(mdFile, ROOT_DIR, trend + '.png',
                        trend_plot, trend + ' Plot')
    mdFile.new_paragraph("")

def _log_smoothing(mdFile, ROOT_DIR, fit_dict):
    mdFile.new_header(level=1, title=f'SMOOTHING')
    _log_plots_util(mdFile, ROOT_DIR, 'smoothing.png',
                    fit_dict['plots']['smoothing'], 'Smoothing Plot')
    for sentence in fit_dict['smoothing_description'].split('\n'):
        mdFile.new_paragraph(sentence)
    mdFile.new_paragraph("")


def _log_fits(mdFile, ROOT_DIR, fit_dict, which_fit='M1'):

    mdFile.new_header(level=1, title=f'{which_fit} FIT')
    
    if fit_dict.get('df_train') is not None:
        mdFile.new_paragraph(f"TRAIN PERIOD START DATE: {fit_dict['df_train'].date.min().date()}")
        mdFile.new_paragraph(f"TRAIN PERIOD END DATE: {fit_dict['df_train'].date.max().date()}")
        mdFile.new_paragraph(f"TRAIN PERIOD: {(fit_dict['df_train'].date.max().date() - fit_dict['df_train'].date.min().date()).days + 1} DAYS")        
    if fit_dict.get('df_val') is not None:
        mdFile.new_paragraph(f"VAL PERIOD START DATE: {fit_dict['df_val'].date.min().date()}")
        mdFile.new_paragraph(f"VAL PERIOD END DATE: {fit_dict['df_val'].date.max().date()}")
        mdFile.new_paragraph(f"VAL PERIOD: {(fit_dict['df_val'].date.max().date() - fit_dict['df_val'].date.min().date()).days + 1} DAYS")
    if fit_dict.get('df_test') is not None:    
        mdFile.new_paragraph(f"TEST PERIOD START DATE: {fit_dict['df_test'].date.min().date()}")
        mdFile.new_paragraph(f"TEST PERIOD END DATE: {fit_dict['df_test'].date.max().date()}")
        mdFile.new_paragraph(f"TEST PERIOD: {(fit_dict['df_test'].date.max().date() - fit_dict['df_test'].date.min().date()).days + 1} DAYS")

    mdFile.new_paragraph("")
    mdFile.new_header(level=2, title=f'')
    mdFile.new_header(level=2, title=f'Variable parameter space')
    mdFile.insert_code(pformat(fit_dict['run_params']['variable_param_ranges']))
    mdFile.new_header(level=2, title=f'Default parameters')
    mdFile.insert_code(pformat(fit_dict['run_params']['default_params']))
    mdFile.new_header(level=2, title=f'Optimal Parameters')
    mdFile.insert_code(pformat(fit_dict['best_params']))
    mdFile.new_header(level=2, title=f'MAPE Loss Values')

    mdFile.new_paragraph(fit_dict['df_loss'].to_markdown(tablefmt="simple"))

    if which_fit.lower() == 'm1' or which_fit.lower() == 'm2':
        mdFile.new_header(level=2, title=f'{which_fit} Fit Curves')
        _log_plots_util(mdFile, ROOT_DIR, f'{which_fit.lower()}-fit.png',
                        fit_dict['plots']['fit'], f'{which_fit} Fit Curve')
    else:
        mdFile.new_header(level=2, title=f'{which_fit} Fit Curves')
        _log_plots_util(mdFile, ROOT_DIR, f'{which_fit.lower()}-ensemble_mean.png',
                        fit_dict['plots']['forecast_ensemble_mean_best'], f'{which_fit} Ensemble Mean Curve')
     


    mdFile.new_header(level=2, title=f'{which_fit} Sensitivity Curves')
    _log_plots_util(mdFile, ROOT_DIR, f'{which_fit.lower()}-sensitivity.png',
                    fit_dict['plots']['sensitivity'], f'{which_fit} Fit Sensitivity')

def _log_uncertainty_fit(mdFile, fit_dict):
    mdFile.new_paragraph(f"beta - {fit_dict['beta']}")
    mdFile.new_paragraph(f"beta loss")
    mdFile.insert_code(pformat(fit_dict['beta_loss']))


def _log_forecasts(mdFile, ROOT_DIR, m1_dict, m2_dict):
    _log_plots_util(mdFile, ROOT_DIR, 'forecast-best.png',
                    m2_dict['plots']['forecast_best'], 'Forecast using M1 Best Params')
    _log_plots_util(mdFile, ROOT_DIR, 'forecast-best-50.png',
                    m2_dict['plots']['forecast_best_50'], 'Forecast using M2 best fit, 50th decile params')
    _log_plots_util(mdFile, ROOT_DIR, 'forecast-best-80.png',
                    m2_dict['plots']['forecast_best_80'], 'Forecast using M2 best fit, 80th decile params')
    _log_plots_util(mdFile, ROOT_DIR, 'forecast-ensemble-mean-50.png',
                    m2_dict['plots']['forecast_ensemble_mean_50'], 'Forecast of M2 ensemble fit, 50th decile, best')

    for column, figure in m1_dict['plots']['forecasts_topk'].items():
        _log_plots_util(mdFile, ROOT_DIR, f'forecast-topk-{column}.png',
                        figure, f'Forecast of top k trials for column {column}')

    for column, figure in m2_dict['plots']['forecasts_ptiles'].items():
        _log_plots_util(mdFile, ROOT_DIR, f'forecast-ptiles-{column}.png',
                        figure, f'Forecast of all ptiles for column {column}')

    if 'scenarios' in m2_dict['plots'].keys():
        mdFile.new_header(level=1, title="What if Scenarios")
        for column, figure in m2_dict['plots']['scenarios'].items():
            _log_plots_util(mdFile, ROOT_DIR, f'forecast-scenarios-{column}.png',
                            figure, '')

    mdFile.new_paragraph("---")


def _log_tables(mdFile, m2_dict):
    trials_processed = copy.deepcopy(m2_dict['trials_processed'])
    trials_processed['losses'] = np.around(trials_processed['losses'], 2)
    trials_processed['params'] = [{key: np.around(value, 2) for key, value in params_dict.items()}
                                  for params_dict in trials_processed['params']]
    mdFile.new_header(level=2, title="Top 10 Trials")
    df = pd.DataFrame.from_dict({(i+1, trials_processed['losses'][i]): trials_processed['params'][i] 
                                 for i in range(10)})
    tbl = df.to_markdown()
    mdFile.new_paragraph(tbl)

    deciles = copy.deepcopy(m2_dict['deciles'])
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

def _log_r0_multipliers(mdFile, ROOT_DIR, m2_dict):
    _log_plots_util(mdFile, ROOT_DIR, 'r0-multipliers.png',
                    m2_dict['plots']['r0_mul_dict'], 'R0 Multipliers')

def save_dict_and_create_report(predictions_dict, config_dict, ROOT_DIR='../../misc/reports/', 
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

    m0_dict = predictions_dict['m0']
    m1_dict = predictions_dict['m1']
    m2_dict = predictions_dict['m2']

    _dump_predictions_dict(predictions_dict, ROOT_DIR)

    with open(f'{config_ROOT_DIR}/{config_filename}') as configfile:
        config = yaml.load(configfile, Loader=yaml.SafeLoader)

    os.system(f'cp {config_ROOT_DIR}/{config_filename} {ROOT_DIR}/{config_filename}')
    
    mdFile, filename = _create_md_file(predictions_dict, config, ROOT_DIR)
    _log_hyperparams(mdFile, predictions_dict, config)

    mdFile.new_header(level=1, title='')
    mdFile.new_header(level=1, title='{} - {}'.format(config_dict['fitting']['data']['dataloading_params']['state'], config_dict['fitting']['data']['dataloading_params']['district']))

    if m0_dict['plots']['trends'] is not None:
        _log_daily_trends(mdFile, ROOT_DIR, m0_dict)

    if m1_dict['plots']['smoothing'] is not None:
       _log_smoothing(mdFile, ROOT_DIR, m1_dict)

    mdFile.new_header(level=1, title=f'FITS')
    _log_fits(mdFile, ROOT_DIR, m0_dict, which_fit='M0')
    _log_fits(mdFile, ROOT_DIR, m1_dict, which_fit='M1')
    _log_fits(mdFile, ROOT_DIR, m2_dict, which_fit='M2')

    mdFile.new_header(level=2, title=f'Uncertainty Fitting')
    _log_uncertainty_fit(mdFile, m2_dict)

    mdFile.new_header(level=1, title=f'FORECASTS')
    _log_forecasts(mdFile, ROOT_DIR, m1_dict, m2_dict)
    
    mdFile.new_header(level=1, title="Tables")
    _log_tables(mdFile, m2_dict)

    mdFile.new_header(level=1, title="What IFs - R0 Multipliers")
    _log_r0_multipliers(mdFile, ROOT_DIR, m2_dict)

    # Create a table of contents
    mdFile.new_table_of_contents(table_title='Contents', depth=2)
    mdFile.create_md_file()

    pypandoc.convert_file("{}.md".format(filename), 'docx', outputfile="{}.docx".format(filename))
    #pypandoc.convert_file("{}.docx".format(filename), 'pdf', outputfile="{}.pdf".format(filename))
    # TODO: pdf conversion has some issues with order of images, low priority
