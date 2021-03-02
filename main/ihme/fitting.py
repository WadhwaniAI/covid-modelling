"""
fitting.py
"""
import os
import sys
import copy
from datetime import timedelta
from tabulate import tabulate

import numpy as np
import pandas as pd
from curvefit.core import functions

from data.processing.processing import get_data, train_val_test_split
from viz import plot_smoothing, plot_fit

sys.path.append('../..')

from models.ihme.model import IHME
from utils.fitting.data import lograte_to_cumulative, rate_to_cumulative
from utils.fitting.loss import Loss_Calculator
from main.ihme.optimiser import Optimiser
from main.ihme.forecast import get_uncertainty_draws
from utils.fitting.smooth_jump import smooth_big_jump


def transform_data(df, population):
    """Adds population normalized versions for each time series in the dataframe

    Args:
        df (pd.DataFrame): dataframe of time series
        population (int): population of the region

    Returns:
        pd.DataFrame: dataframe of time series with additional columns
    """
    data = df.set_index('date')
    which_columns = df.select_dtypes(include='number').columns
    for column in which_columns:
        if column in data.columns:
            data[f'{column}_rate'] = data[column] / population
            data[f'log_{column}_rate'] = data[f'{column}_rate'].apply(lambda x: np.log(x))
    data = data.reset_index()
    data['date'] = pd.to_datetime(data['date'])
    return data


def data_setup(data_source, dataloading_params, smooth_jump, smooth_jump_params, split,
               loss_compartments, rolling_average, rolling_average_params, population, **kwargs):
    """Helper function for single_fitting_cycle where data from different sources (given input) is imported

    Creates the following dataframes:
        df_train, df_val, df_test: Train, val and test splits which have been smoothed and transformed
        df_train_nora, df_val_nora, df_test_nora: Train, val and test splits which have NOT been smoothed,
            but have been transformed
        df_train_nora_notrans, df_val_nora_notrans, df_test_nora_notrans: Train, val and test splits which have
        neither been smoothed nor transformed

    Args:
        data_source ():
        dataloading_params ():
        smooth_jump ():
        smooth_jump_params ():
        split ():
        loss_compartments ():
        rolling_average ():
        rolling_average_params ():
        population():
        **kwargs ():

    Returns:

    """
    # Fetch data dictionary
    data_dict = get_data(data_source, dataloading_params)
    df_district = data_dict['data_frame']

    # Make a copy of original unsmoothed data
    orig_df_district = copy.copy(df_district)

    # Smoothing operations
    smoothing = {}
    if smooth_jump:
        # Perform smoothing
        df_district, description = smooth_big_jump(df_district, smooth_jump_params)

        # Plot smoothed data
        smoothing_plot = plot_smoothing(orig_df_district, df_district, dataloading_params['state'],
                                        dataloading_params['district'], which_compartments=loss_compartments,
                                        description='Smoothing')
        smoothing = {
            'smoothing_description': description,
            'smoothing_plot': smoothing_plot,
            'df_district_unsmoothed': orig_df_district
        }
        print(smoothing['smoothing_description'])

    # Drop rows with NA values
    df_district.dropna(axis=0, how='any', subset=['total'], inplace=True)
    df_district.reset_index(drop=True, inplace=True)

    # Make a copy of data without transformation or smoothing
    df_district_notrans = copy.deepcopy(df_district)

    # Convert data to population normalized rate and apply log transformation
    df_district = transform_data(df_district, population)

    # Add group and covs columns
    df_district.loc[:, 'group'] = len(df_district) * [1.0]
    df_district.loc[:, 'covs'] = len(df_district) * [1.0]

    df_district.loc[:, 'day'] = (df_district['date'] - np.min(df_district['date'])).apply(lambda x: x.days)

    # Perform split with/without rolling average
    rap = rolling_average_params
    if rolling_average:
        df_train, df_val, df_test = train_val_test_split(
            df_district, train_period=split['train_period'], val_period=split['val_period'],
            test_period=split['test_period'], start_date=split['start_date'], end_date=split['end_date'],
            window_size=rap['window_size'], center=rap['center'],
            win_type=rap['win_type'], min_periods=rap['min_periods'])
    else:
        df_train, df_val, df_test = train_val_test_split(
            df_district, train_period=split['train_period'], val_period=split['val_period'],
            test_period=split['test_period'], start_date=split['start_date'], end_date=split['end_date'],
            window_size=1)

    df_train_nora, df_val_nora, df_test_nora = train_val_test_split(
        df_district, train_period=split['train_period'], val_period=split['val_period'],
        test_period=split['test_period'], start_date=split['start_date'], end_date=split['end_date'],
        window_size=1)

    df_train_nora_notrans, df_val_nora_notrans, df_test_nora_notrans = train_val_test_split(
        df_district_notrans, train_period=split['train_period'], val_period=split['val_period'],
        test_period=split['test_period'], start_date=split['start_date'], end_date=split['end_date'],
        window_size=1)

    observed_dataframes = {}
    for name in ['df_district', 'df_district_notrans',
                 'df_train', 'df_val', 'df_test',
                 'df_train_nora', 'df_val_nora', 'df_test_nora',
                 'df_train_nora_notrans', 'df_val_nora_notrans', 'df_test_nora_notrans']:
        observed_dataframes[name] = eval(name)

    return {"observed_dataframes": observed_dataframes, "smoothing": smoothing}


def run_cycle(observed_dataframes, data, model, model_params, default_params, fitting_method_params, split, loss):
    """

    Args:
        observed_dataframes ():
        data ():
        model ():
        model_params ():
        default_params ():
        fitting_method_params ():
        split ():
        loss ():

    Returns:

    """
    # Set up model params TODO: Shift to read_config
    model_params['func'] = getattr(functions, model_params['func'])
    model_params['covs'] = data['covariates']

    # Initialize the model
    model = model(model_params)

    # Get the required data
    df_district, df_district_notrans, df_train, df_val, df_test, \
        df_train_nora, df_val_nora, df_test_nora, df_train_nora_notrans, df_val_nora_notrans, df_test_nora_notrans = [
            observed_dataframes.get(k) for k in observed_dataframes.keys()]

    # Results dictionary
    results_dict = {}

    # Initialize the optimizer
    args = {
        'bounds': copy.copy(model.priors['fe_bounds']),
        'iterations': fitting_method_params['num_evals'],
        'scoring': loss['loss_method'],
        'num_trials': fitting_method_params['num_trials']
    }
    optimiser = Optimiser(model, df_train, df_val, args)

    # Optimize initial parameters
    model.priors['fe_init'], trials = optimiser.optimise()
    results_dict['best_init'] = model.priors['fe_init']

    # Model fitting
    fitting_data = pd.concat([df_train, df_val], axis=0)
    model.fit(fitting_data)

    # Prediction
    # Create dataframe with date and prediction columns
    df_prediction = pd.DataFrame(columns=[model.date, model.ycol])
    # Set date column
    df_prediction.loc[:, model.date] = pd.date_range(df_train[model.date].min(), df_district.iloc[-1, :][model.date])
    # Get predictions from start of train period until last available date of data
    df_prediction.loc[:, model.ycol] = model.predict(df_train[model.date].min(),
                                                     df_train[model.date].min() + timedelta(days=len(df_prediction)-1))

    # Evaluation
    lc = Loss_Calculator()
    # Get inverse transformation function to transform predictions
    transform_func = lograte_to_cumulative if data['log'] else rate_to_cumulative
    # Create dataframe of transformed predictions
    df_prediction_notrans = pd.DataFrame({
        model.date: df_prediction[model.date],
        model.ycol: df_prediction[model.ycol].apply(lambda x: transform_func(x, default_params['N']))
    })
    # Obtain mean and pointwise losses for train, val and test
    df_loss = lc.create_loss_dataframe_region(df_train_nora_notrans, df_val_nora_notrans, df_test_nora_notrans,
                                              df_prediction_notrans, which_compartments=loss['loss_compartments'])
    df_loss_pointwise = lc.create_pointwise_loss_dataframe_region(df_test_nora_notrans, df_val_nora_notrans,
                                                                  df_test_nora_notrans, df_prediction_notrans,
                                                                  which_compartments=loss['loss_compartments'])

    # Uncertainty TODO: Shift to forecast section of code
    draws = get_uncertainty_draws(model, model_params)

    # Plotting
    fit_plot = plot_fit(df_prediction, df_train, df_val, df_district, split['train_period'],
                        location_description=data['dataloading_params']['location_description'],
                        which_compartments=loss['loss_compartments'])

    # Collect results
    results_dict['plots'] = {}
    results_dict['plots']['fit'] = fit_plot
    results_dict['best_params'] = model.pipeline.mod.params
    results_dict['df_prediction'] = df_prediction_notrans
    results_dict['df_district'] = df_district_notrans
    data_last_date = df_district.iloc[-1]['date'].strftime("%Y-%m-%d")
    for name in ['optimiser', 'df_train', 'df_val', 'df_test',
                 'df_train_nora_notrans', 'df_val_nora_notrans', 'df_test_nora_notrans',
                 'df_loss', 'df_loss_pointwise', 'trials',
                 'data_last_date', 'draws']:
        results_dict[name] = eval(name)

    return results_dict


def single_fitting_cycle(data, model, model_params, default_params, fitting_method_params, split, loss):
    """

    Args:
        data ():
        model ():
        model_params ():
        default_params ():
        fitting_method_params ():
        split ():
        loss ():

    Returns:

    """

    # record parameters for reproducibility
    run_params = locals()
    run_params['model'] = model.__name__
    run_params['model_class'] = model

    print('Performing {} fit ..'.format('m2' if split['val_period'] == 0 else 'm1'))

    # Get data
    params = {**data}
    params['split'] = split
    params['loss_compartments'] = loss['loss_compartments']
    params['population'] = default_params['N']
    data_dict = data_setup(**params)

    observed_dataframes, smoothing = data_dict['observed_dataframes'], data_dict['smoothing']
    smoothing_plot = smoothing['smoothing_plot'] if 'smoothing_plot' in smoothing else None
    smoothing_description = smoothing['smoothing_description'] if 'smoothing_description' in smoothing else None

    orig_df_district = smoothing['df_district_unsmoothed'] if 'df_district_unsmoothed' in smoothing else None

    print('train\n', tabulate(observed_dataframes['df_train'].tail().round(2).T, headers='keys', tablefmt='psql'))
    if not observed_dataframes['df_val'] is None:
        print('val\n', tabulate(observed_dataframes['df_val'].tail().round(2).T, headers='keys', tablefmt='psql'))

    predictions_dict = run_cycle(observed_dataframes, data, model, model_params, default_params, fitting_method_params,
                                 split, loss)

    predictions_dict['plots']['smoothing'] = smoothing_plot
    predictions_dict['smoothing_description'] = smoothing_description
    predictions_dict['df_district_unsmoothed'] = orig_df_district

    # record parameters for reproducibility
    predictions_dict['run_params'] = run_params
    return predictions_dict


def create_output_folder(fname):
    """
    creates folder in outputs/ihme/

    Args:
        fname (str): name of folder within outputs/ihme

    Returns:
        str: output_folder path
    """
    output_folder = f'../../outputs/ihme/{fname}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    return output_folder


# def run_cycle_compartments(dataframes, model_params, which_compartments=Columns.curve_fit_compartments(),
#                            forecast_days=30, max_evals=1000, num_hyperopt=1, val_size=7, min_days=7, scoring='mape',
#                            dtp=None, log=True, **config):
#     """
#     runs fitting cycles for all compartments in which_compartments
#     model_params['ycol'] is ignored here
#
#     Args:
#         dataframes (dict): contains smoothed and unsmoothed dataframes: train, test, df
#         model_params (dict): model_params
#         which_compartments (list, optional): List of compartments to fit. Defaults to Columns.curve_fit_compartments().
#         forecast_days (int, optional): how far to predict. Defaults to 30.
#         max_evals (int, optional): num evals in hyperparam optimisation. Defaults to 1000.
#         num_hyperopt (int, optional): number of times to run hyperopt in parallel. Defaults to 1.
#         val_size (int, optional): val size - hyperopt. Defaults to 7.
#         min_days (int, optional): min train_period. Defaults to 7.
#         scoring (str, optional): 'mape', 'rmse', or 'rmsle. Defaults to 'mape'.
#         dtp ([type], optional): district total population. Defaults to None.
#         log (bool, optional): whether to fit to log(rate). Defaults to True.
#
#     Returns:
#         dict: results_dict
#     """
#     xform_func = lograte_to_cumulative if log else rate_to_cumulative
#     compartment_names = [col.name for col in which_compartments]
#     results = {}
#     ycols = {col: '{log}{colname}_rate'.format(log='log_' if log else '', colname=col.name) for col in
#              which_compartments}
#     loss_dict = dict()
#     for i, col in enumerate(which_compartments):
#         col_params = copy.copy(model_params)
#         if config.get("active_log_derf", False) and col.name == 'hospitalised':
#             col_params['func'] = functions.log_derf
#         col_params['ycol'] = ycols[col]
#         results[col.name] = run_cycle(
#             dataframes, col_params, dtp=dtp, xform_func=xform_func,
#             max_evals=max_evals, num_hyperopt=num_hyperopt, val_size=val_size,
#             min_days=min_days, scoring=scoring, log=log, forecast_days=forecast_days,
#             **config)
#
#         # Aggregate Results
#         pred = results[col.name]['df_prediction'].set_index('date')
#         loss_dict[col.name] = results[col.name]['df_loss']
#         if i == 0:
#             predictions = pd.DataFrame(index=pred.index, columns=compartment_names + list(ycols.values()), dtype=float)
#
#         predictions.loc[pred.index, col.name] = xform_func(pred[ycols[col]], dtp)
#         predictions.loc[pred.index, ycols[col]] = pred[ycols[col]]
#
#     df_loss = pd.concat(loss_dict.values(), axis=0, keys=compartment_names,
#                         names=['compartment', 'split', 'loss_function'])
#     df_loss.name = 'loss'
#     df_train_loss_pointwise = pd.concat([results[comp]['df_train_loss_pointwise'] for comp in compartment_names],
#                                         keys=compartment_names, names=['compartment', 'split', 'loss_function'])
#     df_test_loss_pointwise = pd.concat([results[comp]['df_test_loss_pointwise'] for comp in compartment_names],
#                                        keys=compartment_names, names=['compartment', 'split', 'loss_function'])
#
#     predictions.reset_index(inplace=True)
#     df_train = dataframes['train'][compartment_names + list(ycols.values()) + [model_params['date']]]
#     df_val = dataframes['test'][compartment_names + list(ycols.values()) + [model_params['date']]]
#     df_district = dataframes['df'][compartment_names + list(ycols.values()) + [model_params['date']]]
#     df_train_nora = dataframes['train_nora'][compartment_names + list(ycols.values()) + [model_params['date']]]
#     df_val_nora = dataframes['test_nora'][compartment_names + list(ycols.values()) + [model_params['date']]]
#     df_district_nora = dataframes['df_nora'][compartment_names + list(ycols.values()) + [model_params['date']]]
#
#     draws = None
#     if model_params['pipeline_args']['n_draws'] > 0:
#         draws = {
#             col.name: {
#                 'draws': xform_func(results[col.name]['draws'], dtp),
#                 'no_xform_draws': results[col.name]['draws'],
#             } for col in which_compartments
#         }
#
#     final = {
#         'best_params': {col.name: results[col.name]['best_params'] for col in which_compartments},
#         'variable_param_ranges': model_params['priors']['fe_bounds'],
#         'n_days': {col.name: results[col.name]['n_days'] for col in which_compartments},
#         'df_prediction': predictions,
#         'df_district': df_district,
#         'df_train': df_train,
#         'df_val': df_val,
#         'df_district_nora': df_district_nora,
#         'df_train_nora': df_train_nora,
#         'df_val_nora': df_val_nora,
#         'df_loss': df_loss,
#         'df_train_loss_pointwise': df_train_loss_pointwise,
#         'df_test_loss_pointwise': df_test_loss_pointwise,
#         'data_last_date': df_district[model_params['date']].max(),
#         'draws': draws,
#         'mod.params': {col.name: results[col.name]['mod.params'] for col in which_compartments},
#         'individual_results': results,
#         'district_total_pop': results[which_compartments[0].name]['district_total_pop'],
#     }
#
#     return final
