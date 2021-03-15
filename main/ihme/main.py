"""
main.py
"""
import sys
import copy
from datetime import timedelta
from tabulate import tabulate

import numpy as np
import pandas as pd

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


def get_covariates(df, covariates):
    df.loc[:, 'default_cov'] = len(df) * [1.0]
    if covariates[1] != 'default_cov':
        pass  # TODO: Get covariate for beta
    return df


def data_setup(dataloader, dataloading_params, data_columns, smooth_jump, smooth_jump_params, split,
               loss_compartments, rolling_average, rolling_average_params, population, covariates, **kwargs):
    """Helper function for single_fitting_cycle where data from different sources (given input) is imported

    Creates the following dataframes:
        df_train, df_val, df_test: Train, val and test splits which have been smoothed and transformed
        df_train_nora, df_val_nora, df_test_nora: Train, val and test splits which have NOT been smoothed,
            but have been transformed
        df_train_nora_notrans, df_val_nora_notrans, df_test_nora_notrans: Train, val and test splits which have
        neither been smoothed nor transformed

    Args:
        dataloader (str): Name of the dataloader class
        dataloading_params (dict): Dict of dataloading params
        data_columns (list(str)): List of columns output dataframe is expected to have
        smooth_jump (bool): If true, smoothing is done
        smooth_jump_params (list): List of smooth jump params
        split (dict): Dict of params for train val test split
        loss_compartments (list): List of compartments to apply loss on
        rolling_average (bool): If true, rolling average is done
        rolling_average_params (dict): Dict of rolling average params
        population (int): population of the region
        covariates ():
        **kwargs ():

    Returns:

    """
    # Fetch data dictionary
    data_dict = get_data(dataloader, dataloading_params, data_columns)
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
    df_district.dropna(axis=0, how='any', subset=['total'], inplace=True)  # TODO: Replace with ycol?
    df_district.reset_index(drop=True, inplace=True)

    # Make a copy of data without transformation or smoothing
    df_district_notrans = copy.deepcopy(df_district)

    # Convert data to population normalized rate and apply log transformation
    df_district = transform_data(df_district, population)

    # Add group and covs columns
    df_district.loc[:, 'group'] = len(df_district) * [1.0]
    # df_district.loc[:, 'covs'] = len(df_district) * [1.0]
    df_district = get_covariates(df_district, covariates)

    df_district.loc[:, 'day'] = (df_district['date'] - np.min(df_district['date'])).apply(lambda x: x.days)

    # Perform split with/without rolling average
    rap = rolling_average_params
    if rolling_average:
        df_train, df_val, df_test = train_val_test_split(
            df_district, train_period=split['train_period'], val_period=split['val_period'],
            test_period=split['test_period'], start_date=split['start_date'], end_date=split['end_date'],
            window_size=rap['window_size'], center=rap['center'],
            win_type=rap['win_type'], min_periods=rap['min_periods'], trim_excess=True)
    else:
        df_train, df_val, df_test = train_val_test_split(
            df_district, train_period=split['train_period'], val_period=split['val_period'],
            test_period=split['test_period'], start_date=split['start_date'], end_date=split['end_date'],
            window_size=1, trim_excess=True)

    df_train_nora, df_val_nora, df_test_nora = train_val_test_split(
        df_district, train_period=split['train_period'], val_period=split['val_period'],
        test_period=split['test_period'], start_date=split['start_date'], end_date=split['end_date'],
        window_size=1, trim_excess=True)

    df_train_nora_notrans, df_val_nora_notrans, df_test_nora_notrans = train_val_test_split(
        df_district_notrans, train_period=split['train_period'], val_period=split['val_period'],
        test_period=split['test_period'], start_date=split['start_date'], end_date=split['end_date'],
        window_size=1, trim_excess=True)

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
    col = model_params['ycol']
    transformed_col = f'log_{col}_rate' if data['log'] else f'{col}_rate'
    model_params['ycol'] = transformed_col

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
        col: df_prediction[transformed_col].apply(lambda x: transform_func(x, default_params['N']))
    })
    # Obtain mean and pointwise losses for train, val and test
    df_loss = lc.create_loss_dataframe_region(df_train_nora_notrans, df_val_nora_notrans, df_test_nora_notrans,
                                              df_prediction_notrans, which_compartments=[col])
    df_loss_pointwise = lc.create_pointwise_loss_dataframe_region(df_test_nora_notrans, df_val_nora_notrans,
                                                                  df_test_nora_notrans, df_prediction_notrans,
                                                                  which_compartments=[col])

    # Uncertainty
    draws = get_uncertainty_draws(model, model_params)

    # Plotting
    fit_plot = plot_fit(df_prediction_notrans, df_train_nora_notrans, df_val_nora_notrans, df_district_notrans,
                        split['train_period'], location_description=data['dataloading_params']['location_description'],
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
