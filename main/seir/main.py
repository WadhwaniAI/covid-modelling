import copy
import datetime

from tabulate import tabulate

import main.seir.optimisers as optimisers
from data.processing.processing import get_data, train_val_test_split
from utils.fitting.loss import Loss_Calculator
from utils.fitting.smooth_jump import smooth_big_jump, smooth_big_jump_stratified
from viz import plot_fit, plot_smoothing


def data_setup(dataloader, dataloading_params, data_columns, smooth_jump, smooth_jump_params, split,
               loss_compartments, rolling_average, rolling_average_params, **kwargs):
    """Helper function for single_fitting_cycle where data is loaded from given params input.
    Smoothing is done if smoothing params are given as well. And then rolling average is done and 
    the train val test split is implemented

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

    Returns:
        dict: Dict of processed dataframes and ideal params (if present)
    """

    data_dict = get_data(dataloader, dataloading_params, data_columns)
    df_district = data_dict['data_frame']

    smoothing_plot = None
    orig_df_district = copy.copy(df_district)

    if smooth_jump:
        if dataloading_params['stratified_data']:
            df_params_copy = copy.copy(dataloading_params)
            df_params_copy['stratified_data'] = False
            df_not_strat = get_data(dataloader, df_params_copy,
                                    data_columns)['data_frame']
            df_district, description = smooth_big_jump_stratified(
                df_district, df_not_strat, smooth_jump_params)
        else:
            df_district, description = smooth_big_jump(df_district, smooth_jump_params)

        smoothing_plot = plot_smoothing(orig_df_district, df_district, dataloading_params['state'],
                                        dataloading_params['district'], which_compartments=loss_compartments,
                                        description='Smoothing')
    df_district['daily_cases'] = df_district['total'].diff()
    df_district.dropna(axis=0, how='any', subset=['total'],
                       inplace=True)
    df_district.reset_index(drop=True, inplace=True)

    smoothing = {}
    if smooth_jump:
        smoothing = {
            'smoothing_description': description,
            'smoothing_plot': smoothing_plot,
            'df_district_unsmoothed': orig_df_district
        }
        print(smoothing['smoothing_description'])

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

    processed_dataframes = {
        'df_train': df_train,
        'df_val': df_val,
        'df_test': df_test,
        'df_train_nora': df_train_nora,
        'df_val_nora': df_val_nora,
        'df_test_nora': df_test_nora,
        'df_district': df_district
    }
    if 'ideal_params' in data_dict:
        return {"processed_dataframes": processed_dataframes, "smoothing": smoothing,
                "ideal_params": data_dict['ideal_params']}
    return {"processed_dataframes": processed_dataframes, "smoothing": smoothing}


def run_cycle(processed_dataframes, data_args, model, variable_param_ranges, default_params, optimiser,
              optimiser_params, split, loss, forecast):
    """Helper function for single_fitting_cycle where the fitting actually takes place.

    Args:
        processed_dataframes (dict): Dict of processed dataframes
        data_args (dict): Dict of data_args
        model (class): The model class to be used during fitting/training
        variable_param_ranges (dict): Dict of searchspace ranges for all params
        default_params (dict): Dict of static params
        optimiser (str): Name of the optimiser class
        optimiser_params (dict): Dict of optimiser params
        split (dict): Dict of train val test split params
        loss (dict): Dict of loss params
        forecast (dict): Dict of forecasting params

    Returns:
        dict: A predictions_dict file with the results of the fitting and more
    """
    results_dict = {}
    df_train, df_val, _, df_train_nora, df_val_nora, df_test_nora, df_district = [
        processed_dataframes.get(k) for k in processed_dataframes.keys()]

    # Initialise Optimiser
    op = getattr(optimisers, optimiser)(model, df_train, default_params,
                                        variable_param_ranges, 
                                        train_period=split['train_period'])
    # Perform Optimisation
    args = {**optimiser_params, **split, **loss, **forecast}
    trials = op.optimise(**args)
    import pdb;pdb.set_trace()
    print('best parameters\n', trials['params'][0])

    df_prediction = trials['predictions'][0]

    lc = Loss_Calculator()
    df_loss = lc.create_loss_dataframe_region(df_train_nora, df_val_nora, df_test_nora, df_prediction,
                                              loss_method=loss['loss_method'],
                                              loss_compartments=loss['loss_compartments'])

    fit_plot = plot_fit(df_prediction, df_train, df_val, df_district, split['train_period'], 
                        location_description=data_args['dataloading_params']['location_description'],
                        which_compartments=loss['loss_compartments'])

    results_dict['plots'] = {}
    results_dict['plots']['fit'] = fit_plot
    data_last_date = df_district.iloc[-1]['date'].strftime("%Y-%m-%d")

    fitting_date = datetime.datetime.now().strftime("%Y-%m-%d")
    results_dict = {
        'default_params': default_params,
        'df_prediction': df_prediction,
        'df_district': df_district,
        'df_train': df_train,
        'df_val': df_val,
        'df_loss': df_loss,
        'trials': trials,
        'data_last_date': data_last_date,
        'fitting_date': fitting_date
    }

    return results_dict


def single_fitting_cycle(data_args, model_family, model, variable_param_ranges, default_params, 
                         optimiser, optimiser_params, split, loss, forecast):
    """Main function which user runs for running an entire fitting cycle for a particular data input

    Args:
        data_args (dict): Dict of data_args
        model_family (str): The name of the family the model class belongs to
        model (class): The model class to be used during fitting/training
        variable_param_ranges (dict): Dict of searchspace ranges for all params
        default_params (dict): Dict of static params
        optimiser (str): Name of the optimiser class
        optimiser_params (dict): Dict of optimiser params
        split (dict): Dict of train val test split params
        loss (dict): Dict of loss params
        forecast (dict): Dict of forecasting params

    Returns:
        dict: A predictions_dict file with the results of the fitting and more
    """
    # record parameters for reproducibility
    run_params = locals()
    run_params['model'] = model.__name__
    run_params['model_class'] = model
    
    print('Performing fit ..')

    # Get data
    params = {**data_args}
    params['split'] = split
    params['loss_compartments'] = loss['loss_compartments']
    data_dict = data_setup(**params)
    processed_dataframes, smoothing = data_dict['processed_dataframes'], data_dict['smoothing']
    smoothing_plot = smoothing['smoothing_plot'] if 'smoothing_plot' in smoothing else None
    smoothing_description = smoothing['smoothing_description'] if 'smoothing_description' in smoothing else None

    orig_df_district = smoothing['df_district_unsmoothed'] if 'df_district_unsmoothed' in smoothing else None

    print('train\n', tabulate(processed_dataframes['df_train'].tail().round(2).T, headers='keys', tablefmt='psql'))
    if not processed_dataframes['df_val'] is None:
        print('val\n', tabulate(processed_dataframes['df_val'].tail().round(2).T, headers='keys', tablefmt='psql'))
        
    predictions_dict = run_cycle(processed_dataframes, data_args, model, variable_param_ranges,
        default_params, optimiser, optimiser_params, split, loss, forecast)

    predictions_dict['plots']['smoothing'] = smoothing_plot
    predictions_dict['smoothing_description'] = smoothing_description
    predictions_dict['df_district_unsmoothed'] = orig_df_district

    # record parameters for reproducibility
    predictions_dict['run_params'] = run_params
    if 'ideal_params' in data_dict:
        predictions_dict['ideal_params'] = data_dict['ideal_params']
    return predictions_dict
