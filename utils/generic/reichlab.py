from utils.fitting.loss import Loss_Calculator
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pytz import timezone
import copy
import re

from data.dataloader import JHULoader
from utils.generic.config import read_config
from data.processing.processing import get_dataframes_cached

"""
Helper functions for processing different reichlab submissions, processing reichlab ground truth,
Comparing reichlab models with gt, processing and formatting our (Wadhwani AI) submission, 
comparing that with gt as well
"""


def get_mapping(which='location_name_to_code', reichlab_path='../../../covid19-forecast-hub', read_from_github=False):
    if read_from_github:
        reichlab_path = 'https://raw.githubusercontent.com/reichlab/covid19-forecast-hub/master'
    df = pd.read_csv(f'{reichlab_path}/data-locations/locations.csv')
    df.dropna(how='any', axis=0, inplace=True)
    if which == 'location_name_to_code':
        mapping_dict = dict(zip(df['location_name'], df['location']))
    elif which == 'location_name_to_abbv':
        mapping_dict = dict(zip(df['location_name'], df['abbreviation']))
    else:
        mapping_dict = {}
    return mapping_dict


def get_list_of_models(date_of_submission, comp, reichlab_path='../../../covid19-forecast-hub', read_from_github=False, 
                       location_id_filter=78, num_submissions_filter=50):
    """Given an input of submission date, comp, gets list of all models that submitted.

    Args:
        date_of_submission (str): The ensemble creation date (always a Mon), for selecting a particular week
        comp (str): Which compartment (Can be 'inc_case', 'cum_case', 'inc_death', or 'cum_death')
        reichlab_path (str, optional): Path to reichlab repo (if cloned on machine). 
        Defaults to '../../../covid19-forecast-hub'.
        read_from_github (bool, optional): If true, reads files directly from github 
        instead of cloned repo. Defaults to False.
        location_id_filter (int, optional): Only considers locations with location code <= this input. 
        Defaults to 78. All states, territories have code <= 78. > 78, locations are counties
        num_submissions_filter (bool, optional): Only selects models with submissions more than this.
        Defaults to 50.

    Returns:
        list: list of eligible models
    """
    if comp == 'cum_case':
        comp = 'inc_case'
    if read_from_github:
        reichlab_path = 'https://raw.githubusercontent.com/reichlab/covid19-forecast-hub/master'
    try:
        df = pd.read_csv(f'{reichlab_path}/ensemble-metadata/' + \
            f'{date_of_submission}-{comp}-model-eligibility.csv')
    except:
        date_convert = datetime.strptime(date_of_submission, '%Y-%m-%d')
        date_of_filename = (date_convert - timedelta(days=1)).date()
        df = pd.read_csv(f'{reichlab_path}/ensemble-metadata/' +
                         f'{date_of_filename}-{comp}-model-eligibility.csv')
    df['location'] = df['location'].apply(lambda x : int(x) if x != 'US' else 0)
    all_models = list(df['model'])

    df_all_states = df[df['location'] <= location_id_filter]
    df_eligible = df_all_states[df_all_states['overall_eligibility'] == 'eligible']

    df_counts = df_eligible.groupby('model').count()
    
    # Filter all models with > num_submissions_filter submissions
    df_counts = df_counts[df_counts['overall_eligibility'] > num_submissions_filter]
    eligible_models = list(df_counts.index)
    # Add Wadhwani_AI-BayesOpt incase it isn't a part of the list
    if ('Wadhwani_AI-BayesOpt' in all_models) & ('Wadhwani_AI-BayesOpt' not in eligible_models):
        eligible_models.append('Wadhwani_AI-BayesOpt')
    print(eligible_models)
    return eligible_models


def process_single_submission(model, date_of_submission, comp, df_true, reichlab_path='../../../covid19-forecast-hub', 
                              read_from_github=False, location_id_filter=78, num_weeks_filter=4):
    """Processes the CSV file of a single submission (one model, one instance of time)

    Args:
        model (str): The model name to process CSV of 
        date_of_submission (str): The ensemble creation date (always a Mon), for selecting a particular week
        comp (str): Which compartment (Can be 'inc_case', 'cum_case', 'inc_death', or 'cum_death')
        df_true (pd.DataFrame): The ground truth dataframe (Used for processing cum_cases submissions)
        reichlab_path (str, optional): Path to reichlab repo (if cloned on machine). 
        Defaults to '../../../covid19-forecast-hub'.
        read_from_github (bool, optional): If true, reads files directly from github 
        instead of cloned repo. Defaults to False.
        location_id_filter (int, optional): All location ids <= this will be kept. Defaults to 78.
        num_weeks_filter (int, optional): Only forecasts num_weeks_filter weeks ahead 
        will be kept. Defaults to 4.

    Returns:
        pd.DataFrame: model submssion processed dataframe
    """
    if read_from_github:
        reichlab_path = 'https://raw.githubusercontent.com/reichlab/covid19-forecast-hub/master'
    try:
        df = pd.read_csv(f'{reichlab_path}/data-processed/' + \
            f'{model}/{date_of_submission}-{model}.csv')
    except:
        date_convert = datetime.strptime(date_of_submission, '%Y-%m-%d')
        date_of_filename = date_convert - timedelta(days=1)
        try:
            df = pd.read_csv(f'{reichlab_path}/data-processed/' + \
                f'{model}/{date_of_filename.strftime("%Y-%m-%d")}-{model}.csv')
        except:
            date_of_filename = date_of_filename - timedelta(days=1)
            try:
                df = pd.read_csv(f'{reichlab_path}/data-processed/' + \
                    f'{model}/{date_of_filename.strftime("%Y-%m-%d")}-{model}.csv')
            except:
                return None
    # Converting all locations to integers
    df['location'] = df['location'].apply(lambda x : int(x) if x != 'US' else 0)
    # Keeping only states and territories forecasts
    df = df[df['location'] <= location_id_filter]
    df['model'] = model
    
    # Only keeping the wk forecasts
    df = df[df['target'].apply(lambda x : 'wk' in x)]
    
    # Only the forecasts corresponding the comp user are interested in
    if comp == 'cum_case':
        df = df[df['target'].apply(lambda x : 'inc_case'.replace('_', ' ') in x)]
    else:
        df = df[df['target'].apply(lambda x : comp.replace('_', ' ') in x)]
    
    # Pruning the forecasts which are beyond 4 weeks ahead
    df = df[df['target'].apply(lambda x : int(re.findall(r'\d+', x)[0])) <= num_weeks_filter]
    
    df['target_end_date'] = pd.to_datetime(df['target_end_date'])
    df['forecast_date'] = pd.to_datetime(df['forecast_date'])

    if comp == 'cum_case':
        grouped = df.groupby(['location', 'type', 'quantile'], dropna=False)
        df_cumsum = pd.DataFrame(columns=df.columns)
        for _, group in grouped:
            group['value'] = group['value'].cumsum()
            df_cumsum = pd.concat([df_cumsum, group], ignore_index=True)
        
        gt_cases = df_true.loc[df_true['date'] == df_cumsum['target_end_date'].min() -
                               timedelta(days=7), ['Province_State','Confirmed']]
        loc_code_df = pd.read_csv(
            f'{reichlab_path}/data-locations/locations.csv')
        gt_cases = gt_cases.merge(loc_code_df, left_on='Province_State', 
                                  right_on='location_name')
        gt_cases.drop(['Province_State', 'abbreviation',
                       'location_name', 'population'], axis=1, inplace=True)
        gt_cases['location'] = gt_cases['location'].astype(int)
        gt_cases = gt_cases[gt_cases['location'] < 100]
        gt_cases.reset_index(drop=True, inplace=True)
        gt_cases.loc[len(gt_cases), :] = [int(gt_cases.sum(axis=0)['Confirmed']), 0]

        df_cumsum = df_cumsum.merge(gt_cases)
        df_cumsum['value'] = df_cumsum['value'] + df_cumsum['Confirmed']
        df_cumsum.drop(['Confirmed'], axis=1, inplace=True)
        df_cumsum['target'] = df_cumsum['target'].apply(
            lambda x: x.replace('inc case', 'cum case'))
        df = df_cumsum

    return df


def process_all_submissions(list_of_models, date_of_submission, comp, reichlab_path='../../../covid19-forecast-hub', 
                            read_from_github=False, location_id_filter=78, num_weeks_filter=4):
    """Process submissions for all models given as input and concatenate them

    Args:
        list_of_models (list): List of all models to process submission for. Typically output of get_list_of_models
        date_of_submission (str): The ensemble creation date (always a Mon), for selecting a particular week
        comp (str): Which compartment (Can be 'inc_case', 'cum_case', 'inc_death', or 'cum_death')
        reichlab_path (str, optional): Path to reichlab repo (if cloned on machine). 
        Defaults to '../../../covid19-forecast-hub'.
        read_from_github (bool, optional): If true, reads files directly from github 
        instead of cloned repo. Defaults to False.
        location_id_filter (int, optional): All location ids <= this will be kept. Defaults to 78.
        num_weeks_filter (int, optional): Only forecasts num_weeks_filter weeks ahead 
        will be kept. Defaults to 4.

    Returns:
        pd.DataFrame: Dataframe with all submissions processed
    """
    dataframes = get_dataframes_cached(loader_class=JHULoader)
    df_true = dataframes['df_us_states']
    df_all_submissions = process_single_submission(
        list_of_models[0], date_of_submission, comp, df_true, reichlab_path, read_from_github,
        location_id_filter, num_weeks_filter)
    if df_all_submissions is None:
        raise AssertionError('list_of_models[0] has no submission on Monday, Sunday or Saturday' + \
            '. Please skip it')
    for model in list_of_models:
        df_model_subm = process_single_submission(
            model, date_of_submission, comp, df_true, reichlab_path, read_from_github, 
            location_id_filter, num_weeks_filter)
        if df_model_subm is not None:
            df_all_submissions = pd.concat([df_all_submissions, df_model_subm], ignore_index=True)

    return df_all_submissions


def process_gt(comp, start_date, end_date, reichlab_path='../../../covid19-forecast-hub', 
               read_from_github=False, location_id_filter=78):
    """Process gt file in reichlab repo. Aggregate by week, and truncate to dates models forecasted for.

    Args:
        comp (str): Which compartment (Can be 'inc_case', 'cum_case', 'inc_death', or 'cum_death')
        df_all_submissions (pd.DataFrame): The dataframe of all model predictions processed. 
        reichlab_path (str, optional): Path to reichlab repo (if cloned on machine). 
        Defaults to '../../../covid19-forecast-hub'.
        read_from_github (bool, optional): If true, reads files directly from github 
        instead of cloned repo. Defaults to False.
        location_id_filter (int, optional): All location ids <= this will be kept. Defaults to 78.

    Returns:
        [pd.DataFrame]*3, dict : gt df, gt df truncated to model prediction dates (daily), 
        gt df truncated to model prediction dates (aggregated weekly), dict of location name to location key
    """
    replace_dict = {'cum': 'Cumulative', 'inc': 'Incident',
                    'case': 'Cases', 'death': 'Deaths', '_': ' '}
    truth_fname = comp
    for key, value in replace_dict.items():
        truth_fname = truth_fname.replace(key, value)

    if read_from_github:
        reichlab_path = 'https://raw.githubusercontent.com/reichlab/covid19-forecast-hub/master'
    df_gt = pd.read_csv(f'{reichlab_path}/data-truth/truth-{truth_fname}.csv')
    df_gt['location'] = df_gt['location'].apply(
        lambda x: int(x) if x != 'US' else 0)
    df_gt = df_gt[df_gt['location'] <= location_id_filter]
    df_gt['date'] = pd.to_datetime(df_gt['date'])

    df_gt_loss = df_gt[(df_gt['date'] > start_date) & (df_gt['date'] <= end_date)]

    if 'inc' in comp:
        df_gt_loss_wk = df_gt_loss.groupby(['location', 'location_name']).resample(
            '7D', label='right', origin='start', on='date').sum()
    else:
        df_gt_loss_wk = df_gt_loss.groupby(['location', 'location_name']).resample(
            '7D', label='right', origin='start', on='date').max()
    df_gt_loss_wk.drop(['location', 'location_name', 'date'],
                       axis=1, inplace=True, errors='ignore')
    df_gt_loss_wk = df_gt_loss_wk.reset_index()
    df_gt_loss_wk['date'] = pd.to_datetime(df_gt_loss_wk['date'])
    df_gt_loss_wk['date'] = df_gt_loss_wk['date'].apply(
        lambda x: x - np.timedelta64(1, 'D'))

    return df_gt, df_gt_loss, df_gt_loss_wk


def compare_gt_pred(df_all_submissions, df_gt_loss_wk, loss_fn='mape'):
    """Function for comparing all predictions to ground truth 

    Args:
        df_all_submissions (pd.DataFrame): dataframe of all predictions processed
        df_gt_loss_wk (pd.DataFrame): dataframe of ground truth numbers aggregated at a week level

    Returns:
        [pd.DataFrame, pd.DataFrame, pd.DataFrame]: combined dataframe, dataframe of mape values, dataframe of ranks
    """
    df_comb = df_all_submissions.merge(df_gt_loss_wk, 
                                       left_on=['target_end_date', 'location'], 
                                       right_on=['date', 'location'])
    df_comb = df_comb.rename({'value_x': 'forecast_value', 
                              'value_y': 'true_value'}, axis=1)

    lc = Loss_Calculator()
    df_comb['mape'] = df_comb.apply(lambda row: lc._calc_mape(
        np.array([row['forecast_value']]), np.array([row['true_value']])), axis=1)
    df_comb['rmse'] = df_comb.apply(lambda row: lc._calc_rmse(
        np.array([row['forecast_value']]), np.array([row['true_value']])), axis=1)
    df_comb['mape_perc'] = df_comb.apply(lambda row: lc._calc_mape_perc(
        np.array([row['forecast_value']]), np.array([row['true_value']]), 
        row['quantile']) if row['type'] == 'quantile' else np.nan, axis=1)
    num_cols = ['mape', 'rmse', 'mape_perc', 'forecast_value']
    df_comb.loc[:, num_cols] = df_comb.loc[:, num_cols].apply(pd.to_numeric)
    df_temp = df_comb[df_comb['type'] == 'point']
    df_mape = df_temp.groupby(['model', 'location',
                               'location_name']).mean().reset_index()
    
    df_mape = df_mape.pivot(index='model', columns='location_name', 
                            values=loss_fn)

    df_rank = df_mape.rank()

    return df_comb, df_mape, df_rank


def _inc_sum_matches_cum_check(df_loc_submission, which_comp):
    """Function for checking if the sum of incident cases matches cumulative

    Args:
        df_loc_submission (pd.DataFrame): The submission df for a particular location
        which_comp (str): The name of the compartment

    Returns:
        bool: Whether of not sum(inc) == cum for all points in given df
    """
    loc = df_loc_submission.iloc[0, :]['location']
    buggy_forecasts = []
    if which_comp is None:
        comps_to_check_for = ['death', 'case']
    else:
        comps_to_check_for = [which_comp]
    for comp in comps_to_check_for:
        df = df_loc_submission.loc[[
            comp in x for x in df_loc_submission['target']], :]
        grouped = df.groupby(['type', 'quantile'])
        for (type, quantile), group in grouped:
            cum_diff = group.loc[['cum' in x for x in group['target']], 'value'].diff()
            inc = group.loc[['inc' in x for x in group['target']], 'value']
            cum_diff = cum_diff.to_numpy()[1:]
            inc = inc.to_numpy()[1:]
            if int(np.sum(np.logical_not((cum_diff - inc) < 1e-8))) != 0:
                print('Sum of inc != cum for {}, {}, {}, {}'.format(
                    loc, comp, type, quantile))
                print(cum_diff, inc)
                buggy_forecasts.append((loc, comp, type, quantile))
    
    return len(buggy_forecasts) == 0


def _qtiles_nondec_check(df_loc_submission):
    """Check if qtiles are non decreasing

    Args:
        df_loc_submission (pd.DataFrame): The submission dataframe for a particular location 

    Returns:
        bool: Whether or not qtiles are non decreasing in given df
    """
    grouped = df_loc_submission[df_loc_submission['type']
                                == 'quantile'].groupby('target')
    nondec_check = [sum(np.diff(group['value']) < 0) > 0 for _, group in grouped]
    nondec_check = np.array(nondec_check)
    return sum(nondec_check) > 0


def _qtiles_nondec_correct(df_loc_submission):
    """If qtiles are not non decreasing, correct them

    Args:
        df_loc_submission (pd.DataFrame): The submission dataframe for a particular location

    Returns:
        pd.DataFrame: Corrected df
    """
    grouped = df_loc_submission[df_loc_submission['type']
                                == 'quantile'].groupby('target')
    for target, group in grouped:
        diff_less_than_0 = np.diff(group['value']) < 0
        if sum(diff_less_than_0) > 0:
            indices = np.where(diff_less_than_0 == True)[0]
            for idx in indices:
                df_idx1, df_idx2 = (group.iloc[idx, :].name, 
                                    group.iloc[idx+1, :].name)
                df_loc_submission.loc[df_idx2, 'value'] = df_loc_submission.loc[df_idx1, 'value']

    return df_loc_submission

def format_wiai_submission(predictions_dict, loc_name_to_key_dict, formatting_mode='analysis', which_fit='m2',
                           use_as_point_forecast='ensemble_mean', which_comp=None, skip_percentiles=False):
    """Function for formatting our submission in the reichlab format 

    Args:
        predictions_dict (dict): Predictions dict of all locations
        loc_name_to_key_dict (dict): Dict mapping location names to location key
        which_fit (str, optional): Which fit to use for forecasting ('m1'/'m2'). Defaults to 'm2'.
        use_as_point_forecast (str, optional): Which forecast to use as point forecast ('best'/'ensemble_mean'). 
        Defaults to 'ensemble_mean'.
        skip_percentiles (bool, optional): If true, processing of all percentiles is skipped. Defaults to False.

    Returns:
        pd.DataFrame: Processed Wadhwani AI submission
    """
    end_date = list(predictions_dict.values())[0]['m2']['run_params']['split']['end_date']
    columns = ['forecast_date', 'target', 'target_end_date', 'location', 'type',
               'quantile', 'value', 'model']
    df_wiai_submission = pd.DataFrame(columns=columns)

    # Loop across all locations
    for loc in predictions_dict.keys():
        df_loc_submission = pd.DataFrame(columns=columns)

        # Loop across all percentiles
        if not which_fit in predictions_dict[loc].keys():
            continue
        for percentile in predictions_dict[loc][which_fit]['forecasts'].keys():
            if isinstance(percentile, str):
                # Skipping all point forecasts that are not what the user specified
                if not percentile == use_as_point_forecast:
                    continue
            # Skipping all percentiles if the flag is True
            if skip_percentiles:
                if (isinstance(percentile, int)) or (isinstance(percentile, float)):
                    continue
            # Loop across cumulative and deceased
            for mode in ['cum', 'inc']:
                df_forecast = copy.deepcopy(predictions_dict[loc][which_fit]['forecasts'][percentile])
                for comp in ['deceased', 'total']:
                    dec_indices = np.where(np.diff(df_forecast[comp]) < 0)[0]
                    if len(dec_indices) > 0:
                        for idx in dec_indices:
                            df_forecast.loc[idx+1, comp] = df_forecast.loc[idx, comp]

                # Take diff for the forecasts (by default forecasts are cumulative)
                if mode == 'inc':
                    num_cols = df_forecast.select_dtypes(
                        include=['int64', 'float64']).columns
                    df_forecast.loc[:, num_cols] = df_forecast.loc[:, num_cols].diff()
                    df_forecast.dropna(axis=0, how='any', inplace=True)

                # Truncate forecasts df to only beyond the training date
                df_forecast = df_forecast[df_forecast['date'].dt.date > end_date]
                # Aggregate the forecasts by a week (def of week : Sun-Sat)
                if mode == 'cum':
                    df_forecast = df_forecast.resample(
                        'W-Sat', label='right', origin='start', on='date').max()
                if mode == 'inc':
                    df_forecast = df_forecast.resample(
                        'W-Sat', label='right', origin='start', on='date').sum()
                df_forecast['date'] = df_forecast.index
                if formatting_mode == 'submission':
                    now = datetime.now(timezone('US/Eastern'))
                    df_forecast = df_forecast[df_forecast['date'].dt.date > now.date()]
                    if not ((now.strftime("%A") == 'Sunday') or (now.strftime("%A") == 'Monday')):
                        df_forecast = df_forecast.iloc[1:, :]
                    df_forecast = df_forecast.iloc[:-1, :]
                else:
                    now = datetime.now()
                # Only keep those forecasts that correspond to the forecasts others submitted
                df_forecast.drop(['date'], axis=1, inplace=True, errors='ignore')
                df_forecast.reset_index(inplace=True)

                # Create forecasts for both deceased and total cases
                df_subm_d = copy.copy(df_forecast.loc[:, ['date', 'deceased']])
                df_subm_d['target'] = pd.Series(
                    list(map(lambda x: f'{x+1} wk ahead {mode} death', np.arange(len(df_subm_d)))))
                df_subm_t = copy.copy(df_forecast.loc[:, ['date', 'total']])
                df_subm_t['target'] = pd.Series(
                    list(map(lambda x: f'{x+1} wk ahead {mode} case', np.arange(len(df_subm_t)))))
                # Rename Columns
                df_subm_d.rename({'date': 'target_end_date',
                                'deceased': 'value'}, axis=1, inplace=True)
                df_subm_t.rename({'date': 'target_end_date',
                                'total': 'value'}, axis=1, inplace=True)

                # Create the type quantile  columns for all forecasts
                if which_comp is not None:
                    if which_comp == 'death':
                        df_subm = df_subm_d
                    elif which_comp == 'case':
                        df_subm = df_subm_t
                    else:
                        raise ValueError('Incorrect option of which_comp given. ' + \
                            'which_comp can be either death/case')
                else:
                    df_subm = pd.concat([df_subm_d, df_subm_t], ignore_index=True)
                if percentile == use_as_point_forecast:
                    df_subm['type'] = 'point'
                    df_subm['quantile'] = np.nan
                else:
                    df_subm['type'] = 'quantile'
                    df_subm['quantile'] = percentile/100
                if formatting_mode == 'submission':
                    df_subm['location'] = loc_name_to_key_dict[loc]
                else:
                    df_subm['location'] = int(loc_name_to_key_dict[loc])
                df_subm['model'] = 'Wadhwani_AI-'
                df_subm['forecast_date'] = datetime.combine(now.date(),
                                                            datetime.min.time())
                df_loc_submission = pd.concat([df_loc_submission, df_subm], 
                                              ignore_index=True)
        if formatting_mode == 'submission':
            if not _inc_sum_matches_cum_check(df_loc_submission, which_comp):
                raise AssertionError('Sum of inc != cum for some forecasts')
            while(_qtiles_nondec_check(df_loc_submission)):
                df_loc_submission = _qtiles_nondec_correct(df_loc_submission)
                
        df_wiai_submission = pd.concat([df_wiai_submission, df_loc_submission],
                                        ignore_index=True)
        print(f'{loc} done')

    return df_wiai_submission


def combine_wiai_subm_with_all(df_all_submissions, df_wiai_submission, comp):
    """Function for combining WIAI submission with all model submissions.

    Args:
        df_all_submissions (pd.DataFrame): Processed df of all model submissions 
        df_wiai_submission (pd.DataFrame): Processed WIAI submission
        comp (str): Which compartment (Can be 'inc_case', 'cum_case', 'inc_death', or 'cum_death')

    Returns:
        pd.DataFrame: Combined DataFrame
    """
    df_all_submissions = df_all_submissions[df_all_submissions['target'].apply(
        lambda x: comp.replace('_', ' ') in x)]

    df_wiai_submission = df_wiai_submission[df_wiai_submission['target'].apply(
        lambda x: comp.replace('_', ' ') in x)]

    target_end_dates = pd.unique(df_all_submissions['target_end_date'])
    df_wiai_submission = df_wiai_submission[df_wiai_submission['target_end_date'].isin(
        target_end_dates)]

    df_all =  pd.concat([df_all_submissions, df_wiai_submission], 
                        ignore_index=True)
    return df_all

def calculate_z_score(df_mape, df_rank, model_name='Wadhwani_AI-BayesOpt'):
    """Function for calculating Z score and non param Z score

    Args:
        df_mape (pd.DataFrame): dataframes of mape values for all models, locations
        df_rank (pd.DataFrame): dataframes of ranks values for all models, locations
        model_name (str, optional): Which model to calculate Z scores for. Defaults to 'Wadhwani_AI-BayesOpt'.

    Returns:
        pd.DataFrame: dataframe with the calculated Z scores
    """

    df = pd.concat([df_mape.mean(axis=0), df_mape.std(axis=0), 
                    df_mape.median(axis=0), df_mape.mad(axis=0),
                    df_mape.loc[model_name, :], df_rank.loc[model_name, :]], axis=1)
    df.columns = ['mean_mape', 'std_mape', 'median_mape',
                  'mad_mape', 'model_mape', 'model_rank']
    df['z_score'] = (df['model_mape'] - df['mean_mape'])/(df['std_mape'])
    df['non_param_z_score'] = (df['model_mape'] - df['median_mape'])/df['mad_mape']
    return df


def combine_with_train_data(predictions_dict, df):
    """Combine the Z score dataframe with the train error data from (read from predictions_dict file)

    Args:
        predictions_dict (dict): The predictions_dict output file
        df (pd.DataFrame): Z Score dataframe

    Returns:
        pd.DataFrame: df with the z scores and train error
    """
    df_wadhwani = pd.DataFrame(index=list(predictions_dict.keys()),
                               columns=['best_loss_train', 'test_loss',
                                        'T_recov_fatal', 'P_fatal'])
    for loc in predictions_dict.keys():
        df_wadhwani.loc[loc, 'best_loss_train'] = predictions_dict[loc]['m2']['df_loss'].to_numpy()[
            0][0]
        df_wadhwani.loc[loc,
                        'T_recov_fatal'] = predictions_dict[loc]['m2']['best_params']['T_recov_fatal']
        df_wadhwani.loc[loc,
                        'P_fatal'] = predictions_dict[loc]['m2']['best_params']['P_fatal']

    df_wadhwani = df_wadhwani.merge(df, left_index=True, right_index=True)

    df_wadhwani.drop(['Northern Mariana Islands', 'Guam',
                      'Virgin Islands'], axis=0, inplace=True, errors='ignore')

    return df_wadhwani

def create_performance_table(df_mape, df_rank):
    """Creates dataframe of all models sorted by MAPE value and rank, to compare relative performance 

    Args:
        df_mape (pd.DataFrame): df of MAPE values for all models, regions
        df_rank (pd.DataFrame): df of ranks for all models, regions

    Returns:
        pd.DataFrame: The performance table
    """
    median_mape = df_mape.loc[:, np.logical_not(
        df_mape.loc['Wadhwani_AI-BayesOpt', :].isna())].median(axis=1).rename('median_mape')
    median_rank = df_rank.loc[:, np.logical_not(
        df_rank.loc['Wadhwani_AI-BayesOpt', :].isna())].median(axis=1).rename('median_rank')
    merged = pd.concat([median_mape, median_rank], axis=1)
    merged.reset_index(inplace=True)
    merged['model1'] = merged['model']
    merged = merged[['model', 'median_mape', 'model1', 'median_rank']]
    merged = merged.sort_values('median_mape')
    merged.reset_index(drop=True, inplace=True)
    temp = copy.copy(merged.loc[:, ['model1', 'median_rank']])
    temp.sort_values('median_rank', inplace=True)
    temp.reset_index(drop=True, inplace=True)
    merged.loc[:, ['model1', 'median_rank']
            ] = temp.loc[:, ['model1', 'median_rank']]
    merged.index = merged.index + 1
    return merged


def end_to_end_comparison(hparam_source='predictions_dict', predictions_dict=None, config_filename=None, 
                          comp=None, date_of_submission=None, process_wiai_submission=False, 
                          which_fit='m2', use_as_point_forecast='ensemble_mean', drop_territories=True,
                          num_submissions_filter=45, location_id_filter=78):
    if hparam_source != 'input':
        if hparam_source == 'predictions_dict':
            config = predictions_dict[list(predictions_dict.keys())[0]]['m2']['run_params']
        elif hparam_source == 'config':
            config = read_config(config_filename)['fitting']
        else:
            raise ValueError('Please give correct input of hparam_source')

        loss_comp = config['loss']['loss_compartments'][0]
        data_last_date = config['split']['end_date']

        if date_of_submission is None:
            date_of_submission = (data_last_date + timedelta(days=2)).strftime('%Y-%m-%d')
        if comp is None:
            if loss_comp == 'deceased':
                comp = 'cum_death'
            if loss_comp == 'total':
                comp = 'cum_case'
    else:
        if (comp is None) or (date_of_submission is None):
            raise ValueError('comp and date_of_submission should not be None if hparam_source==input')
    
    print(comp, date_of_submission)

    list_of_models = get_list_of_models(date_of_submission, comp,
                                        num_submissions_filter=num_submissions_filter)
    df_all_submissions = process_all_submissions(list_of_models, date_of_submission, comp)
    target_end_dates = pd.unique(df_all_submissions['target_end_date'])
    start_date = target_end_dates[0] - np.timedelta64(7, 'D')
    end_date = target_end_dates[-1]
    df_gt, df_gt_loss, df_gt_loss_wk = process_gt(comp, start_date, end_date)

    if process_wiai_submission:
        if predictions_dict is None:
            raise ValueError('Please give predictions_dict file if process_wiai_submission = True')
        loc_name_to_key_dict = get_mapping(which='location_name_to_code')
        df_wiai_submission = format_wiai_submission(predictions_dict, loc_name_to_key_dict, 
                                                    use_as_point_forecast=use_as_point_forecast, 
                                                    which_fit=which_fit, skip_percentiles=False)

        df_all_submissions = combine_wiai_subm_with_all(
            df_all_submissions, df_wiai_submission, comp)

    df_comb, df_mape, df_rank = compare_gt_pred(df_all_submissions, df_gt_loss_wk)
    if drop_territories:
        list_of_territories = ['Guam', 'Virgin Islands', 'American Samoa',
                               'Puerto Rico', 'Northern Mariana Islands']
        df_comb = df_comb[np.logical_not(
            df_comb['location_name'].isin(list_of_territories))]
        df_mape.drop(list_of_territories, axis=1, inplace=True)
        df_rank.drop(list_of_territories, axis=1, inplace=True)

    num_models = len(df_mape.median(axis=1))
    print(f'Total # of models - {num_models}')

    return df_comb, df_mape, df_rank
