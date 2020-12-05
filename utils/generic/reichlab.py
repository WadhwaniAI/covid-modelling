import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from pandas.core import groupby
from pytz import timezone
import copy
import re

from data.dataloader import JHULoader
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
    df = pd.read_csv(f'{reichlab_path}/ensemble-metadata/' + \
        f'{date_of_submission}-{comp}-model-eligibility.csv')
    df['location'] = df['location'].apply(lambda x : int(x) if x != 'US' else 0)

    df_all_states = df[df['location'] <= location_id_filter]
    df_eligible = df_all_states[df_all_states['overall_eligibility'] == 'eligible']

    df_counts = df_eligible.groupby('model').count()
    
    # Filter all models with > 50 submissions
    df_counts = df_counts[df_counts['overall_eligibility'] > num_submissions_filter]
    list_of_models = df_counts.index
    return list_of_models


def process_single_submission(model, date_of_submission, comp, df_true, reichlab_path='../../../covid19-forecast-hub', 
                              read_from_github=False):
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

    Returns:
        pd.DataFrame: model submssion processed dataframe
    """
    if read_from_github:
        reichlab_path = 'https://raw.githubusercontent.com/reichlab/covid19-forecast-hub/master'
    try:
        df = pd.read_csv(f'{reichlab_path}/data-processed/' + \
            f'{model}/{date_of_submission}-{model}.csv')
    except Exception:
        date_convert = datetime.strptime(date_of_submission, '%Y-%m-%d')
        date_of_filename = date_convert - timedelta(days=1)
        df = pd.read_csv(f'{reichlab_path}/data-processed/' + \
            f'{model}/{date_of_filename.strftime("%Y-%m-%d")}-{model}.csv')
    # Converting all locations to integers
    df['location'] = df['location'].apply(lambda x : int(x) if x != 'US' else 0)
    # Keeping only states and territories forecasts
    df = df[df['location'] <= 78]
    df['model'] = model
    
    # Only keeping the wk forecasts
    df = df[df['target'].apply(lambda x : 'wk' in x)]
    
    # Only the forecasts corresponding the comp user are interested in
    if comp == 'cum_case':
        df = df[df['target'].apply(lambda x : 'inc_case'.replace('_', ' ') in x)]
    else:
        df = df[df['target'].apply(lambda x : comp.replace('_', ' ') in x)]
    
    # Pruning the forecasts which are beyond 4 weeks ahead
    df = df[df['target'].apply(lambda x : int(re.findall(r'\d+', x)[0])) <= 4]
    
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
                            read_from_github=False):
    """Process submissions for all models given as input and concatenate them

    Args:
        list_of_models (list): List of all models to process submission for. Typically output of get_list_of_models
        date_of_submission (str): The ensemble creation date (always a Mon), for selecting a particular week
        comp (str): Which compartment (Can be 'inc_case', 'cum_case', 'inc_death', or 'cum_death')
        reichlab_path (str, optional): Path to reichlab repo (if cloned on machine). 
        Defaults to '../../../covid19-forecast-hub'.
        read_from_github (bool, optional): If true, reads files directly from github 
        instead of cloned repo. Defaults to False.

    Returns:
        [type]: [description]
    """
    dataframes = get_dataframes_cached(loader_class=JHULoader)
    df_true = dataframes['df_us_states']
    df_all_submissions = process_single_submission(
        list_of_models[0], date_of_submission, comp, df_true, reichlab_path, read_from_github)
    for model in list_of_models:
        df_model_subm = process_single_submission(
            model, date_of_submission, comp, df_true, reichlab_path, read_from_github)
        df_all_submissions = pd.concat([df_all_submissions, df_model_subm], ignore_index=True)

    return df_all_submissions


def process_gt(comp, df_all_submissions, reichlab_path='../', read_from_github=False):
    """Process gt file in reichlab repo. Aggregate by week, and truncate to dates models forecasted for.

    Args:
        comp (str): Which compartment (Can be 'inc_case', 'cum_case', 'inc_death', or 'cum_death')
        df_all_submissions (pd.DataFrame): The dataframe of all model predictions processed. 
        reichlab_path (str, optional): Path to reichlab repo (if cloned on machine). 
        Defaults to '../../../covid19-forecast-hub'.
        read_from_github (bool, optional): If true, reads files directly from github 
        instead of cloned repo. Defaults to False.

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
    df_gt = df_gt[df_gt['location'] <= 78]
    df_gt['date'] = pd.to_datetime(df_gt['date'])

    target_end_dates = pd.unique(df_all_submissions['target_end_date'])
    df_gt_loss = df_gt[(df_gt['date'] > target_end_dates[0] -
                        np.timedelta64(7, 'D')) & (df_gt['date'] <= target_end_dates[-1])]

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


def compare_gt_pred(df_all_submissions, df_gt_loss_wk):
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

    df_comb['mape'] = np.abs(df_comb['forecast_value'] - df_comb['true_value'])*100/(df_comb['true_value']+1e-8)
    num_cols = ['mape', 'forecast_value']
    df_comb.loc[:, num_cols] = df_comb.loc[:, num_cols].apply(pd.to_numeric)
    df_temp = df_comb[df_comb['type'] == 'point']
    df_mape = df_temp.groupby(['model', 'location',
                               'location_name']).mean().reset_index()
    
    df_mape = df_mape.pivot(index='model', columns='location_name', 
                            values='mape')

    df_rank = df_mape.rank()

    return df_comb, df_mape, df_rank


def _qtiles_nondec_check(df_loc_submission):
    grouped = df_loc_submission[df_loc_submission['type']
                                == 'quantile'].groupby('target')
    nondec_check = [sum(np.diff(group['value']) < 0) > 0 for _, group in grouped]
    nondec_check = np.array(nondec_check)
    return sum(nondec_check) > 0


def _qtiles_nondec_correct(df_loc_submission):
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
                           use_as_point_forecast='ensemble_mean', skip_percentiles=False):
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

                # Take diff for the forecasts (by default forecasts are cumulative)
                if mode == 'inc':
                    num_cols = df_forecast.select_dtypes(
                        include=['int64', 'float64']).columns
                    df_forecast.loc[:, num_cols] = df_forecast.loc[:, num_cols].diff()
                    df_forecast.dropna(axis=0, how='any', inplace=True)

                # Truncate forecasts df to only beyond the training date
                df_forecast = df_forecast[df_forecast['date'].dt.date > end_date]
                # Aggregate the forecasts by a week (def of week : Sun-Sat)
                df_forecast = df_forecast.resample(
                    'W-Sat', label='right', origin='start', on='date').max()
                if formatting_mode == 'submission':
                    now = datetime.now(timezone('US/Eastern'))
                    if not ((now.strftime("%A") == 'Sunday') or (now.strftime("%A") == 'Monday')):
                        df_forecast = df_forecast.iloc[1:, :]
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
                df_subm = pd.concat([df_subm_d, df_subm_t], ignore_index=True)
                if isinstance(percentile, str):
                    df_subm['type'] = 'point'
                    df_subm['quantile'] = np.nan
                else:
                    df_subm['type'] = 'quantile'
                    df_subm['quantile'] = percentile/100
                if formatting_mode == 'submission':
                    df_subm['location'] = loc_name_to_key_dict[loc]
                else:
                    df_subm['location'] = int(loc_name_to_key_dict[loc])
                df_subm['model'] = 'Wadhwani_AI'
                df_subm['forecast_date'] = datetime.combine(now.date(),
                                                            datetime.min.time())
                df_loc_submission = pd.concat([df_loc_submission, df_subm], 
                                            ignore_index=True)
        if formatting_mode == 'submission':
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
