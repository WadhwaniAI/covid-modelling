import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import copy
import re

"""
Helper functions for processing different reichlab submissions, processing reichlab ground truth,
Comparing reichlab models with gt, processing and formatting our (Wadhwani AI) submission, 
comparing that with gt as well
"""

def get_list_of_models(date_of_submission, comp, reichlab_path='..', read_from_github=False):
    if read_from_github:
        reichlab_path = 'https://raw.githubusercontent.com/reichlab/covid19-forecast-hub/master'
    df = pd.read_csv(f'{reichlab_path}/ensemble-metadata/' + \
        f'{date_of_submission}-{comp}-model-eligibility.csv')
    df['location'] = df['location'].apply(lambda x : int(x) if x != 'US' else 0)

    df_all_states = df[df['location'] <= 78]
    df_eligible = df_all_states[df_all_states['overall_eligibility'] == 'eligible']

    df_counts = df_eligible.groupby('model').count()
    
    # Filter all models with > 50 submissions
    df_counts = df_counts[df_counts['overall_eligibility'] > 50]
    list_of_models = df_counts.index
    return list_of_models

def process_single_submission(model, date_of_submission, comp, reichlab_path='..', read_from_github=False):
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
    
    # Only the forecasts corresponnding the comp user are interested in
    df = df[df['target'].apply(lambda x : comp.replace('_', ' ') in x)]
    
    # Pruning the forecasts which are beyond 4 weeks ahead
    df = df[df['target'].apply(lambda x : int(re.findall(r'\d+', x)[0])) <= 4]
    
    df['target_end_date'] = pd.to_datetime(df['target_end_date'])
    df['forecast_date'] = pd.to_datetime(df['forecast_date'])
    
    return df


def process_all_submissions(list_of_models, date_of_submission, comp, reichlab_path='..', read_from_github=False):
    df_all_submissions = process_single_submission(
        list_of_models[0], date_of_submission, comp, reichlab_path, read_from_github)
    for model in list_of_models:
        df_model_subm = process_single_submission(
            model, date_of_submission, comp, reichlab_path, read_from_github)
        df_all_submissions = pd.concat([df_all_submissions, df_model_subm], ignore_index=True)

    return df_all_submissions


def process_gt(comp, df_all_submissions, reichlab_path='../', read_from_github=False):
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

    loc_name_to_key_dict = dict(zip(df_gt_loss['location_name'], 
                                    df_gt_loss['location']))

    return df_gt, df_gt_loss, df_gt_loss_wk, loc_name_to_key_dict


def compare_gt_pred(df_all_submissions, df_gt_loss_wk):
    df_comb = df_all_submissions.merge(df_gt_loss_wk, 
                                       left_on=['target_end_date', 'location'], 
                                       right_on=['date', 'location'])
    df_comb = df_comb.rename({'value_x': 'forecast_value', 
                              'value_y': 'true_value'}, axis=1)

    df_comb = df_comb[df_comb['type'] == 'point']
    df_comb['p_error'] = np.abs(df_comb['forecast_value'] - df_comb['true_value'])*100/(df_comb['true_value']+1e-8)
    num_cols = ['p_error', 'forecast_value']
    df_comb.loc[:, num_cols] = df_comb.loc[:, num_cols].apply(pd.to_numeric)
    df_mape = df_comb.groupby(['model', 'location', 
                               'location_name']).mean().reset_index()
    
    df_mape = df_mape.pivot(index='model', columns='location_name', 
                            values='p_error')

    df_rank = df_mape.rank()

    return df_comb, df_mape, df_rank


def format_wiai_submission(predictions_dict, df_all_submissions, loc_name_to_key_dict, which_fit='m2', 
                           use_as_point_forecast='ensemble_mean', skip_percentiles=False):
    df_wiai_submission = pd.DataFrame(columns=df_all_submissions.columns)
    target_end_dates = pd.unique(df_all_submissions['target_end_date'])

    # Loop across all locations
    for loc in predictions_dict.keys():
        df_loc_submission = pd.DataFrame(columns=df_all_submissions.columns)

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

                # Aggregate the forecasts by a week (def of week : Sun-Sat)
                df_forecast = df_forecast.resample(
                    'W-Sat', label='right', origin='start', on='date').max()
                # Only keep those forecasts that correspond to the forecasts others submitted
                df_forecast = df_forecast[df_forecast.index.isin(target_end_dates)]
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
                df_subm['location'] = loc_name_to_key_dict[loc]
                df_subm['model'] = 'Wadhwani_AI'
                df_subm['forecast_date'] = datetime.combine(date.today(), 
                                                            datetime.min.time())
                df_loc_submission = pd.concat([df_loc_submission, df_subm], 
                                            ignore_index=True)
        df_wiai_submission = pd.concat([df_wiai_submission, df_loc_submission],
                                        ignore_index=True)
        print(f'{loc} done')

    return df_wiai_submission


def combine_wiai_subm_with_all(df_all_submissions, df_wiai_submission, comp):
    df_all_submissions = df_all_submissions[df_all_submissions['target'].apply(
        lambda x: comp.replace('_', ' ') in x)]

    df_wiai_submission = df_wiai_submission[df_wiai_submission['target'].apply(
        lambda x: comp.replace('_', ' ') in x)]

    df_all =  pd.concat([df_all_submissions, df_wiai_submission], 
                        ignore_index=True)
    return df_all
