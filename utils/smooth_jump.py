import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import copy

from data.processing import processing


def smooth_big_jump_helper(df_district, smoothing_var, auxillary_var, d1, d2=None, smoothing_length=None, 
                           method='uniform', t_recov=14, aux_var_add=False):
    """Helper function for performing smoothing of big jumps

    Args:
        df_district (pd.DataFrame): The input dataframe
        smoothing_length (int): The time window to smooth over
        d1 (str): Date just before smoothing YYYY-MM-DD
        d2 (str): Date on which smoothing happens YYYY-MM-DD
        smoothing_var (str): Variable to smooth over
        auxillary_var (str): Corresponding auxillary variable that also gets smoothed 
        t_recov (int, optional): The assumed recovery time. For weigthed method. Defaults to 14.
        method (str, optional): [description]. Defaults to 'uniform'.

    Raises:
        Exception: [description]

    Returns:
        [type]: [description]
    """
    df_district['date'] = pd.to_datetime(df_district['date'])
    df_district = df_district.set_index('date')
    if d2 == None:
        d2 = datetime.strptime(d1, '%Y-%m-%d') + timedelta(days=1)
    big_jump = df_district.loc[d2, smoothing_var] - df_district.loc[d1, smoothing_var]
    aux_var_weight = (int(aux_var_add) - 0.5)*2

    print(big_jump)
    if method == 'uniform':
        for i, day_number in enumerate(range(smoothing_length-2, -1, -1)):
            date = datetime.strptime(d1, '%Y-%m-%d') - timedelta(days=day_number)
            offset = np.random.binomial(1, (big_jump%smoothing_length)/smoothing_length)
            df_district.loc[date, smoothing_var] += ((i+1)*big_jump)//smoothing_length + offset
            df_district.loc[date, auxillary_var] -= aux_var_weight*(((i+1)*big_jump)//smoothing_length + offset)

    elif 'weighted' in method:
        if method == 'weighted-recov':
            newcases = df_district['total_infected'].shift(t_recov) - df_district['total_infected'].shift(t_recov + 1)
        elif method == 'weighted-diff' or method == 'weighted':
            newcases = df_district.loc[:, smoothing_var].shift(0) - df_district.loc[:, smoothing_var].shift(1)
        elif method == 'weighted-mag':
            newcases = df_district.loc[:, smoothing_var]
        else:
            raise Exception("unknown smoothing method provided")

        valid_idx = newcases.first_valid_index()
        if smoothing_length != None:
            window_start = datetime.strptime(d1, '%Y-%m-%d') - timedelta(days=smoothing_length - 1)
        else:
            window_start = datetime.strptime(d1, '%Y-%m-%d') - timedelta(days=365)
        newcases = newcases.loc[max(valid_idx, window_start):d1]
        truncated = df_district.loc[max(valid_idx, window_start):d1, :]
        smoothing_length = len(truncated)
        print(f'smoothing length truncated to {smoothing_length}')
        invpercent = newcases.sum()/newcases
        for day_number in range(smoothing_length-1, -1, -1):
            date = datetime.strptime(d1, '%Y-%m-%d') - timedelta(days=day_number)
            offset = np.random.binomial(1, (big_jump%invpercent.loc[date])/invpercent.loc[date])
            truncated.loc[date:, smoothing_var] += (big_jump // invpercent.loc[date]) + offset
            truncated.loc[date:, auxillary_var] += aux_var_weight*((big_jump // invpercent.loc[date]) + offset)
        df_district.loc[truncated.index, smoothing_var] = truncated[smoothing_var].astype('int64')
        df_district.loc[truncated.index, auxillary_var] = truncated[auxillary_var].astype('int64')

    else:
        raise Exception("unknown smoothing method provided")
    
    return df_district.reset_index()


def smooth_big_jump(df_district, data_from_tracker=False):
    if data_from_tracker:
        d1 = '2020-05-29'
        length = (datetime.strptime(d1, '%Y-%m-%d') - df_district.loc[0, 'date']).days
        df_district = smooth_big_jump_helper(
            df_district, 'recovered', 'hospitalised', d1, smoothing_length=length, method='weighted-mag')

        d1 = '2020-06-15'
        length = (datetime.strptime(d1, '%Y-%m-%d') - df_district.loc[0, 'date']).days
        df_district = smooth_big_jump_helper(
            df_district, 'recovered', 'hospitalised', d1, smoothing_length=length, method='weighted-mag')

        d1 = '2020-06-16'
        length = (datetime.strptime(d1, '%Y-%m-%d') - df_district.loc[0, 'date']).days
        df_district = smooth_big_jump_helper(
            df_district, 'recovered', 'hospitalised', d1, smoothing_length=length, method='weighted-mag')

        d1 = '2020-06-23'
        length = (datetime.strptime(d1, '%Y-%m-%d') - datetime.strptime('2020-06-15', '%Y-%m-%d')).days
        df_district = smooth_big_jump_helper(
            df_district, 'recovered', 'hospitalised', d1, smoothing_length=length, method='weighted-mag')

        d1 = '2020-06-24'
        length = (datetime.strptime(d1, '%Y-%m-%d') - datetime.strptime('2020-06-15', '%Y-%m-%d')).days
        df_district = smooth_big_jump_helper(
            df_district, 'recovered', 'hospitalised', d1, smoothing_length=length, method='weighted-mag')

        d1 = '2020-06-24'
        length = (datetime.strptime(d1, '%Y-%m-%d') - datetime.strptime('2020-06-15', '%Y-%m-%d')).days
        df_district = smooth_big_jump_helper(
            df_district, 'recovered', 'hospitalised', d1, smoothing_length=length, method='weighted-mag')

        # d1 = '2020-06-27'
        # length = (datetime.strptime(d1, '%Y-%m-%d') - datetime.strptime('2020-06-15', '%Y-%m-%d')).days
        # df_district = smooth_big_jump_helper(
        #     df_district, 'recovered', 'hospitalised', d1, smoothing_length=length, method='weighted-mag')

        d1 = '2020-07-01'
        length = (datetime.strptime(d1, '%Y-%m-%d') - datetime.strptime('2020-05-29', '%Y-%m-%d')).days
        df_district = smooth_big_jump_helper(
            df_district, 'recovered', 'hospitalised', d1, smoothing_length=length, method='weighted-mag')

        d1 = '2020-06-16'
        length = (datetime.strptime(d1, '%Y-%m-%d') - df_district.loc[0, 'date']).days
        df_district = smooth_big_jump_helper(
            df_district, 'deceased', 'total_infected', d1, smoothing_length=length, method='weighted-mag', 
            aux_var_add=True)

    else:
        d1 = '2020-05-28'
        df_district = smooth_big_jump_helper(
            df_district, 'recovered', 'hospitalised', d1, smoothing_length=33, method=method)
        
        d1 = '2020-06-14'
        length = (datetime.strptime(d1, '%Y-%m-%d') - datetime.strptime('2020-05-28', '%Y-%m-%d')).days
        df_district = smooth_big_jump_helper(
            df_district, 'recovered', 'hospitalised', d1, smoothing_length=length, method=method)

        d1 = '2020-06-15'
        length = (datetime.strptime(d1, '%Y-%m-%d') - datetime.strptime('2020-05-28', '%Y-%m-%d')).days
        df_district = smooth_big_jump_helper(
            df_district, 'hospitalised', 'recovered', d1, smoothing_length=length, method=method)

        d1 = '2020-06-15'
        length = (datetime.strptime(d1, '%Y-%m-%d') - df_district.loc[0, 'date']).days
        df_district = smooth_big_jump_helper(
            df_district, 'deceased', 'total_infected', d1, smoothing_length=length, method=method, aux_var_add=True)
    
    # assert((df_district['total_infected'] == df_district['recovered'] \
    #     + df_district['deceased'] + df_district['hospitalised']).all())
    print((df_district['total_infected'] == df_district['recovered'] \
        + df_district['deceased'] + df_district['hospitalised']))

    return df_district


def smooth_big_jump_stratified(df_strat):
    filename = '../../data/data/mumbai/case_summary_2006.csv'
    df = processing.get_data(filename=filename, disable_tracker=True)
    df_smoothed = smooth_big_jump(df)

    df_strat_smoothed = copy.copy(df_strat)
    diff_array = df_smoothed.loc[df_smoothed['date'].isin(
        df_strat['date']), 'hospitalised'].reset_index(drop=True) - df_strat['hospitalised']

    for column in df_strat_smoothed.loc[:, 'stable_asymptomatic':].columns:
        df_strat_smoothed[column] += diff_array * \
            (df_strat_smoothed[column]/df_strat_smoothed['hospitalised'])

    base_columns = ['total_infected', 'hospitalised', 'recovered', 'deceased']
    df_strat_smoothed.loc[:, base_columns] = df_smoothed.loc[df_smoothed['date'].isin(
        df_strat['date']), base_columns].reset_index(drop=True)

    d1 = '2020-06-10'
    length = (datetime.strptime(d1, '%Y-%m-%d') - df_strat_smoothed.loc[0, 'date']).days
    df_strat_smoothed = smooth_big_jump_helper(
        df_strat_smoothed, 'stable_symptomatic', 'stable_asymptomatic', d1, smoothing_length=length, method='weighted')

    d1 = '2020-06-15'
    length = (datetime.strptime(d1, '%Y-%m-%d') - df_strat_smoothed.loc[0, 'date']).days
    df_strat_smoothed = smooth_big_jump_helper(
        df_strat_smoothed, 'stable_asymptomatic', 'stable_symptomatic', d1, smoothing_length=length, method='weighted')

    d1 = '2020-06-15'
    length = (datetime.strptime(d1, '%Y-%m-%d') - df_strat_smoothed.loc[0, 'date']).days
    df_strat_smoothed = smooth_big_jump_helper(
        df_strat_smoothed, 'ventilator', 'icu', d1, smoothing_length=length, method='weighted-mag')

    d1 = '2020-06-15'
    length = (datetime.strptime(d1, '%Y-%m-%d') - df_strat_smoothed.loc[0, 'date']).days
    df_strat_smoothed = smooth_big_jump_helper(
        df_strat_smoothed, 'icu', 'o2_beds', d1, smoothing_length=length, method='weighted-mag')

    d1 = '2020-05-31'
    length = (datetime.strptime(d1, '%Y-%m-%d') - df_strat_smoothed.loc[0, 'date']).days
    df_strat_smoothed = smooth_big_jump_helper(
        df_strat_smoothed, 'icu', 'o2_beds', d1, smoothing_length=length, method='weighted-mag')

    d1 = '2020-06-08'
    length = (datetime.strptime(d1, '%Y-%m-%d') - df_strat_smoothed.loc[0, 'date']).days
    df_strat_smoothed = smooth_big_jump_helper(
        df_strat_smoothed, 'o2_beds', 'non_o2_beds', d1, smoothing_length=length, method='weighted-mag')

    d1 = '2020-06-14'
    length = (datetime.strptime(d1, '%Y-%m-%d') - df_strat_smoothed.loc[0, 'date']).days
    df_strat_smoothed = smooth_big_jump_helper(
        df_strat_smoothed, 'o2_beds', 'non_o2_beds', d1, smoothing_length=length, method='weighted-mag')

    d1 = '2020-06-15'
    length = (datetime.strptime(d1, '%Y-%m-%d') - df_strat_smoothed.loc[0, 'date']).days
    df_strat_smoothed = smooth_big_jump_helper(
        df_strat_smoothed, 'o2_beds', 'non_o2_beds', d1, smoothing_length=length, method='weighted-mag')

    d1 = '2020-05-28'
    length = (datetime.strptime(d1, '%Y-%m-%d') - df_strat_smoothed.loc[0, 'date']).days
    df_strat_smoothed = smooth_big_jump_helper(
        df_strat_smoothed, 'non_o2_beds', 'hq', d1, smoothing_length=length, method='weighted-mag')

    d1 = '2020-05-31'
    length = (datetime.strptime(d1, '%Y-%m-%d') - df_strat_smoothed.loc[0, 'date']).days
    df_strat_smoothed = smooth_big_jump_helper(
        df_strat_smoothed, 'non_o2_beds', 'hq', d1, smoothing_length=length, method='weighted-mag')

    d1 = '2020-06-14'
    length = (datetime.strptime(d1, '%Y-%m-%d') - df_strat_smoothed.loc[0, 'date']).days
    df_strat_smoothed = smooth_big_jump_helper(
        df_strat_smoothed, 'non_o2_beds', 'o2_beds', d1, smoothing_length=length, method='weighted-mag')

    d1 = '2020-06-19'
    length = (datetime.strptime(d1, '%Y-%m-%d') - df_strat_smoothed.loc[0, 'date']).days
    df_strat_smoothed = smooth_big_jump_helper(
        df_strat_smoothed, 'non_o2_beds', 'o2_beds', d1, smoothing_length=length, method='weighted-mag')

    d1 = '2020-06-09'
    length = (datetime.strptime(d1, '%Y-%m-%d') - df_strat_smoothed.loc[0, 'date']).days
    df_strat_smoothed = smooth_big_jump_helper(
        df_strat_smoothed, 'hq', 'non_o2_beds', d1, smoothing_length=length, method='weighted-mag')

    d1 = '2020-06-17'
    length = (datetime.strptime(d1, '%Y-%m-%d') - df_strat_smoothed.loc[0, 'date']).days
    df_strat_smoothed = smooth_big_jump_helper(
        df_strat_smoothed, 'hq', 'non_o2_beds', d1, smoothing_length=length, method='weighted-mag')

    d1 = '2020-06-17'
    length = (datetime.strptime(d1, '%Y-%m-%d') - df_strat_smoothed.loc[0, 'date']).days
    df_strat_smoothed = smooth_big_jump_helper(
        df_strat_smoothed, 'non_o2_beds', 'hq', d1, smoothing_length=length, method='weighted-mag')

    d1 = '2020-06-14'
    length = (datetime.strptime(d1, '%Y-%m-%d') - df_strat_smoothed.loc[0, 'date']).days
    df_strat_smoothed = smooth_big_jump_helper(
        df_strat_smoothed, 'o2_beds', 'non_o2_beds', d1, smoothing_length=length, method='weighted-mag')

    d1 = '2020-06-14'
    length = (datetime.strptime(d1, '%Y-%m-%d') - df_strat_smoothed.loc[0, 'date']).days
    df_strat_smoothed = smooth_big_jump_helper(
        df_strat_smoothed, 'non_o2_beds', 'hq', d1, smoothing_length=length, method='weighted-mag')

    d1 = '2020-06-15'
    length = (datetime.strptime(d1, '%Y-%m-%d') - df_strat_smoothed.loc[0, 'date']).days
    df_strat_smoothed = smooth_big_jump_helper(
        df_strat_smoothed, 'non_o2_beds', 'hq', d1, smoothing_length=length, method='weighted-mag')

    d1 = '2020-06-09'
    length = (datetime.strptime(d1, '%Y-%m-%d') - df_strat_smoothed.loc[0, 'date']).days
    df_strat_smoothed = smooth_big_jump_helper(
        df_strat_smoothed, 'non_o2_beds', 'hq', d1, smoothing_length=length, method='weighted-mag')

    return df_strat_smoothed
