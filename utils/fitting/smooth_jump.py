import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import copy

def smooth_big_jump_helper(df_district, smoothing_var, auxillary_var, d1, d2=None, smoothing_length=None, 
                           method='uniform', t_recov=14, description="", aux_var_add=False):
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
        d2 = d1 + timedelta(days=1)

    d1 = datetime.combine(d1, datetime.min.time())
    d2 = datetime.combine(d2, datetime.min.time())
    big_jump = df_district.loc[d2, smoothing_var] - df_district.loc[d1, smoothing_var]
    aux_var_weight = (int(aux_var_add) - 0.5)*2

    description += (f'Smoothing {big_jump} {smoothing_var} between {d1.strftime("%Y-%m-%d")} and ' +
                    f'{(d1 - timedelta(days=smoothing_length)).strftime("%Y-%m-%d")} ({smoothing_length}) days ' +
                    f'in a {method} manner\n')
    if method == 'uniform':
        for i, day_number in enumerate(range(smoothing_length-2, -1, -1)):
            date = d1 - timedelta(days=day_number)
            offset = np.random.binomial(1, (big_jump%smoothing_length)/smoothing_length)
            df_district.loc[date, smoothing_var] += ((i+1)*big_jump)//smoothing_length + offset
            df_district.loc[date, auxillary_var] -= aux_var_weight*(((i+1)*big_jump)//smoothing_length + offset)

    elif 'weighted' in method:
        if method == 'weighted-recov':
            newcases = df_district['total'].shift(t_recov) - df_district['total'].shift(t_recov + 1)
        elif method == 'weighted-diff' or method == 'weighted':
            newcases = df_district.loc[:, smoothing_var].shift(0) - df_district.loc[:, smoothing_var].shift(1)
        elif method == 'weighted-mag':
            newcases = df_district.loc[:, smoothing_var]
        else:
            raise Exception("unknown smoothing method provided")

        valid_idx = newcases.first_valid_index()
        if smoothing_length != None:
            window_start = d1 - timedelta(days=smoothing_length - 1)
        else:
            window_start = d1 - timedelta(days=365)
        newcases = newcases.loc[max(valid_idx, window_start):d1]
        truncated = df_district.loc[max(valid_idx, window_start):d1, :]
        smoothing_length = len(truncated)
        invpercent = newcases.sum()/newcases
        for day_number in range(smoothing_length-1, -1, -1):
            date = d1 - timedelta(days=day_number)
            offset = np.random.binomial(1, (big_jump%invpercent.loc[date])/invpercent.loc[date])
            truncated.loc[date:, smoothing_var] += (big_jump // invpercent.loc[date]) + offset
            truncated.loc[date:, auxillary_var] += aux_var_weight*((big_jump // invpercent.loc[date]) + offset)
        df_district.loc[truncated.index, smoothing_var] = truncated[smoothing_var].astype('int64')
        df_district.loc[truncated.index, auxillary_var] = truncated[auxillary_var].astype('int64')

    else:
        raise Exception("unknown smoothing method provided")
    
    return df_district.reset_index(), description


def smooth_big_jump(df_district, smooth_jump_params):
    if 'stratified_smoothing' in smooth_jump_params.keys():
        del smooth_jump_params['stratified_smoothing']
    description = ""
    for date, params in smooth_jump_params.items():
        smooth_to_date, main_comp, aux_comp, aux_var_add, method = params

        if smooth_to_date is None:
            length = (date - df_district.loc[0, 'date'].date()).days
        else:
            length = (date - smooth_to_date).days
        
        df_district, description = smooth_big_jump_helper(
            df_district, main_comp, aux_comp, date, smoothing_length=length, method=method,
            description=description, aux_var_add=aux_var_add)

    print(sum(df_district['total'] == df_district['recovered'] \
              + df_district['deceased'] + df_district['active']), len(df_district))

    return df_district, description


def smooth_big_jump_stratified(df_strat, df_not_strat, smooth_jump_params):
    # Smooth unstratified array
    smooth_jump_params = copy.deepcopy(smooth_jump_params)
    stratified_smooth_params = copy.deepcopy(smooth_jump_params['stratified_smoothing'])
    del smooth_jump_params['stratified_smoothing']
    df_smoothed, description = smooth_big_jump(df_not_strat, smooth_jump_params)
    df_strat_smoothed = copy.copy(df_strat)
    # Compute difference array
    diff_array = df_smoothed.loc[df_smoothed['date'].isin(
        df_strat['date']), 'active'].reset_index(drop=True) - df_strat['active']

    # Copy the unstratified array smoothed columns to the smoothed stratified dataframe
    base_columns = ['total', 'active', 'recovered', 'deceased']
    df_strat_smoothed.loc[:, base_columns] = df_smoothed.loc[df_smoothed['date'].isin(
        df_strat['date']), base_columns].reset_index(drop=True)
    
    # Since hq and asymptomatic are inferred time series, infer them again with new time smoothed active time series
    df_strat_smoothed['hq'] = df_strat_smoothed['active'] - \
        df_strat_smoothed.loc[:, ['o2_beds', 'non_o2_beds', 'icu', 'ventilator']].sum(axis=1)
    df_strat_smoothed['asymptomatic'] = df_strat_smoothed['active'] - (
        df_strat_smoothed['symptomatic'] + df_strat_smoothed['critical'])

    for date, params in stratified_smooth_params.items():
        smooth_to_date, main_comp, aux_comp, aux_var_add, method = params

        if smooth_to_date is None:
            length = (date - df_strat_smoothed.loc[0, 'date'].date()).days
        else:
            length = (date - smooth_to_date).days

        df_strat_smoothed, description = smooth_big_jump_helper(
            df_strat_smoothed, main_comp, aux_comp, date, smoothing_length=length, method=method,
            description=description, aux_var_add=aux_var_add)

    return df_strat_smoothed, description
