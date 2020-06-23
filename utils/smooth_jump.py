import numpy as np
import pandas as pd
import datetime


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
        d2 = datetime.datetime.strptime(d1, '%Y-%m-%d') + datetime.timedelta(days=1)
    big_jump = df_district.loc[d2, smoothing_var] - df_district.loc[d1, smoothing_var]
    aux_var_weight = (int(aux_var_add) - 0.5)*2

    print(big_jump)
    if method == 'uniform':
        for i, day_number in enumerate(range(smoothing_length-2, -1, -1)):
            date = datetime.datetime.strptime(d1, '%Y-%m-%d') - datetime.timedelta(days=day_number)
            offset = np.random.binomial(1, (big_jump%smoothing_length)/smoothing_length)
            df_district.loc[date, smoothing_var] += ((i+1)*big_jump)//smoothing_length + offset
            df_district.loc[date, auxillary_var] -= aux_var_weight*(((i+1)*big_jump)//smoothing_length + offset)

    elif method == 'weighted-recov':
        newcases = df_district['total_infected'].shift(t_recov) - df_district['total_infected'].shift(t_recov + 1)
        valid_idx = newcases.first_valid_index()
        if smoothing_length != None:
            window_start = datetime.datetime.strptime(d1, '%Y-%m-%d') - datetime.timedelta(days=smoothing_length - 1)
        else:
            window_start = datetime.datetime.strptime(d1, '%Y-%m-%d') - datetime.timedelta(days=365)
        newcases = newcases.loc[max(valid_idx, window_start):d1]
        truncated = df_district.loc[max(valid_idx, window_start):d1, :]
        smoothing_length = len(truncated)
        print(f'smoothing length truncated to {smoothing_length}')
        invpercent = newcases.sum()/newcases
        for day_number in range(smoothing_length-1, -1, -1):
            date = datetime.datetime.strptime(d1, '%Y-%m-%d') - datetime.timedelta(days=day_number)
            offset = np.random.binomial(1, (big_jump%invpercent.loc[date])/invpercent.loc[date])
            truncated.loc[date:, smoothing_var] += (big_jump // invpercent.loc[date]) + offset
            truncated.loc[date:, auxillary_var] += aux_var_weight*((big_jump // invpercent.loc[date]) + offset)
        df_district.loc[truncated.index, smoothing_var] = truncated[smoothing_var].astype('int64')
        df_district.loc[truncated.index, auxillary_var] = truncated[auxillary_var].astype('int64')

    elif method == 'weighted':
        newcases = df_district[smoothing_var].shift(0) - df_district[smoothing_var].shift(1)
        valid_idx = newcases.first_valid_index()
        if smoothing_length != None:
            window_start = datetime.datetime.strptime(d1, '%Y-%m-%d') - datetime.timedelta(days=smoothing_length - 1)
        else:
            window_start = datetime.datetime.strptime(d1, '%Y-%m-%d') - datetime.timedelta(days=365)
        newcases = newcases.loc[max(valid_idx, window_start):d1]
        truncated = df_district.loc[max(valid_idx, window_start):d1, :]
        smoothing_length = len(truncated)
        print(f'smoothing length truncated to {smoothing_length}')
        invpercent = newcases.sum()/newcases
        for day_number in range(smoothing_length-1, -1, -1):
            date = datetime.datetime.strptime(d1, '%Y-%m-%d') - datetime.timedelta(days=day_number)
            offset = np.random.binomial(1, (big_jump%invpercent.loc[date])/invpercent.loc[date])
            truncated.loc[date:, smoothing_var] += (big_jump // invpercent.loc[date]) + offset
            truncated.loc[date:, auxillary_var] += aux_var_weight*((big_jump // invpercent.loc[date]) + offset)
        df_district.loc[truncated.index, smoothing_var] = truncated[smoothing_var].astype('int64')
        df_district.loc[truncated.index, auxillary_var] = truncated[auxillary_var].astype('int64')

    else:
        raise Exception("unknown smoothing method provided")
    
    return df_district.reset_index()


def smooth_big_jump(df_district, data_from_tracker=False, method='weighted'):
    d1 = '2020-05-28'
    df_district = smooth_big_jump_helper(
        df_district, 'recovered', 'hospitalised', d1, smoothing_length=33, method=method)
    
    d1 = '2020-06-14'
    length = (datetime.datetime.strptime(d1, '%Y-%m-%d') - datetime.datetime.strptime('2020-05-28', '%Y-%m-%d')).days
    df_district = smooth_big_jump_helper(
        df_district, 'recovered', 'hospitalised', d1, smoothing_length=length, method=method)

    d1 = '2020-06-15'
    length = (datetime.datetime.strptime(d1, '%Y-%m-%d') - datetime.datetime.strptime('2020-05-28', '%Y-%m-%d')).days
    df_district = smooth_big_jump_helper(
        df_district, 'hospitalised', 'recovered', d1, smoothing_length=length, method=method)

    d1 = '2020-06-15'
    length = (datetime.datetime.strptime(d1, '%Y-%m-%d') - df_district.loc[0, 'date']).days
    df_district = smooth_big_jump_helper(
        df_district, 'deceased', 'total_infected', d1, smoothing_length=length, method=method, aux_var_add=True)
    assert((df_district['total_infected'] == df_district['recovered'] \
        + df_district['deceased'] + df_district['hospitalised']).all())
    print((df_district['total_infected'] == df_district['recovered'] \
        + df_district['deceased'] + df_district['hospitalised']).all())

    return df_district
