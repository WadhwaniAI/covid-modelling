import pandas as pd
import numpy as np
import copy

from data.processing.processing import get_custom_data_from_db


def scale_up_acc_to_testing(predictions_dict, scenario_on_which_df='best', testing_scaling_factor=1.5, 
                            time_window_to_scale=14):
    
    df_true = copy.copy(predictions_dict['m2']['forecasts'][scenario_on_which_df])
    df_true['daily_cases'] = df_true['total_infected'].diff()
    df_true.dropna(axis=0, how='any', inplace=True)

    last_date = predictions_dict['m2']['df_train'].iloc[-1]['date']
    df_true = df_true.loc[df_true['date'] > last_date, :]
    df_true.reset_index(inplace=True, drop=True)

    df_subset = copy.copy(df_true.iloc[:time_window_to_scale, :])

    df_subset['total_infected'] = df_subset['total_infected'] + \
        (df_subset['daily_cases']*testing_scaling_factor - df_subset['daily_cases']).cumsum()

    df_true.set_index('date', inplace=True)
    df_subset.set_index('date', inplace=True)

    df_true.loc[df_true.index.isin(df_subset.index), :] = df_subset.loc[:, :]
    del df_true['daily_cases']
    df_true.reset_index(inplace=True)
    df_subset.reset_index(inplace=True)
    return df_subset
