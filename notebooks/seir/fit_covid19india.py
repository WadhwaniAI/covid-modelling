import os
import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from hyperopt import hp, tpe, fmin, Trials
from tqdm import tqdm

from collections import OrderedDict
import itertools
from functools import partial
from datetime import datetime
from joblib import Parallel, delayed
import copy

from data.dataloader import get_jhu_data, get_covid19india_api_data
from data.processing import get_district_time_series

from models.seir.seir_testing import SEIR_Testing
from main.seir.optimiser import Optimiser
from main.seir.losses import Loss_Calculator
from utils.plotting import create_plots


now = str(datetime.now())

def train_val_split(df_district, val_rollingmean=False, val_size=5):
    df_true_fitting = copy.copy(df_district)
    df_true_fitting['total_infected'] = df_true_fitting['total_infected'].rolling(5, center=True).mean()
    df_true_fitting = df_true_fitting[np.logical_not(df_true_fitting['total_infected'].isna())]
    df_true_fitting.reset_index(inplace=True, drop=True)
    
    if val_size == 0:
        df_train = pd.concat([df_true_fitting, df_district.iloc[-(val_size+2):, :]], ignore_index=True)
        return df_train, None, df_true_fitting
    else:
        df_train = pd.concat([df_true_fitting.iloc[:-val_size, :], df_district.iloc[-(val_size+2):-val_size, :]], ignore_index=True)
    if val_rollingmean:
        df_val = pd.concat([df_true_fitting.iloc[-(val_size-2):, :], df_district.iloc[-2:, :]], ignore_index=True)
    else:
        df_val = df_district.iloc[-val_size:, :]
    df_val.reset_index(inplace=True, drop=True)
    return df_train, df_val, df_true_fitting

def fit_district(dataframes, state, district, train_period=10, val_period=5, train_on_val=False):
    if not os.path.exists('./plots'):
        os.makedirs('./plots')

    df_district = get_district_time_series(dataframes, state=state, district=district)
    if district is None:
        district = ''
        
    # Get train val split
    if train_on_val:
        df_train, df_val, df_true_fitting = train_val_split(df_district, val_rollingmean=False, val_size=0)
    else:
        df_train, df_val, df_true_fitting = train_val_split(df_district, val_rollingmean=False, val_size=val_period)

    print('train\n', df_train.tail())
    print('val\n', df_val)

    # Initialise Optimiser
    optimiser = Optimiser()
    
    # Get the fixed params
    default_params = optimiser.init_default_params(df_train)

    # Create searchspace of variable params
    variable_param_ranges = {
        'R0' : hp.uniform('R0', 1.6, 5),
        'T_inc' : hp.uniform('T_inc', 4, 5),
        'T_inf' : hp.uniform('T_inf', 3, 4),
        'T_recov_severe' : hp.uniform('T_recov_severe', 9, 20),
        'P_severe' : hp.uniform('P_severe', 0.3, 0.99),
        'intervention_amount' : hp.uniform('intervention_amount', 0, 1)
    }
    
    # Perform Bayesian Optimisation
    best, trials = optimiser.bayes_opt(df_train, default_params, variable_param_ranges, method='rmse', num_evals=1500, loss_indices=[-train_period, None])

    print('best parameters\n', best)
    
    # Get Predictions dataframe
    df_prediction = optimiser.solve(best, default_params, df_train)

    # Create plots
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.plot(df_district['date'], df_district['total_infected'], label='Confirmed Cases (Observed)')
    ax.plot(df_true_fitting['date'], df_true_fitting['total_infected'], label='Confirmed Cases (Rolling Avg(5))')
    if train_on_val:
        ax.plot([df_train.iloc[-train_period, :]['date'], df_train.iloc[-train_period, :]['date']], [min(df_train['total_infected']), max(df_train['total_infected'])], '--r', label='Train Period Starts')
    else:
        ax.plot([df_train.iloc[-1, :]['date'], df_train.iloc[-1, :]['date']], [min(df_train['total_infected']), max(df_val['total_infected'])], '--r', label='Train Test Boundary')
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=15))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.ylabel('No of People')
    plt.xlabel('Time')
    plt.legend()
    plt.title('Rolling Avg vs Observed ({} {})'.format(state, district))
    plt.grid()
    plt.savefig('./plots/{}_observed_{}_{}.png'.format(now, state, district))

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.plot(df_train['date'], df_train['total_infected'], color='orange', label='Confirmed Cases (Rolling Avg (5))')
    ax.plot(df_prediction['date'], df_prediction['total_infected'], '-g', label='Confirmed Cases (Predicted)')
    if train_on_val:
        ax.plot([df_train.iloc[-train_period, :]['date'], df_train.iloc[-train_period, :]['date']], [min(df_train['total_infected']), max(df_train['total_infected'])], '--r', label='Train Period Starts')
    else:
        ax.plot([df_train.iloc[-1, :]['date'], df_train.iloc[-1, :]['date']], [min(df_train['total_infected']), max(df_val['total_infected'])], '--r', label='Train Test Boundary')
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=15))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.ylabel('No of People')
    plt.xlabel('Time')
    plt.legend()
    plt.title('Total Confirmed Cases ({} {})'.format(state, district))
    plt.grid()
    plt.savefig('./plots/{}_predictions_{}_{}.png'.format(now, state, district))
    
    return best, default_params, optimiser, df_prediction, df_district

def predict(dataframes, district_to_plot, train_on_val=False):
    predictions_dict = {}

    for state, district in district_to_plot:
        print('state - {}, district - {}'.format(state, district))
        best, default_params, optimiser, df_prediction, df_district = fit_district(dataframes, state, district, train_on_val)
        predictions_dict[(state, district)] = {}
        for name in ['best', 'default_params', 'optimiser', 'df_prediction', 'df_district']:
            predictions_dict[(state, district)][name] = eval(name)

    return predictions_dict

def change_mumbai_df(predictions_dict):
    df_temp = predictions_dict[('Maharashtra', 'Mumbai')]['df_district']
    df_temp.set_index('date', inplace=True)
    df_temp.loc['2020-04-18', 'total_infected'] = 2085
    df_temp.loc['2020-04-19', 'total_infected'] = 2268
    df_temp.loc['2020-04-20', 'total_infected'] = 2724
    df_temp.loc['2020-04-21', 'total_infected'] = 3032
    df_temp.loc['2020-04-22', 'total_infected'] = 3451
    df_temp.reset_index(inplace=True)
    predictions_dict[('Maharashtra', 'Mumbai')]['df_district'] = df_temp
    predictions_dict[('Maharashtra', 'Mumbai')]['df_district']

    return predictions_dict

def create_classification_report(predictions_dict, predictions_dict_val_train=None):
    
    df_result = pd.DataFrame(columns=['state', 'district', 'train_rmse', 'train_mape', 'pre_intervention_r0', 'post_intervention_r0',
                                      'val_rmse_observed', 'val_mape_observed', 'val_rmse_rolling', 'val_mape_rolling'])
    
    loss_calculator = Loss_Calculator()
    for i, key in enumerate(predictions_dict.keys()):
        df_result.loc[i, 'state'] = key[0]
        df_result.loc[i, 'district'] = key[1]
        df_district = predictions_dict[key]['df_district']
        df_prediction = predictions_dict[key]['df_prediction']
        best = predictions_dict[key]['best']
        optimiser = predictions_dict[key]['optimiser']
        default_params = predictions_dict[key]['default_params']
        
        df_train, df_val, df_true_fitting = train_val_split(df_district, val_rollingmean=False, val_size=5)

        loss = loss_calculator._calc_mape(df_prediction.iloc[-10:, :]['total_infected'], df_train.iloc[-10:, :]['total_infected'])
        df_result.loc[i, 'train_mape'] = loss
        loss = loss_calculator._calc_rmse(df_prediction.iloc[-10:, :]['total_infected'], df_train.iloc[-10:, :]['total_infected'])
        df_result.loc[i, 'train_rmse'] = loss

        df_result.loc[i, 'pre_intervention_r0'] = best['R0']
        df_result.loc[i, 'post_intervention_r0'] = best['R0']*best['intervention_amount']

        normalising_ratio = (df_train['total_infected'].iloc[-1] / df_prediction['total_infected'].iloc[-1])
        df_prediction = optimiser.solve(best, default_params, df_train, end_date=df_val.iloc[-1, :]['date'])
        df_prediction.loc[df_prediction['date'] == df_prediction.iloc[-1]['date'], 'total_infected'] = df_prediction.loc[ df_prediction['date'] == df_prediction.iloc[-1]['date'], 'total_infected'] * normalising_ratio
        df_prediction.loc[df_prediction['date'].isin(df_val['date']), 'total_infected'] = df_prediction.loc[ df_prediction['date'].isin(df_val['date']), 'total_infected'] * normalising_ratio
        df_prediction = df_prediction[df_prediction['date'].isin(df_val['date'])]
        df_prediction.reset_index(inplace=True, drop=True)

        loss = loss_calculator._calc_mape(df_prediction['total_infected'], df_val['total_infected'])
        df_result.loc[i, 'val_mape_observed'] = loss
        loss = loss_calculator._calc_rmse(df_prediction['total_infected'], df_val['total_infected'])
        df_result.loc[i, 'val_rmse_observed'] = loss

        _, df_val, _ = train_val_split(df_district, val_rollingmean=True, val_size=5)

        loss = loss_calculator._calc_mape(df_prediction['total_infected'], df_val['total_infected'])
        df_result.loc[i, 'val_mape_rolling'] = loss
        loss = loss_calculator._calc_rmse(df_prediction['total_infected'], df_val['total_infected'])
        df_result.loc[i, 'val_rmse_rolling'] = loss
        
        df_result.loc[i, 'train_period'] = '{} to {}'.format(df_train['date'].iloc[-10].date(), df_train['date'].iloc[-1].date())
        df_result.loc[i, 'val_period'] = '{} to {}'.format(df_val['date'].iloc[0].date(), df_val['date'].iloc[-1].date())
        df_result.loc[i, 'init_date'] = '{}'.format(df_train['date'].iloc[0].date())
        
        if predictions_dict_val_train != None:
            df_district = predictions_dict_val_train[key]['df_district']
            df_prediction = predictions_dict_val_train[key]['df_prediction']
            
            df_result.loc[i, 'second_train_period'] = '{} to {}'.format(df_district['date'].iloc[-10].date(), df_district['date'].iloc[-1].date())
            
            df_district.set_index('date', inplace=True)
            df_prediction.set_index('date', inplace=True)
            loss = loss_calculator._calc_mape(df_prediction.iloc[-10:, :]['total_infected'], df_district.iloc[-10:, :]['total_infected'])
            df_result.loc[i, 'second_train_mape'] = loss
            df_district.reset_index(inplace=True)
            df_prediction.reset_index(inplace=True)

    return df_result
    
def create_full_plots(predictions_dict):
    for i, key in enumerate(predictions_dict.keys()):
        
        df_district = predictions_dict[key]['df_district']
        df_prediction = predictions_dict[key]['df_prediction']
        best = predictions_dict[key]['best']
        optimiser = predictions_dict[key]['optimiser']
        default_params = predictions_dict[key]['default_params']
        
        df_train, df_val, df_true_fitting = train_val_split(df_district, val_rollingmean=False, val_size=5)
        
        # Create state init values before solving on val set
        last_prediction = df_prediction.iloc[-1, 1:12]
        normalising_ratio = (df_train['total_infected'].iloc[-1] / df_prediction['total_infected'].iloc[-1])
        last_prediction[1:] = list(map(lambda x : int(round(x)), df_prediction.iloc[-1, 2:12] * normalising_ratio))
        last_prediction[0] = default_params['N'] - sum(last_prediction[1:])
        state_init_values = last_prediction.to_dict(OrderedDict)
        
#         df_prediction = optimiser.solve(best, default_params, df_train, start_date=df_val.iloc[0, :]['date'], end_date=df_val.iloc[-1, :]['date'], 
#                                         state_init_values=state_init_values)
        df_prediction = optimiser.solve(best, default_params, df_train, end_date=df_val.iloc[-1, :]['date'])
        df_prediction.loc[df_prediction['date'] == df_prediction.iloc[-1]['date'], 'total_infected'] = df_prediction.loc[ df_prediction['date'] == df_prediction.iloc[-1]['date'], 'total_infected'] * normalising_ratio
        df_prediction.loc[df_prediction['date'].isin(df_val['date']), 'total_infected'] = df_prediction.loc[ df_prediction['date'].isin(df_val['date']), 'total_infected'] * normalising_ratio
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.plot(df_district['date'], df_district['total_infected'], label='Confirmed Cases (Observed)')
        ax.plot(df_true_fitting['date'], df_true_fitting['total_infected'], color='orange', label='Confirmed Cases (Rolling Avg (5))')
        ax.plot(df_prediction['date'], df_prediction['total_infected'], '-g', label='Confirmed Cases (Predicted)')
        ax.plot([df_train.iloc[-1, :]['date'], df_train.iloc[-1, :]['date']], [min(df_train['total_infected']), max(df_val['total_infected'])], '--r', label='Train Test Boundary')
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=15))
        ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.ylabel('No of People')
        plt.xlabel('Time')
        plt.legend()
        plt.title('Total Confirmed Cases Extrapolated ({})'.format(key))
        plt.grid()
        plt.savefig('./plots/{}_full_plot_{}.png',format(now, key))

def main():
    dataframes = get_covid19india_api_data()
    district_to_plot = [['Delhi', None],
                        ['Rajasthan', 'Jaipur'], 
                        ['Maharashtra', 'Mumbai'], 
                        ['Maharashtra', 'Pune'], 
                        ['Karnataka', 'Bengaluru'], 
                        ['Gujarat', 'Ahmadabad']
                       ]

    predictions_dict = predict(dataframes, district_to_plot)
    predictions_dict = change_mumbai_df(predictions_dict)
    predictions_dict_final = copy.deepcopy(predictions_dict)
    create_full_plots(predictions_dict)
    predictions_dict_val_train = predict(dataframes, district_to_plot, train_on_val=True)
    df_result = create_classification_report(predictions_dict, predictions_dict_val_train)
    if not os.path.exists('./results'):
        os.makedirs('./results')
    df_result.to_csv('./results/{}_df_result.csv'.format(now))
    


if __name__ == "__main__":
    main()