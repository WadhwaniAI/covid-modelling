import os
import pdb
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from hyperopt import hp, tpe, fmin, Trials
from tqdm import tqdm

from collections import OrderedDict, defaultdict
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


np.random.seed(42)
now = str(datetime.now())


def get_data(dataframes, state, district, use_dataframe='districts_daily', disable_tracker=False, filename=None):
    if disable_tracker:
        df_district = pd.read_csv(filename)
        df_district['date'] = pd.to_datetime(df_district['date'])
        #TODO add support of adding 0s column for the ones which don't exist
        return df_district

    df_district = get_district_time_series(dataframes, state=state, district=district, use_dataframe=use_dataframe)
    return df_district

def train_val_split(df_district, train_rollingmean=False, val_rollingmean=False, val_size=5, 
                    which_columns = ['hospitalised', 'total_infected', 'deceased', 'recovered']):
    print("splitting data ..")
    df_true_fitting = copy.copy(df_district)
    for column in which_columns:
        df_true_fitting[column] = df_true_fitting[column].rolling(5, center=True).mean()
    
    df_true_fitting = df_true_fitting[np.logical_not(df_true_fitting['total_infected'].isna())]
    df_true_fitting.reset_index(inplace=True, drop=True)
    
    if train_rollingmean:
        if val_size == 0:
            df_train = pd.concat([df_true_fitting, df_district.iloc[-(val_size+2):, :]], ignore_index=True)
            return df_train, None, df_true_fitting
        else:
            df_train = pd.concat([df_true_fitting.iloc[:-val_size, :], df_district.iloc[-(val_size+2):-val_size, :]], 
                                 ignore_index=True)   
    else:
        if val_size == 0:
            return df_district, None, df_true_fitting  
        else:
            df_train = df_district.iloc[:-val_size, :]
        
    if val_rollingmean:
        df_val = pd.concat([df_true_fitting.iloc[-(val_size-2):, :], df_district.iloc[-2:, :]], ignore_index=True)
    else:
        df_val = df_district.iloc[-val_size:, :]
    df_val.reset_index(inplace=True, drop=True)
    return df_train, df_val, df_true_fitting

def single_pass(dataframes, state, district, train_period=7, val_period=7, train_on_val=False, plot_fit=True, 
                 pre_lockdown=False):

    df_district = get_district_time_series(dataframes, state=state, district=district)
    if district is None:
        district = ''
    
    if pre_lockdown:
        lockdown_index = df_district.index[df_district['date'] == "2020-03-24"][0]
        df_district = df_district.loc[:lockdown_index, :]
        train_on_val = True

    print('fitting to data with "train_on_val" set to {} ..'.format(train_on_val))

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

    if plot_fit:
        create_fitting_plots(df_prediction, df_district, df_train, df_val, df_true_fitting, train_period, state, district, train_on_val)

    return best, default_params, optimiser, df_prediction, df_district

def create_fitting_plots(df_prediction, df_district, df_train, df_val, df_true_fitting, train_period, state, district, train_on_val):
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
    fname = './plots/{}_predictions_{}_{}.png'.format(now, state, district)
    plt.savefig(fname)
    print("plot saved as {}".format(fname))

def predict(dataframes, district_to_plot, train_on_val=False, plot_fit=True, pre_lockdown=False):
    predictions_dict = {}

    for state, district in district_to_plot:
        print('state - {}, district - {}'.format(state, district))
        best, default_params, optimiser, df_prediction, df_district = fit_district(dataframes, state, district, train_on_val=train_on_val, 
                                                                                    plot_fit=plot_fit, pre_lockdown=pre_lockdown)
        predictions_dict[(state, district)] = {}
        for name in ['best', 'default_params', 'optimiser', 'df_prediction', 'df_district']:
            predictions_dict[(state, district)][name] = eval(name)

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

        loss = loss_calculator._calc_mape(np.array(df_prediction.iloc[-10:, :]['total_infected']), np.array(df_train.iloc[-10:, :]['total_infected']))
        df_result.loc[i, 'train_mape'] = loss

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
            loss = loss_calculator._calc_mape(np.array(df_prediction.iloc[-10:, :]['total_infected']), np.array(df_district.iloc[-10:, :]['total_infected']))
            df_result.loc[i, 'second_train_mape'] = loss
            df_district.reset_index(inplace=True)
            df_prediction.reset_index(inplace=True)

    return df_result

def save_regional_params(predictions_dict_val_train, df_result):
    print("saving best found parameters ..")
    params = {}
    regional_model_params_array = []

    for key in predictions_dict_val_train.keys():
        district_dict = {}
        state, district = key[0], key[1]
        try:
            district_dict['region_name'] = district.lower()
            district_dict['region_type'] = 'district'
        except Exception as e:
            district_dict['region_name'] = state.lower()
            district_dict['region_type'] = 'state'
        district_dict['model_name'] = 'SEIR'
        
        district_dict['model_parameters'] = {**copy.deepcopy(predictions_dict_val_train[key]['best']), **copy.deepcopy(predictions_dict_val_train[key]['default_params'])}
        district_dict['model_parameters']['starting_date'] = district_dict['model_parameters']['starting_date'].strftime("%Y-%m-%d") 
        
        if state == 'Delhi':
            error = df_result.loc[np.logical_and(df_result['state'] == state, 1), 'val_mape_observed'].tolist()[0]
        else:
            error = df_result.loc[np.logical_and(df_result['state'] == state, df_result['district'] == district), 'val_mape_observed'].tolist()[0]
        district_dict['val_error'] = error
        
        regional_model_params_array.append(district_dict)
        
    params['regional_model_params'] = regional_model_params_array

    if not os.path.exists('./params'):
        os.makedirs('./params')
    fname = './params/model_params_seir_t.json'
    with open(fname, 'w') as fp:
        json.dump(params, fp)
    print("parameters saved as {}".format(fname))

def get_forecasts(predictions_dict_val_train: dict):
    print("getting forecasts ..")
    new_lockdown_removal_date = "2020-06-15"
    new_lockdown_removal_date = datetime.strptime(new_lockdown_removal_date, '%Y-%m-%d')
    end_date = "2020-06-30"
    forecasts = defaultdict(dict)
    for region in predictions_dict_val_train:
        city = predictions_dict_val_train[region].copy()
        city['default_params']['intervention_removal_day'] = (new_lockdown_removal_date - city['default_params']['starting_date']).days
        df_train, df_val, df_true_fitting = train_val_split(city['df_district'], val_rollingmean=False, val_size=7)
        optimiser = city['optimiser']
        forecasts[region] = optimiser.solve(city['best'], city['default_params'], df_val, end_date=end_date)
    return forecasts

def create_csv_data(forecasts: dict, predictions_dict_val_train: dict, df_result: pd.DataFrame, end_date: str):
    print("compiling csv data ..")
    simulate_till = datetime.strptime(end_date, '%Y-%m-%d')
    dfs = defaultdict()
    for region in forecasts:
        state, district = region
        
        columns = ['forecastRunDate', 'regionType', 'region', 'model_name', 'error_function', 'error_value', 'current_total', 'current_active', 'current_recovered', 
               'current_deceased', 'current_hosptialized', 'current_icu', 'current_ventilator', 'predictionDate', 'active_mean', 'active_min', 
               'active_max', 'hospitalized_mean', 'hospitalized_min', 'hospitalized_max', 'icu_mean', 'icu_min', 'icu_max', 'deceased_mean', 
               'deceased_min', 'deceased_max', 'recovered_mean', 'recovered_min', 'recovered_max', 'total_mean', 'total_min', 'total_max']

        df_output = pd.DataFrame(columns = columns)
        
        city = predictions_dict_val_train[region].copy()
        df_train, df_val, df_true_fitting = train_val_split(city['df_district'], val_rollingmean=False, val_size=5)
        start_date = df_train.iloc[0, 0]
        
        prediction_daterange = pd.date_range(start=start_date, end=simulate_till)
        no_of_predictions = len(prediction_daterange)
        
        df_output['predictionDate'] = prediction_daterange
        df_output['forecastRunDate'] = [datetime.today().date()]*no_of_predictions
        
        df_output['regionType'] = ['city']*no_of_predictions
        
        df_output['model_name'] = ['SEIR']*no_of_predictions
        df_output['error_function'] = ['MAPE']*no_of_predictions
        
        if state == 'Delhi':
            error = df_result.loc[np.logical_and(df_result['state'] == state, 1), 'val_mape_observed'].tolist()
        else:
            error = df_result.loc[np.logical_and(df_result['state'] == state, df_result['district'] == district), 'val_mape_observed'].tolist()
            
        df_output['error_value'] = [error[0]]*no_of_predictions

        pred_hospitalisations = forecasts[region]['hospitalisations']
        df_output['active_mean'] = pred_hospitalisations
        df_output['active_min'] = (1 - 0.01*error[0])*pred_hospitalisations
        df_output['active_max'] = (1 + 0.01*error[0])*pred_hospitalisations
        
        df_output['hospitalized_mean'] = pred_hospitalisations
        df_output['hospitalized_min'] = (1 - 0.01*error[0])*pred_hospitalisations
        df_output['hospitalized_max'] = (1 - 0.01*error[0])*pred_hospitalisations
        
        df_output['icu_mean'] = 0.02*pred_hospitalisations
        df_output['icu_min'] = (1 - 0.01*error[0])*0.02*pred_hospitalisations
        df_output['icu_max'] = (1 - 0.01*error[0])*0.02*pred_hospitalisations
        
        pred_recoveries = forecasts[region]['recoveries']
        df_output['recovered_mean'] = pred_recoveries
        df_output['recovered_min'] = (1 - 0.01*error[0])*pred_recoveries
        df_output['recovered_max'] = (1 - 0.01*error[0])*pred_recoveries
        
        pred_fatalities = forecasts[region]['fatalities']
        df_output['deceased_mean'] = pred_fatalities
        df_output['deceased_min'] = (1 - 0.01*error[0])*pred_fatalities
        df_output['deceased_max'] = (1 - 0.01*error[0])*pred_fatalities
        
        pred_total_cases = pred_hospitalisations + pred_recoveries + pred_fatalities
        df_output['total_mean'] = pred_total_cases
        df_output['total_min'] = (1 - 0.01*error[0])*pred_total_cases
        df_output['total_max'] = (1 - 0.01*error[0])*pred_total_cases
        
        if state == 'Delhi':
            district = 'Delhi'
        df_output['region'] = [district]*no_of_predictions
        
        df_output.set_index('predictionDate', inplace=True)
        df_district = predictions_dict_val_train[region]['df_district']
        df_output.loc[df_output.index.isin(df_district['date']), 'current_total'] = df_district['total_infected'].iloc[2:].to_numpy()
        df_output.reset_index(inplace=True)
        df_output = df_output[columns]
        
        dfs[region] = df_output

    return dfs

def write_csv(dfs: dict):
    print("dumping csv ..")
    df_final = pd.DataFrame(columns=dfs[('Delhi', None)].columns)
    for key in dfs.keys():
        df_final = pd.concat([df_final, dfs[key]], ignore_index=True)

    if not os.path.exists('./csv_output'):
        os.makedirs('./csv_output')
    fname = './csv_output/{}_final_output.csv'.format(now)
    df_final.to_csv(fname, index=False)
    print("csv saved as {}".format(fname))

def pre_lockdown_R0(dataframes, regions):
    print("calculating pre-lockdown R0s")
    r0 = defaultdict()
    predictions_dict = predict(dataframes, regions, plot_fit=False, pre_lockdown=True)
    for region in predictions_dict:
        r0[region] = predictions_dict[region]['best']['R0']

    print("pre-lockdown R0s are", r0)
    return r0

def main():
    print("all files will be saved with prefix {}.".format(now))
    dataframes = get_covid19india_api_data()
    district_to_plot = [['Maharashtra', 'Pune'],
                        ['Delhi', None],
                        ['Rajasthan', 'Jaipur'], 
                        ['Maharashtra', 'Mumbai'],
                        ['Gujarat', 'Ahmedabad'],
                        ['Karnataka', 'Bengaluru Urban']
                       ]

    old_r0 = pre_lockdown_R0(dataframes, district_to_plot)
    predictions_dict = predict(dataframes, district_to_plot)
    predictions_dict = change_mumbai_df(predictions_dict)
    predictions_dict_final = copy.deepcopy(predictions_dict)
    create_full_plots(predictions_dict)
    predictions_dict_val_train = predict(dataframes, district_to_plot, train_on_val=True)
    df_result = create_classification_report(predictions_dict, predictions_dict_val_train)
    if not os.path.exists('./results'):
        os.makedirs('./results')
    df_result.to_csv('./results/{}_df_result.csv'.format(now))
    save_regional_params(predictions_dict_val_train, df_result)
    forecasts = get_forecasts(predictions_dict_val_train)
    csv_data = create_csv_data(forecasts, predictions_dict_val_train, df_result, end_date="2020-06-30")
    write_csv(csv_data)

if __name__ == "__main__":
    main()

