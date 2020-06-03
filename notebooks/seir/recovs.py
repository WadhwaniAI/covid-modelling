import os
import numpy as np
import pandas as pd
import datetime
import argparse
from copy import copy

import sys
sys.path.append('../..')
from data.dataloader import get_covid19india_api_data
from models.ihme.dataloader import get_dataframes_cached
from data.processing import get_concat_data
from main.seir.fitting import get_regional_data, data_setup, run_cycle, get_variable_param_ranges
from main.seir.forecast import create_all_csvs, write_csv, plot_forecast
from utils.create_report import create_report

def rsingle_fitting_cycle(smoothingfunc, dataframes, state, 
    district, train_period=7, val_period=7, train_on_val=False, 
    num_evals=1500, data_from_tracker=True, filename=None, data_format='new', pre_lockdown=False, N=1e7, which_compartments=['hospitalised', 'total_infected'], initialisation='starting', n_days_back_smooth=60):
    print('fitting to data with "train_on_val" set to {} ..'.format(train_on_val))

    # Get date
    _, df_district_raw_data = get_regional_data(dataframes, state, district, 
        data_from_tracker, data_format, filename)

    df_district = get_concat_data(dataframes, state=state, district=district, concat=True)

    new_df_district = smoothingfunc(df_district, last_n_days=n_days_back_smooth)

    new_df_district['recovered'] = new_df_district['n_recovered']
    new_df_district['hospitalised'] = new_df_district['n_hospitalised']
    del new_df_district['n_recovered']
    del new_df_district['n_hospitalised']
    df_district = new_df_district
    
    observed_dataframes = data_setup(
        df_district, df_district_raw_data, val_period
    )

    print('train\n', observed_dataframes['df_train'].tail())
    print('val\n', observed_dataframes['df_val'])
    
    return run_cycle(
        state, district, observed_dataframes, data_from_tracker=data_from_tracker,
        train_period=train_period, num_evals=num_evals, N=N, 
        which_compartments=which_compartments, initialisation=initialisation
    )

def smooth_using_active1(df_district, df_district_raw_data=None, last_n_days=60):
    districtdf = copy(df_district)
    df = copy(df_district)

    df['delta_total_14ago'] = df['total_infected'].shift(14) - df['total_infected'].shift(15)
    df = df.reset_index(drop=True)
    trunc = df
    jump_date = trunc.index[trunc['date'] == np.datetime64('2020-05-30')].tolist()[0] - 1
    trunc = trunc.loc[:jump_date]
    trunc = trunc[-last_n_days:]

    recovs_to_redistribute = df.loc[jump_date+1, 'recovered'] - df.loc[jump_date, 'recovered']
    active_to_redistribute = df.loc[jump_date+1, 'hospitalised'] - df.loc[jump_date, 'hospitalised']
    deaths_to_redistribute = df.loc[jump_date+1, 'deceased'] - df.loc[jump_date, 'deceased']
    
    allinfected = trunc['delta_total_14ago'].sum()
    trunc['%_total_14ago'] = trunc['delta_total_14ago']/allinfected
    
    trunc['new_recovered'] = (recovs_to_redistribute * trunc['%_total_14ago']).round(0)
    trunc['new_hospitalised'] = (active_to_redistribute * trunc['%_total_14ago']).round(0)
    trunc['new_deceased'] = (deaths_to_redistribute * trunc['%_total_14ago']).round(0)
    # print(recovs_to_redistribute, trunc['new_recovered'].sum())
    # print(active_to_redistribute, trunc['new_hospitalised'].sum())
    # print(deaths_to_redistribute, trunc['new_deceased'].sum())
    
    trunc.loc[trunc.index[-1], 'new_recovered'] += recovs_to_redistribute - trunc['new_recovered'].sum()
    trunc.loc[trunc.index[-1], 'new_hospitalised'] += active_to_redistribute - trunc['new_hospitalised'].sum()
    trunc.loc[trunc.index[-1], 'new_deceased'] += deaths_to_redistribute - trunc['new_deceased'].sum()

    trunc = trunc.set_index('date')
    districtdf = districtdf.set_index('date')

    trunc['recovered'] = trunc['recovered'] + trunc['new_recovered'].cumsum()
    trunc['hospitalised'] = trunc['hospitalised'] + trunc['new_hospitalised'].cumsum()
    trunc['deceased'] = trunc['deceased'] + trunc['new_deceased'].cumsum()

    districtdf.loc[:,'n_recovered'] = districtdf['recovered']
    districtdf.loc[trunc.index, 'n_recovered'] = trunc['recovered']
    
    districtdf.loc[:,'n_hospitalised'] = districtdf['hospitalised']
    districtdf.loc[trunc.index, 'n_hospitalised'] = trunc['hospitalised']

    districtdf.loc[:,'n_deceased'] = districtdf['deceased']
    districtdf.loc[trunc.index, 'n_deceased'] = trunc['deceased']
    
    districtdf = districtdf.reset_index()
    districtdf = districtdf[44:] # truncate old data from other dfs
    
    districtdf['n_recovered'] = districtdf['n_recovered'].astype(dtype=int)
    districtdf['n_hospitalised'] = districtdf['n_hospitalised'].astype(dtype=int)
    districtdf['n_deceased'] = districtdf['n_deceased'].astype(dtype=int)
    return districtdf

def smooth_using_active(df_district, last_n_days=60):
    df = copy(df_district)
    districtdf = copy(df_district)

    df['delta14'] = df['total_infected'].shift(14) - df['total_infected'].shift(15)
    df = df.reset_index(drop=True)

    jump_date = df.index[df['date'] == np.datetime64('2020-05-30')].tolist()[0] - 1
    recovs_to_redistribute = df.loc[jump_date+1, 'recovered'] - df.loc[jump_date, 'recovered']
    active_to_redistribute = df.loc[jump_date+1, 'hospitalised'] - df.loc[jump_date, 'hospitalised']
    deaths_to_redistribute = df.loc[jump_date+1, 'deceased'] - df.loc[jump_date, 'deceased']
    df = df.loc[:jump_date]
    df = df[-last_n_days:]

    all_infected = df['delta14'].sum()
    df['percent_delta14'] = df['delta14']/all_infected
    
    df['n_recovered'] = (recovs_to_redistribute * df['percent_delta14']).round(0)
    df['n_hospitalised'] = (active_to_redistribute * df['percent_delta14']).round(0)
    df['n_deceased'] = (deaths_to_redistribute * df['percent_delta14']).round(0)

    df.loc[df.index[-1], 'n_recovered'] += recovs_to_redistribute - df['n_recovered'].sum()
    df.loc[df.index[-1], 'n_hospitalised'] += active_to_redistribute - df['n_hospitalised'].sum()
    df.loc[df.index[-1], 'n_deceased'] += deaths_to_redistribute - df['n_deceased'].sum()

    df = df.set_index('date')
    districtdf = districtdf.set_index('date')

    df['recovered'] = df['recovered'] + df['n_recovered'].cumsum()
    df['hospitalised'] = df['hospitalised'] + df['n_hospitalised'].cumsum()
    df['deceased'] = df['deceased'] + df['n_deceased'].cumsum()
  
    districtdf.loc[:,'n_recovered'] = districtdf['recovered']
    districtdf.loc[df.index, 'n_recovered'] = df['recovered']
    
    districtdf.loc[:,'n_hospitalised'] = districtdf['hospitalised']
    districtdf.loc[df.index, 'n_hospitalised'] = df['hospitalised']

    districtdf.loc[:,'n_deceased'] = districtdf['deceased']
    districtdf.loc[df.index, 'n_deceased'] = df['deceased']
    
    districtdf = districtdf.reset_index()
    districtdf = districtdf[44:] # truncate old data from other dfs
    
    districtdf['n_recovered'] = districtdf['n_recovered'].astype(dtype=int)
    districtdf['n_hospitalised'] = districtdf['n_hospitalised'].astype(dtype=int)
    districtdf['n_deceased'] = districtdf['n_deceased'].astype(dtype=int)
    return districtdf
    

# ---
if __name__ == "__main__":
    # --- turn into command line args
    parser = argparse.ArgumentParser() 
    parser.add_argument("-t", "--use-tracker", help="district name", required=False, action='store_true')
    args = parser.parse_args()
    use_tracker = args.use_tracker
    # ---

    dataframes = get_dataframes_cached()
    predictions_dict = {}
    save_json = {
        'variable_param_ranges': get_variable_param_ranges(as_str=True)
    }
    today = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_folder = os.path.join(os.getcwd(), f'output/{today}')
    if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    districts_to_show = [('Maharashtra', 'Mumbai')]
    for ndays in [15]:
    # for ndays in [7, 15, 30, 60]:
        for state, district in districts_to_show:
            predictions_dict[(state, district)] = {}
            predictions_dict[(state, district)]['m1'] = rsingle_fitting_cycle(smooth_using_active,
                dataframes, state, district, train_period=7, val_period=7, 
                n_days_back_smooth=ndays,
                data_from_tracker=args.use_tracker, initialisation='intermediate', num_evals=700,
                which_compartments=['hospitalised', 'total_infected', 'deceased', 'recovered'])
            predictions_dict[(state, district)]['m2'] = rsingle_fitting_cycle(smooth_using_active,
                dataframes, state, district, train_period=7, val_period=0, num_evals=700,
                n_days_back_smooth=ndays,
                train_on_val=True, data_from_tracker=args.use_tracker, initialisation='intermediate',
                which_compartments=['hospitalised', 'total_infected', 'deceased', 'recovered'])
            
            predictions_dict[(state, district)]['state'] = state
            predictions_dict[(state, district)]['dist'] = district
            predictions_dict[(state, district)]['fitting_date'] = datetime.datetime.now().strftime("%Y-%m-%d")
            predictions_dict[(state, district)]['datasource'] = 'covid19api' if predictions_dict[(state, district)]['m1']['data_from_tracker'] else 'municipality'
            predictions_dict[(state, district)]['variable_param_ranges'] = predictions_dict[(state, district)]['m1']['variable_param_ranges']
            predictions_dict[(state, district)]['data_last_date'] = predictions_dict[(state, district)]['m2']['data_last_date']

        for m in [  'm1', 'm2']:
            starting_key = list(predictions_dict.keys())[0]

            loss_columns = pd.MultiIndex.from_product([predictions_dict[starting_key][m]['df_loss'].columns, predictions_dict[starting_key][m]['df_loss'].index])
            loss_index = predictions_dict.keys()

            df_loss_master = pd.DataFrame(columns=loss_columns, index=loss_index)
            for region in predictions_dict.keys():
                region_folder = os.path.join(output_folder, region[1])
                if not os.path.exists(region_folder):
                    os.makedirs(region_folder)

                df_loss_master.loc[region, :] = np.around(predictions_dict[region][m]['df_loss'].values.T.flatten().astype('float'), decimals=2)
                
                print (df_loss_master)
                fig = predictions_dict[region][m]['ax'].figure
                path = f'{m}-fit-{ndays}.png'
                fig.savefig(os.path.join(region_folder, path))

        # for region in predictions_dict.keys():
        #     predictions_dict[region]['forecast'] = plot_forecast(predictions_dict[region], region, both_forecasts=False, error_bars=True)

        # for region in predictions_dict.keys():
        #     create_report(predictions_dict[region])
            
        for region in predictions_dict.keys():
            region_folder = os.path.join(output_folder, region[1])
            if not os.path.exists(region_folder):
                os.makedirs(region_folder)
            
            m2_val_m1 = plot_forecast(predictions_dict[region], region, both_forecasts=False, error_bars=True)
            
            fig = m2_val_m1.figure
            path = f'm2_val_m1-fit-{ndays}.png'
            fig.savefig(os.path.join(region_folder, path))
            
            # m2_trials = predictions_dict[region]['m2']['trials']
            # m2_best_trials = sorted(m2_trials.trials, key=lambda x: x['result']['loss'], reverse=False)
            # report['top_10_trials'] = [trial['misc']['vals'] for trial in m2_best_trials[:10]]
            

        # df_output = create_all_csvs(predictions_dict, icu_fraction=0.02)
        # write_csv(df_output, '../../output-{}.csv'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

