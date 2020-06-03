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

    df_district = smoothingfunc(df_district, last_n_days=n_days_back_smooth)
    
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

def smooth_using_active(df_district, last_n_days=60, cols=['hospitalised', 'recovered']):
    df = copy(df_district)
    districtdf = copy(df_district)    
    # create column of new total infected cases 14 days ago
    df['newcases_14back'] = df['total_infected'].shift(14) - df['total_infected'].shift(15)
    df = df.reset_index(drop=True)
    # truncate df to end on 5/29 and start on 5/29-ndays
    jump_date = np.datetime64('2020-05-30')
    df = df[df['date'] < jump_date]
    df = df[-last_n_days:]

    # calculate how many new infected cases over len(df)
    allinfected = df['newcases_14back'].sum()
    # to distribute proportionally, use delta/allinfected -> %of new cases for each day
    df['percent'] = df['newcases_14back']/allinfected
    # TODO: uniformly, delta/len(df)
    
    districtdf = districtdf.set_index('date')
    df = df.set_index('date')
    for col in cols:
        new_col = f'new_{col}'
        # calc jump for each col
        jump = districtdf.loc[jump_date, col] - districtdf.loc[jump_date - 1, col]
        # jump * %newcases for each col, round
        df[new_col] = (jump * df['percent']).round(0)
        # check if all were redistributed, add surplus to last row
        df.loc[df.index[-1], new_col] += jump - df[new_col].sum()
        # add new_additions.cumsum() to orig_col
        df[col] +=  df[new_col].cumsum()
        # merge into original df
        districtdf.loc[df.index, col] = df[col].astype(int)
    districtdf['total_infected'] = districtdf['deceased'] + districtdf['hospitalised'] + districtdf['recovered']
    return districtdf.reset_index()[44:]

# ---
if __name__ == "__main__":
    # --- turn into command line args
    parser = argparse.ArgumentParser() 
    parser.add_argument("-t", "--use-tracker", help="district name", required=False, action='store_true')
    parser.add_argument("-i", "--iterations", help="optimiser iterations", required=False, default=700, type=int)
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
                data_from_tracker=args.use_tracker, initialisation='intermediate', num_evals=args.iterations,
                which_compartments=['hospitalised', 'total_infected', 'deceased', 'recovered'])
            predictions_dict[(state, district)]['m2'] = rsingle_fitting_cycle(smooth_using_active,
                dataframes, state, district, train_period=7, val_period=0, num_evals=args.iterations,
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

