import pandas as pd
from datetime import datetime
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl

import sys
sys.path.append('../..')
from main.seir.fitting import data_setup, run_cycle
from viz import setup_plt

class SEIRBacktest:
    def __init__(self, state, district, df_district, df_district_raw_data, data_from_tracker):
        self.state = state
        self.district = district
        self.df_district = df_district
        self.df_district_raw_data = df_district_raw_data
        self.data_from_tracker = data_from_tracker
        self.num_evals = 700 if self.data_from_tracker else 1000
        self.results = None

    def test(self, fit, train_period=7, val_period=7, increment=5, 
        future_days=7, N=1e7, num_evals=None, pre_lockdown=False, 
        initialisation='intermediate',
        which_compartments=['hospitalised', 'total_infected', 'deceased', 'recovered']):
        
        if num_evals is None:
            num_evals = self.num_evals
        val_period = val_period if fit is 'm1' else 0
        train_on_val = False if fit is 'm1' else True

        runtime_s = time.time()
        start = pd.to_datetime(self.df_district['date']).min()
        end = pd.to_datetime(self.df_district['date']).max()
        print(start, end)
        n_days = (end - start).days + 1 - future_days

        results = {}
        seed = datetime.today().timestamp()
        for run_day in range(train_period + val_period + 1, n_days, increment):
            end_date = pd.to_datetime(self.df_district['date'], format='%Y-%m-%d').loc[run_day]
            print ("\rbacktesting for", end_date, end="")

            #  TRUNCATE DATA
            df_district_incr = self.df_district[pd.to_datetime(self.df_district['date'], format='%Y-%m-%d') <= end_date]
            df_district_raw_data_incr = self.df_district_raw_data[pd.to_datetime(self.df_district_raw_data['date'], format='%Y-%m-%d') <= end_date]
            
            observed_dataframes = data_setup(df_district_incr, df_district_raw_data_incr, val_period)
            
            # FIT/PREDICT
            res = run_cycle(
                self.state, self.district, observed_dataframes, data_from_tracker=self.data_from_tracker,
                train_period=train_period, num_evals=num_evals, N=N, 
                which_compartments=which_compartments, initialisation=initialisation
            )

            results[end_date] = {
                'n_days': n_days,
                'seed': seed,
                'results': res,
            }
        runtime = time.time() - runtime_s
        print (runtime)
        if df_val is None:
            df_val = pd.DataFrame(columns=observed_dataframes['df_train'].columns)
            df_val_nora = pd.DataFrame(columns=observed_dataframes['df_train_nora'].columns)
        else:
            df_val = observed_dataframes['df_val']
            df_val_nora = observed_dataframes['df_val_nora']
        self.results = {
            'results': results,
            'df_district': self.df_district,
            'df_true_plotting_rolling': pd.concat([observed_dataframes['df_train'], df_val], ignore_index=True),
            'df_true_plotting': pd.concat([observed_dataframes['df_train_nora'], df_val_nora], ignore_index=True),
            'future_days': future_days,
            'train_period': train_period,
            'runtime': runtime,
        }
        return self.results

    def plot_results(self, which_compartments=['hospitalised', 'total_infected'], description='', results=None):
        results = self.results['results'] if results is None else results
        # Create plots
        df_true_plotting_rolling = self.results['df_true_plotting_rolling']
        df_true_plotting = self.results['df_true_plotting']
        
        # import pdb; pdb.set_trace()
        if 'total_infected' in which_compartments:
            plt.plot(df_true_plotting['date'], df_true_plotting['total_infected'],
                    '-o', color='C0', label='Confirmed Cases', markersize=3)
            plt.plot(df_true_plotting_rolling['date'], df_true_plotting_rolling['total_infected'],
                    '-', color='C0', linewidth=.25)
        if 'hospitalised' in which_compartments:
            plt.plot(df_true_plotting['date'], df_true_plotting['hospitalised'],
                    '-o', color='orange', label='Active Cases', markersize=3)
            plt.plot(df_true_plotting_rolling['date'], df_true_plotting_rolling['hospitalised'],
                    '-', color='orange', linewidth=.25)
        if 'recovered' in which_compartments:
            plt.plot(df_true_plotting['date'], df_true_plotting['recovered'],
                    '-o', color='green', label='Recovered Cases', markersize=3)
            plt.plot(df_true_plotting_rolling['date'], df_true_plotting_rolling['recovered'],
                    '-', color='green', linewidth=.25)
        if 'deceased' in which_compartments:
            plt.plot(df_true_plotting['date'], df_true_plotting['deceased'],
                    '-o', color='red', label='Deceased Cases', markersize=3)
            plt.plot(df_true_plotting_rolling['date'], df_true_plotting_rolling['deceased'],
                    '-', color='red', linewidth=.25)

        for i, run in enumerate(results.keys()):
            color_idx = i/len(results.keys())
            df_pred_run = results[run]['results']['df_prediction']
            df_predicted_plotting_run = df_pred_run.loc[df_pred_run['date'].isin(
                df_true_plotting['date']), ['date', 'hospitalised', 'total_infected', 'deceased', 'recovered']]
            if 'total_infected' in which_compartments:
                cmap = mpl.cm.get_cmap('winter')
                color = cmap(color_idx)
                plt.plot(df_predicted_plotting_run['date'], df_predicted_plotting_run['total_infected'],
                        '-.', color=cmap(color_idx))
            if 'hospitalised' in which_compartments:
                cmap = mpl.cm.get_cmap('Wistia')
                plt.plot(df_predicted_plotting_run['date'], df_predicted_plotting_run['hospitalised'],
                        '-.', color=cmap(color_idx))
            if 'recovered' in which_compartments:
                cmap = mpl.cm.get_cmap('summer')
                plt.plot(df_predicted_plotting_run['date'], df_predicted_plotting_run['recovered'],
                        '-.', color=cmap(color_idx))
            if 'deceased' in which_compartments:
                cmap = mpl.cm.get_cmap('autumn')
                plt.plot(df_predicted_plotting_run['date'], df_predicted_plotting_run['deceased'],
                        '-.', color=cmap(color_idx))
        
        ax = plt.gca()
        setup_plt('No of People', yscale='linear')
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
        plt.legend()
        plt.title('{} - ({} {})'.format(description, self.state, self.district))
        # plt.grid()

    def plot_errors(self, compartment='total_infected', description='', results=None):
        results = self.results['results'] if results is None else results
        plt.title('{} - ({} {})'.format(description, self.state, self.district))
        # plot error
        dates = [run_day for run_day in results.keys()]
        errs = [results[run_day]['results']['df_loss'].loc[compartment,'val'] for run_day in results.keys()]
        plt.plot(dates, errs, ls='-', c='crimson',
            label='mape')

        ax = plt.gca()
        setup_plt('MAPE', yscale='linear')
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
        plt.legend()
        # plt.grid()
        return