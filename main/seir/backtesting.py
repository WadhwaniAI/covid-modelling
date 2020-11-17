import pandas as pd
import time
import sys
sys.path.append('../..')
from main.seir.fitting import data_setup, run_cycle

class SEIRBacktest:
    def __init__(self, state, district, df_district, df_district_raw_data, data_from_tracker):
        self.state = state
        self.district = district
        self.df_district = df_district
        self.df_district_raw_data = df_district_raw_data
        self.data_from_tracker = data_from_tracker
        self.results = None

    def test(self, fit, train_period=7, val_period=7, increment=5, 
        future_days=7, N=1e7, num_evals=1000, pre_lockdown=False,
        initialisation='intermediate',
        which_compartments=['active', 'total', 'deceased', 'recovered']):
        
        val_period = val_period if fit == 'm1' else 0

        runtime_s = time.time()
        start = pd.to_datetime(self.df_district['date']).min()
        end = pd.to_datetime(self.df_district['date']).max()
        print(start, end)
        n_days = (end - start).days + 1 - future_days

        results = {}
        for run_day in range(train_period + val_period, n_days, increment):
            end_date = pd.to_datetime(self.df_district['date'], format='%Y-%m-%d').iloc[run_day+future_days]
            print ("\rbacktesting for", end_date, end="")

            #  TRUNCATE DATA
            df_district_incr = self.df_district[pd.to_datetime(self.df_district['date'], format='%Y-%m-%d') <= end_date]
            df_district_raw_data_incr = self.df_district_raw_data[pd.to_datetime(self.df_district_raw_data['date'], format='%Y-%m-%d') <= end_date]
            observed_dataframes = data_setup(df_district_incr, df_district_raw_data_incr, future_days)
            
            # FIT/PREDICT
            res = run_cycle(
                self.state, self.district, observed_dataframes, data_from_tracker=self.data_from_tracker,
                train_period=train_period, num_evals=num_evals, N=N, 
                which_compartments=which_compartments, initialisation=initialisation
            )

            results[run_day] = res

        runtime = time.time() - runtime_s
        print (runtime)
        df_val = observed_dataframes['df_val']
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
