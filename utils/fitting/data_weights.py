import numpy as np
import pandas as pd
import datetime
import copy

class Data_Weights():
    
    def make_weights_df(self, df_district, start_date, end_date, weights):

        df_data_weights = copy.deepcopy(df_district)

        print(df_data_weights)

        indices = ['active', 'total', 'deceased', 'recovered']

        for index in indices:
            df_data_weights[index] = np.ones(len(df_district[index]))

        if start_date != None and end_date != None:

            start_index = df_data_weights.loc[df_data_weights['date'].dt.date == start_date].index[0]
            end_index = df_data_weights.loc[df_data_weights['date'].dt.date == end_date + datetime.timedelta(1)].index[0]

            for index in indices:
                df_data_weights[index].iloc[start_index: end_index] = np.array(weights[index])

        return df_data_weights

    def implement_split(self, df_data_weights, split):

        start_date, end_date, train_period, val_period, test_period = split.values()
        weights_dataframes = {'df_data_weights_train': None, 'df_data_weights_val': None, 'df_data_weights_test': None}
        if start_date is not None and end_date is not None:
            raise ValueError('Both start_date and end_date cannot be specified. Please specify only 1')
        elif start_date is not None:
            if isinstance(start_date, int):
                if start_date < 0:
                    raise ValueError('Please enter a positive value for start_date if entering an integer')
            if isinstance(start_date, datetime.date):
                start_date = df_data_weights.loc[df_data_weights['date'].dt.date == start_date].index[0]

            weights_dataframes['df_data_weights_train'] = df_data_weights.iloc[:start_date + train_period, :]
            weights_dataframes['df_data_weights_val'] = df_data_weights.iloc[start_date + train_period:start_date + train_period + val_period, :]
            weights_dataframes['df_data_weights_test'] = df_data_weights.iloc[start_date + train_period + val_period: \
                            start_date + train_period + val_period + test_period, :]
        else:    
            if end_date is not None:
                if isinstance(end_date, int):
                    if end_date > 0:
                        raise ValueError('Please enter a negative value for end_date if entering an integer')
                if isinstance(end_date, datetime.date):
                    end_date = df_data_weights.loc[df_data_weights['date'].dt.date == end_date].index[0] - len(df_data_weights) + 1
            else:
                end_date = 0  

            weights_dataframes['df_data_weights_test'] = df_data_weights.iloc[len(df_data_weights) - test_period+end_date:end_date, :]
            weights_dataframes['df_data_weights_val'] = df_data_weights.iloc[len(df_data_weights) - (val_period+test_period) +
                            end_date:len(df_data_weights) - test_period+end_date, :]
            weights_dataframes['df_data_weights_train'] = df_data_weights.iloc[:len(df_data_weights) - (val_period+test_period)+end_date, :]

        return weights_dataframes

