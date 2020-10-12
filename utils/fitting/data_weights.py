import numpy as np
import pandas as pd
import datetime
import copy

class Data_Weights():
    
    def make_weights_df(self, df_district, start_date, end_date, weights):

        df_data_weights = copy.deepcopy(df_district)
        
        # print(len(weights['active']))
        # print(end_date-start_date)

        print(df_data_weights)

        indices = ['active', 'total', 'deceased', 'recovered']

        for index in indices:
            df_data_weights[index] = np.ones(len(df_district[index]))

        start_index = df_data_weights.loc[df_data_weights['date'].dt.date == start_date].index[0]
        end_index = df_data_weights.loc[df_data_weights['date'].dt.date == end_date + datetime.timedelta(1)].index[0]

        # print(start_index)
        # print(end_index)

        for index in indices:
            df_data_weights[index].iloc[start_index: end_index] = np.array(weights[index])

        return df_data_weights

    def implement_split(self, df_data_weights, train_period, val_period, test_period, start_date, end_date):
    
        if start_date is not None and end_date is not None:
            raise ValueError('Both start_date and end_date cannot be specified. Please specify only 1')
        elif start_date is not None:
            if isinstance(start_date, int):
                if start_date < 0:
                    raise ValueError('Please enter a positive value for start_date if entering an integer')
            if isinstance(start_date, datetime.date):
                start_date = df_data_weights.loc[df_data_weights['date'].dt.date == start_date].index[0]

            df_data_weights_train = df_data_weights.iloc[:start_date + train_period, :]
            df_data_weights_val = df_data_weights.iloc[start_date + train_period:start_date + train_period + val_period, :]
            df_data_weights_test = df_data_weights.iloc[start_date + train_period + val_period: \
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

            df_data_weights_test = df_data_weights.iloc[len(df_data_weights) - test_period+end_date:end_date, :]
            df_data_weights_val = df_data_weights.iloc[len(df_data_weights) - (val_period+test_period) +
                            end_date:len(df_data_weights) - test_period+end_date, :]
            df_data_weights_train = df_data_weights.iloc[:len(df_data_weights) - (val_period+test_period)+end_date, :]

        return df_data_weights_train, df_data_weights_val, df_data_weights_test

