import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from hyperopt import hp, tpe, fmin, Trials
from tqdm import tqdm
# from tqdm.notebook import tqdm

from collections import OrderedDict
import itertools
from functools import partial
import datetime
from joblib import Parallel, delayed

from models.seir.seir_testing import SEIR_Testing
from data.dataloader import get_global_data, get_indian_data

def loadbbmpdata(filename):
    df = pd.read_csv('../../data/data/{}'.format(filename))

    df.columns = [x if x != 'Result declared on' else 'Result Declaration Date' for x in df.columns]
    df.columns = [x.title() for x in df.columns]

    # Replace all non entries with 01.01.2000
    date_columns = [x for x in df.columns if 'Date' in x]
    for column in date_columns:
        # Replace with 01.01.2000
        df.loc[df[column].isna(), column] = '01.01.2000'
        df.loc[df[column] == '-', column] = '01.01.2000'

    # Convert to pd.datetime
    date_columns = [x for x in df.columns if 'Date' in x]
    for column in date_columns:
        if column != 'Release Date':
            df[column] = df[column].apply(lambda x : x.strip())
            df.loc[:, column] = pd.to_datetime(df.loc[:, column], format='%d.%m.%Y', errors='ignore')
        else:
            df.loc[:, column] = pd.to_datetime(df.loc[:, column], errors='ignore')
            df.loc[:, column] = pd.to_datetime(df.loc[:, column], format='%m.%d.%Y', errors='ignore')
        

    # Convert all 01/01/2000 to NaN
    date_columns = [x for x in df.columns if 'Date' in x]
    for column in date_columns:
        df.loc[df[column].apply(lambda x : x.year) == 2000, column] = np.nan
        

    # Create ICU and Ventilator variable
    df['On ICU'] = df['Current Status'].apply(lambda x : (not pd.isna(x)) and ('icu' in x.lower() or 'ventilator' in x.lower()) )
    df['On Ventilator'] = df['Current Status'].apply(lambda x : (not pd.isna(x)) and ('ventilator' in x.lower()) )

    # Create Exposed, Infectious, Hospitalisation Time variables
    df['Exposed Time'] = np.maximum((df['Date Of Onset Of Symptoms'] - df['Date Of Travel']).astype('timedelta64[D]'), 0)
    df['Infectious Time'] = np.maximum((df['Date Of Hospitalization'] - df['Date Of Onset Of Symptoms']).astype('timedelta64[D]'), 0)
    df['Hospitalisation Time'] = np.maximum((df['Release Date'] - df['Result Declaration Date']).astype('timedelta64[D]'), 0)

    # Fill in missing values

    for i, row in df.iterrows():
        if pd.isna(row['Date Of Hospitalization']):
            if not pd.isna(row['Date Of Onset Of Symptoms']):
                df.loc[i , 'Date Of Hospitalization'] = row['Date Of Onset Of Symptoms'] + datetime.timedelta(days=3)
            else:
                df.loc[i , 'Date Of Hospitalization'] = df.loc[i-1 , 'Date Of Hospitalization']
                
        if pd.isna(row['Date Of Onset Of Symptoms']):   
            df.loc[i , 'Date Of Onset Of Symptoms'] = df.loc[i , 'Date Of Hospitalization'] - datetime.timedelta(days=3)
                
        if pd.isna(row['Date Of Sample Collection']):
            df.loc[i , 'Date Of Sample Collection'] = df.loc[i , 'Date Of Hospitalization']
        
        if pd.isna(row['Result Declaration Date']):
            df.loc[i , 'Result Declaration Date'] = df.loc[i , 'Date Of Hospitalization'] + datetime.timedelta(days=1)

    # Create processed dataframe from bbmp data
    min_values = []
    for column in date_columns:
        min_values.append(np.min(df[column]))
        
    start_date = np.nanmin(np.array(min_values))

    daterange = pd.date_range(start=start_date, end=datetime.datetime.today().date())
    daterange

    df_agg = pd.DataFrame(index=daterange, columns=['Active Infections (Unknown)', 'Hospitalised', 'On ICU', 'On Ventilator', 'Fatalities', 'Total Infected', 'Recovered'])
    df_agg.loc[:, :] = 0
    df_agg.head()

    for i, row in df.iterrows():
            
        df_agg.loc[row['Date Of Onset Of Symptoms']:row['Date Of Hospitalization'], 'Active Infections (Unknown)'] += 1
            
        if not pd.isna(row['Release Date']):
            df_agg.loc[row['Date Of Hospitalization']:row['Release Date'], 'Hospitalised'] += 1
            if row['On ICU']:
                df_agg.loc[row['Date Of Hospitalization']:row['Release Date'], 'On ICU'] += 1
            if row['On Ventilator']:
                df_agg.loc[row['Date Of Hospitalization']:row['Release Date'], 'On Ventilator'] += 1
            
        else:
            df_agg.loc[row['Date Of Hospitalization']:, 'Hospitalised'] += 1
            if row['On ICU']:
                df_agg.loc[row['Date Of Hospitalization']:, 'On ICU'] += 1
            if row['On Ventilator']:
                df_agg.loc[row['Date Of Hospitalization']:, 'On Ventilator'] += 1
            
        df_agg.loc[row['Date Of Hospitalization']:, 'Total Infected'] += 1
        
        if not pd.isna(row['Release Date']):
            df_agg.loc[row['Release Date']:, 'Recovered'] += 1
            
    df_agg.reset_index(inplace=True) 
    df_agg.columns = [x if x != 'index' else 'Date' for x in df_agg.columns]

    df_agg.to_csv('../../data/data/bbmp-processed.csv')
    return df, df_agg