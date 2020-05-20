import sys
import json
import numpy as np
import pandas as pd

from datetime import timedelta
from copy import deepcopy
import curvefit

import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
from matplotlib.dates import DateFormatter
from sklearn.metrics import mean_squared_error, mean_squared_log_error

from . import data as dataloader
sys.path.append('../..')
from utils.population import get_district_population

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

def evaluate(y_true, y_pred):
    err = {}
    err['mape'] = mape(y_true, y_pred)
    err['rmse'] = rmse(y_true, y_pred)
    try:
        err['rmsle'] = rmsle(y_true, y_pred)
    except:
        err['rmsle'] = None
    return err

def smooth(y, smoothing_window):
    box = np.ones(smoothing_window)/smoothing_window
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def get_daily_vals(df, col):
        return df[col] - df[col].shift(1)

def get_mortality(district_timeseries, state, area_names):
    data = district_timeseries.set_index('date')
    district_total_pop = get_district_population(state, area_names)
    data['mortality'] = data['deceased']/district_total_pop
    data[f'log_mortality'] = data['mortality'].apply(np.log)
    return data.reset_index(), district_total_pop
    
def setup_plt(ycol):
    sns.set()
    register_matplotlib_converters()
    plt.yscale("log")
    plt.gca().xaxis.set_major_formatter(DateFormatter("%d.%m"))
    plt.xlabel("Date")
    plt.ylabel(ycol)
