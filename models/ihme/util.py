import numpy as np

import curvefit

import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
from matplotlib.dates import DateFormatter

import sys
sys.path.append('../..')
from models.ihme.population import get_district_population

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

def lograte_to_cumulative(to_transform, population):
    cumulative = np.exp(to_transform) * population
    return cumulative

def rate_to_cumulative(to_transform, population):
    cumulative = to_transform * population
    return cumulative

