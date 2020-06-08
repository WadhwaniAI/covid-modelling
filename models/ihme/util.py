import numpy as np

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

def lograte_to_cumulative(to_transform, population):
    cumulative = np.exp(to_transform) * population
    return cumulative

def rate_to_cumulative(to_transform, population):
    cumulative = to_transform * population
    return cumulative

