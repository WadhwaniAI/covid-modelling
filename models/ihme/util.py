import numpy as np
import pandas as pd

import sys
sys.path.append('../..')
from models.ihme.population import get_district_population

# tuples: (district, state, census_area_name(s))
mumbai = 'Mumbai', 'Maharashtra', ['District - Mumbai (23)', 'District - Mumbai Suburban (22)']
amd = 'Ahmedabad', 'Gujarat', ['District - Ahmadabad (07)']
jaipur = 'Jaipur', 'Rajasthan', ['District - Jaipur (12)']
pune = 'Pune', 'Maharashtra', ['District - Pune (25)']
delhi = 'Delhi', 'Delhi', ['State - NCT OF DELHI (07)']
bengaluru = 'Bengaluru', 'Karnataka', ['District - Bangalore (18)', 'District - Bangalore Rural (29)']

cities = {
    'mumbai': mumbai,
    'ahmedabad': amd,
    'jaipur': jaipur,
    'pune': pune,
    'delhi': delhi,
    'bengaluru': bengaluru,
}

def get_daily_vals(df, col):
        return df[col] - df[col].shift(1)

def get_mortality(district_timeseries, state, area_names):
    data = district_timeseries.set_index('date')
    district_total_pop = get_district_population(state, area_names)
    data['mortality'] = data['deceased']/district_total_pop
    data[f'log_mortality'] = data['mortality'].apply(np.log)
    data.loc[:,'group'] = len(data) * [ 1.0 ]
    data.loc[:,'covs'] = len(data) * [ 1.0 ]
    data = data.reset_index()
    data['date']= pd.to_datetime(data['date'])
    return data, district_total_pop

def lograte_to_cumulative(to_transform, population):
    cumulative = np.exp(to_transform) * population
    return cumulative

def rate_to_cumulative(to_transform, population):
    cumulative = to_transform * population
    return cumulative

