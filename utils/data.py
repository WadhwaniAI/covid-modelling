import sys

import numpy as np
import pandas as pd
import yaml

sys.path.append('../..')

from utils.population import get_district_population, get_population
from utils.enums import Columns

# tuples: (district, state, census_area_name(s))
mumbai = 'Mumbai', 'Maharashtra', ['District - Mumbai (23)', 'District - Mumbai Suburban (22)']
amd = 'Ahmedabad', 'Gujarat', ['District - Ahmadabad (07)']
jaipur = 'Jaipur', 'Rajasthan', ['District - Jaipur (12)']
pune = 'Pune', 'Maharashtra', ['District - Pune (25)']
delhi = 'Delhi', 'Delhi', ['State - NCT OF DELHI (07)']
bengaluru = 'Bengaluru', 'Karnataka', ['District - Bangalore (18)', 'District - Bangalore Rural (29)']
bengaluru_urban = 'Bengaluru Urban', 'Karnataka', ['District - Bangalore (18)']

regions = {
    'mumbai': mumbai,
    'ahmedabad': amd,
    'jaipur': jaipur,
    'pune': pune,
    'delhi': delhi,
    'bengaluru': bengaluru,
    'bengaluru urban': bengaluru_urban
}


def get_supported_regions():
    path = '../../data/data/ihme_data/regions_supported.yaml'
    with open(path) as infile:
        regions = yaml.load(infile, Loader=yaml.SafeLoader)
    return regions


def get_daily_vals(df, col):
    return df[col] - df[col].shift(1)


def get_rates(timeseries, region, sub_region=None, area_names=None):
    data = timeseries.set_index('date')
    if area_names is not None:
        total_pop = get_district_population(region, area_names)
    else:
        total_pop = get_population(region, sub_region=sub_region)
    for col in Columns.which_compartments():
        if col.name in data.columns:
            data[f'{col.name}_rate'] = data[col.name] / total_pop
            data[f'log_{col.name}_rate'] = data[f'{col.name}_rate'].apply(lambda x: np.log(x))
    data.loc[:, 'group'] = len(data) * [1.0]
    data.loc[:, 'covs'] = len(data) * [1.0]
    data = data.reset_index()
    data['date'] = pd.to_datetime(data['date'])
    return data, total_pop


def lograte_to_cumulative(to_transform, population):
    cumulative = np.exp(to_transform) * population
    return cumulative


def rate_to_cumulative(to_transform, population):
    cumulative = to_transform * population
    return cumulative
