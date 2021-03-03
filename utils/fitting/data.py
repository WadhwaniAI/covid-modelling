import sys

import numpy as np
import yaml

sys.path.append('../..')

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


def lograte_to_cumulative(to_transform, population):
    cumulative = np.exp(to_transform) * population
    return cumulative


def rate_to_cumulative(to_transform, population):
    cumulative = to_transform * population
    return cumulative
