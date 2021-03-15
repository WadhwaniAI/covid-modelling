import sys

import numpy as np

sys.path.append('../..')


def get_daily_vals(df, col):
    return df[col] - df[col].shift(1)


def lograte_to_cumulative(to_transform, population):
    cumulative = np.exp(to_transform) * population
    return cumulative


def rate_to_cumulative(to_transform, population):
    cumulative = to_transform * population
    return cumulative
