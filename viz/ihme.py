import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pandas.plotting import register_matplotlib_converters
from matplotlib.dates import DateFormatter
import pandas as pd
from datetime import timedelta

import sys
sys.path.append('../..')
from viz.utils import setup_plt

def plot_ihme_results(model_params, converged_params, df, train_size, test, predictions, predictdate, testerr,
        file_prefix, val_size, draws=None, yaxis_name=None):
    ycol = model_params['ycol']
    maperr = testerr[model_params['ycol']]
    title = f'{file_prefix} {ycol}' +  ' fit to {}'
    # plot predictions against actual
    yaxis_name = yaxis_name if yaxis_name is not None else ycol
    setup_plt(yaxis_name)
    plt.yscale("linear")
    plt.title(title.format(model_params['func'].__name__))
    n_data = train_size + len(test)
    # plot predictions
    plt.plot(predictdate, predictions, ls='-', c='dodgerblue', 
        label='fit: {}: {}'.format(model_params['func'].__name__, converged_params))
    # plot error bars based on MAPE
    future_x = predictdate[n_data:]
    future_y = predictions[n_data:]
    plt.errorbar(future_x, 
        future_y,
        yerr=future_y*maperr/100, lw=0.5, color='palegoldenrod', barsabove='False', label='mape')
    if draws is not None:
        # plot error bars based on draws
        plt.errorbar(future_x, 
            future_y, 
            yerr=draws[:,n_data:], lw=0.5, color='lightcoral', barsabove='False', label='draws')
    # plot boundaries
    plt.axvline(test[model_params['date']].min() - timedelta(days=val_size), ls=':', c='darkorchid', label='train/val + boundary')
    plt.axvline(test[model_params['date']].min(), ls=':', c='slategrey', label='train/test boundary')
    # plot data we fit on
    plt.scatter(df[model_params['date']], df[ycol], c='crimson', marker='+', label='data')
    
    plt.legend()
    return
