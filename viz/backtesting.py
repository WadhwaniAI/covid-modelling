
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
import pandas as pd
import numpy as np
import seaborn as sns
from adjustText import adjust_text
import datetime
import copy
from datetime import timedelta

from viz.utils import setup_plt
from utils.generic.enums import Columns, SEIRParams
from utils.generic.enums.columns import *

def plot_backtest(results, data, dist, which_compartments=Columns.which_compartments(), 
                  scoring='mape', dtp=None, axis_name='No. People', savepath=None):
    title = f'{dist}' +  ' backtesting'
    # plot predictions against actual
    setup_plt(axis_name)
    plt.yscale("linear")
    plt.title(title)
    def div(series):
        if scoring=='mape':
            return series/100
        return series
    
    fig, ax = plt.subplots(figsize=(12, 12))
    # plot predictions
    cmap = mpl.cm.get_cmap('winter')
    for col in which_compartments:
        for i, run_day in enumerate(results.keys()):
            run_dict = results[run_day]
            preds = run_dict['df_prediction'].set_index('date')
            val_dates = run_dict['df_val']['date'] if run_dict['df_val'] is not None and len(run_dict['df_val']) > 0 else None
            errkey = 'val' if val_dates is not None else 'train'
            fit_dates = [n for n in run_dict['df_train']['date'] if n in preds.index]
            # fit_dates = run_dict['df_train']['date']
            
            color = cmap(i/len(results.keys()))
            ax.plot(fit_dates, preds.loc[fit_dates, col.name], ls='solid', c=color)
            ax.errorbar(fit_dates, preds.loc[fit_dates, col.name],
                yerr=preds.loc[fit_dates, col.name]*(div(run_dict['df_loss'].loc[col.name, errkey])), lw=0.5,
                color='lightcoral', barsabove='False', label=scoring)
            if val_dates is not None:
                ax.plot(val_dates, preds.loc[val_dates, col.name], ls='dashed', c=color,
                    label=f'run day: {run_day}')
                ax.errorbar(val_dates, preds.loc[val_dates, col.name],
                    yerr=preds.loc[val_dates, col.name]*(div(run_dict['df_loss'].loc[col.name, errkey])), lw=0.5,
                    color='lightcoral', barsabove='False', label=scoring)
                
            
        # plot data we fit on
        ax.scatter(data['date'].values, data[col.name].values, c='crimson', marker='+', label='data')
        plt.text(x=data['date'].iloc[-1], y=data[col.name].iloc[-1], s=col.name)
        
    # plt.legend()
    if savepath is not None:
        plt.savefig(savepath)
        plt.clf()
    return

def plot_backtest_errors(results, data, file_prefix, which_compartments=Columns.which_compartments(), 
                         scoring='mape', savepath=None):
    start = data['date'].min()
    
    title = f'{file_prefix}' +  ' backtesting errors'
    errkey = 'val'

    setup_plt(scoring)
    plt.yscale("linear")
    plt.title(title)

    # plot error
    for col in which_compartments:
        ycol = col.name
        dates = [start + timedelta(days=run_day) for run_day in results.keys()]
        errs = [results[run_day]['df_loss'].loc[ycol, errkey] for run_day in results.keys()]
        plt.plot(dates, errs, ls='-', c='crimson',
            label=scoring)
    plt.legend()
    if savepath is not None:
        plt.savefig(savepath)
        plt.clf()
    return