
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
from viz.utils import axis_formatter
from data.processing.processing import get_data
from utils.generic.enums.columns import *


def plot_backtest_seir(gt_data_source='athena', preds_source='filename', fname_format='1', filename=None, 
                       predictions_dict=None, which_forecast=80, truncate_pretrain_data=False, 
                       separate_compartments=False):
    
    # Getting gt data
    dataloading_params = {'state': 'Maharashtra', 'district': 'Mumbai'}
    df_true = get_data(gt_data_source, dataloading_params)
    if preds_source == 'filename':
        if fname_format == 'old_output':
            pass
        elif fname_format == 'new_deciles':
            pass
        else:
            raise ValueError('Please give legal fname_format : old_output or new_deciles')
    elif preds_source == 'pickle':
        if predictions_dict is None:
            raise ValueError('Please give a predictions_dict input, current input is None')
        
        df_prediction = copy.copy(
            predictions_dict['m2']['forecasts'][which_forecast])
        df_train = copy.copy(predictions_dict['m2']['df_train'])
        train_period = predictions_dict['m2']['run_params']['split']['train_period']
        if truncate_pretrain_data:
            df_prediction = df_prediction.loc[(df_prediction['date'] > df_train.iloc[-train_period, :]['date']) &
                                            (df_prediction['date'] <= df_true.iloc[-1, :]['date'])]
            df_true = df_true.loc[df_true['date'] >
                                df_train.iloc[-train_period, :]['date']]
            df_prediction.reset_index(inplace=True, drop=True)
            df_true.reset_index(inplace=True, drop=True)

    else:
        raise ValueError('Please give legal preds_source : either filename or pickle')

    if separate_compartments:
        fig, axs = plt.subplots(figsize=(18, 12), nrows=2, ncols=2)
    else:
        fig, ax = plt.subplots(figsize=(12, 12))

    for i, compartment in enumerate(compartments['base']):
        if separate_compartments:
            ax = axs.flat[i]
        ax.plot(df_true[compartments['date'][0].name].to_numpy(),
                df_true[compartment.name].to_numpy(),
                '-o', color=compartment.color, label='{} (Observed)'.format(compartment.label))
        ax.plot(df_prediction[compartments['date'][0].name].to_numpy(),
                df_prediction[compartment.name].to_numpy(),
                '-.', color=compartment.color, label='{} (Predicted)'.format(compartment.label))

    if separate_compartments:
        iterable_axes = axs.flat
    else:
        iterable_axes = [ax]
    for i, ax in enumerate(iterable_axes):
        ax.axvline(x=df_train.iloc[-train_period, :]['date'],
                   ls=':', color='brown', label='Train starts')
        ax.axvline(x=df_train.iloc[-1, :]['date'], ls=':',
                   color='black', label='Last data point seen by model')
        axis_formatter(ax, None, custom_legend=False)

    fig.suptitle(
        f'Predictions of {which_forecast} vs Ground Truth (Unseen Data)')
    plt.tight_layout()


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
