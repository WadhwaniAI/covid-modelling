import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pandas.plotting import register_matplotlib_converters
from matplotlib.dates import DateFormatter
import pandas as pd
from datetime import timedelta

import sys
sys.path.append('../..')
from viz import setup_plt

def plot_results(model_params, converged_params, df, train_size, test, predictions, predictdate, testerr,
        file_prefix, val_size, draws=None, yaxis_name=None):
    ycol = model_params['ycol']
    maperr = testerr['mape']
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

def plot_backtesting_results(model_params, df, results, future_days, file_prefix, transform_y=None, dtp=None, axis_name=None):
    ycol = model_params['ycol']
    title = f'{file_prefix} {ycol}' +  ' backtesting'
    # plot predictions against actual
    if axis_name is not None:
        setup_plt(axis_name)
    else:
        setup_plt(ycol)
    plt.yscale("linear")
    plt.title(title.format(model_params['func'].__name__))

    if transform_y is not None:
        df[model_params['ycol']] = transform_y(df[model_params['ycol']], dtp)
    errkey = 'xform_error' if transform_y is not None else 'error'

    # plot predictions
    
    cmap = mpl.cm.get_cmap('winter')
    for i, run_day in enumerate(results.keys()):
        pred_dict = results[run_day]['predictions']
        if transform_y is not None:
            val_preds = transform_y(pred_dict['val_preds'], dtp)
            fit_preds = transform_y(pred_dict['fit_preds'], dtp)
            # fit_preds = transform_y(pred_dict['fit_preds'][-14:], dtp)
        else:
            val_preds = pred_dict['val_preds']
            fit_preds = pred_dict['fit_preds']#[-14:]
        val_dates = pred_dict['val_dates']
        fit_dates = pred_dict['fit_dates']#[-14:]
        
        color = cmap(i/len(results.keys()))
        plt.plot(val_dates, val_preds, ls='dashed', c=color,
            label=f'run day: {run_day}')
        plt.plot(fit_dates, fit_preds, ls='solid', c=color,
            label=f'run day: {run_day}')
        plt.errorbar(val_dates, val_preds,
            yerr=val_preds*results[run_day][errkey]['mape']/100, lw=0.5,
            color='lightcoral', barsabove='False', label='MAPE')
        plt.errorbar(fit_dates, fit_preds,
            yerr=fit_preds*results[run_day][errkey]['mape']/100, lw=0.5,
            color='lightcoral', barsabove='False', label='MAPE')

    # plot data we fit on
    plt.scatter(df[model_params['date']], df[ycol], c='crimson', marker='+', label='data')

    # plt.legend()
    return

def plot_backtesting_errors(model_params, df, start_date, results, file_prefix,
                            scoring='mape', use_xform=True, axis_name=None):
    ycol = model_params['ycol']
    title = f'{file_prefix} {ycol}' +  ' backtesting errors'
    errkey = 'xform_error' if use_xform else 'error'

    setup_plt(scoring)
    plt.yscale("linear")
    plt.title(title)

    # plot error
    dates = [start_date + timedelta(days=run_day) for run_day in results.keys()]
    errs = [results[run_day][errkey][scoring] for run_day in results.keys()]
    plt.plot(dates, errs, ls='-', c='crimson',
        label=scoring)
    plt.legend()
    return

def plot(x, y, title, yaxis_name=None, log=False, scatter=False):
    plt.title(title)
    setup_plt(yaxis_name)
    yscale = 'log' if log else "linear"
    plt.yscale(yscale)

    # plot error
    if scatter:
        plt.scatter(x,y,c='dodgerblue', marker='+')
    else:
        plt.plot(x, y, ls='-', c='crimson')
    return