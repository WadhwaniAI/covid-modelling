import numpy as np
from models.ihme.util import setup_plt
import matplotlib.pyplot as plt
import matplotlib as mpl
from pandas.plotting import register_matplotlib_converters
from matplotlib.dates import DateFormatter
import pandas as pd
from datetime import timedelta

def plot_results(model, train, test, predictions, predictdate, testerr,
        file_prefix, draws=None):
    ycol = model.ycol
    maperr = testerr['mape']
    title = f'{file_prefix} {ycol}' +  ' fit to {}'
    # plot predictions against actual
    setup_plt(ycol)
    plt.yscale("linear")
    plt.title(title.format(model.func.__name__))
    n_data = len(train) + len(test)
    # plot predictions
    plt.plot(predictdate, predictions, ls='-', c='dodgerblue', 
        label='fit: {}: {}'.format(model.func.__name__, model.pipeline.mod.params))
    # plot error bars based on MAPE
    future_x = predictdate[n_data:]
    future_y = predictions[n_data:]
    plt.errorbar(future_x, 
        future_y,
        yerr=future_y*maperr, lw=0.5, color='palegoldenrod', barsabove='False', label='mape')
    if draws is not None:
        # plot error bars based on draws
        plt.errorbar(future_x, 
            future_y, 
            yerr=draws[:,n_data:], lw=0.5, color='lightcoral', barsabove='False', label='draws')
    # plot train test boundary
    plt.axvline(train[model.date].max(), ls=':', c='slategrey', label='train/test boundary')
    # plot data we fit on
    plt.scatter(train[model.date], train[ycol], c='crimson', marker='+', label='data')
    plt.scatter(test[model.date], test[ycol], c='crimson', marker='+')
    if model.smoothing:
        plt.plot(train[model.date], train[model.ycol.split('_smoothed')[0]], c='k', marker='+', label='unsmoothed data')
        plt.plot(test[model.date], test[model.ycol.split('_smoothed')[0]], c='k', marker='+')
    
    plt.legend()
    return

def plot_backtesting_results(model, df, results, future_days, file_prefix, transform_y=None, dtp=None, axis_name=None):
    ycol = model.ycol
    title = f'{file_prefix} {ycol}' +  ' backtesting'
    # plot predictions against actual
    if axis_name is not None:
        setup_plt(axis_name)
    else:
        setup_plt(ycol)
    plt.yscale("linear")
    plt.title(title.format(model.func.__name__))

    if transform_y is not None:
        df[model.ycol] = transform_y(df[model.ycol], dtp)
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
        plt.plot(val_dates, val_preds, ls='-', c=color,
            label=f'run day: {run_day}')
        plt.plot(fit_dates, fit_preds, ls='-', c=color,
            label=f'run day: {run_day}')
        plt.errorbar(val_dates, val_preds,
            yerr=val_preds*results[run_day][errkey]['mape'], lw=0.5,
            color='lightcoral', barsabove='False', label='MAPE')
        plt.errorbar(fit_dates, fit_preds,
            yerr=fit_preds*results[run_day][errkey]['mape'], lw=0.5,
            color='lightcoral', barsabove='False', label='MAPE')

    # plot data we fit on
    plt.scatter(df[model.date], df[ycol], c='crimson', marker='+', label='data')

    # plt.legend()
    return

def plot_backtesting_errors(model, df, start_date, results, file_prefix,
                            scoring='mape', use_xform=True, axis_name=None):
    ycol = model.ycol
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