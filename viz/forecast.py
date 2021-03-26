import copy
import datetime
from datetime import timedelta

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text

from utils.generic.enums.columns import Columns, compartments
from viz.utils import axis_formatter


def plot_forecast(predictions_dict: dict, fits_to_plot=['best'], log_scale=False, 
                  which_compartments=['active', 'total', 'deceased', 'recovered'], 
                  truncate_series=True, left_truncation_buffer=30, 
                  separate_compartments_separate_ax=False, figsize=(12, 12), **kwargs):
    """Function for plotting forecasts (both best fit and uncertainty deciles)
    Arguments:
        predictions_dict {dict} -- Dict of predictions for a particular district 
        region {tuple} -- Region Name eg : ('Maharashtra', 'Mumbai')
    Keyword Argument
        which_compartments {list} -- Which compartments to plot (default: {['active', 'total', 'deceased', 'recovered']})
        df_prediction {pd.DataFrame} -- DataFrame of predictions (default: {None})
        both_forecasts {bool} -- If true, plot both forecasts (default: {False})
        log_scale {bool} -- If true, y is in log scale (default: {False})
        fileformat {str} -- The format in which the plot will be saved (default: {'eps'})
        error_bars {bool} -- If true, error bars will be plotted (default: {False})
    Returns:
        ax -- Matplotlib ax figure
    """
    legend_title_dict = {}
    deciles = np.sort(np.concatenate(( np.arange(10, 100, 10), np.array([2.5, 5, 95, 97.5] ))))
    for key in deciles:
        legend_title_dict[key] = '{}th Decile'.format(int(key))

    legend_title_dict['best'] = 'Best'
    legend_title_dict['mean'] = 'Mean'
    legend_title_dict['ensemble_mean'] = 'Ensemble Mean'

    linestyles_arr = ['-', '--', '-.', ':', '-x']

    if len(fits_to_plot) > 5:
        raise ValueError('Cannot plot more than 5 forecasts together')

    predictions = []
    for i, _ in enumerate(fits_to_plot):
        predictions.append(predictions_dict['forecasts'][fits_to_plot[i]])
    
    train_period = predictions_dict['run_params']['split']['train_period']
    val_period = predictions_dict['run_params']['split']['val_period']
    val_period = 0 if val_period is None else val_period
    df_true = predictions_dict['df_district']
    if truncate_series:
        df_true = df_true[df_true['date'] > \
                          (predictions[0]['date'].iloc[0] - timedelta(days=left_truncation_buffer))]
        df_true.reset_index(drop=True, inplace=True)

    if separate_compartments_separate_ax:
        fig, axs = plt.subplots(figsize=figsize, nrows=2, ncols=2)
    else:
        fig, axs = plt.subplots(figsize=figsize)


    for i, compartment in enumerate(compartments['base']):
        if separate_compartments_separate_ax:
            ax = axs.flat[i]
        else:
            ax = axs
        if compartment.name in which_compartments:
            ax.plot(df_true[compartments['date'].name], df_true[compartment.name],
                    '-o', ms=2.5, color=compartment.color, label='{} (Observed)'.format(compartment.label))
            for j, df_prediction in enumerate(predictions):
                sns.lineplot(x=compartments['date'].name, y=compartment.name, data=df_prediction,
                             ls='-', color=compartment.color, ax=ax,
                             label='{} ({} Forecast)'.format(compartment.label, legend_title_dict[fits_to_plot[j]]))
                ax.lines[-1].set_linestyle(linestyles_arr[j])

            if separate_compartments_separate_ax:
                ax.axvline(x=predictions[0].iloc[0, :]['date'],
                           ls='--', color='black', label='Training Range')
                ax.axvline(x=predictions[0].iloc[train_period + val_period - 1, :]['date'],
                           ls='--', color='black')
                ax.set_title(compartment.name.title())
                axis_formatter(ax, log_scale=log_scale)
    if not separate_compartments_separate_ax:
        ax.axvline(x=predictions[0].iloc[0, :]['date'],
                   ls='--', color='black', label='Training Range')
        ax.axvline(x=predictions[0].iloc[train_period + val_period - 1, :]['date'],
                   ls='--', color='black')
        axis_formatter(ax, log_scale=log_scale)
    fig.suptitle('Forecast')

    return fig

def plot_forecast_agnostic(df_true, df_prediction, region, log_scale=False,
                           model_name='M2', which_compartments=Columns.CARD_compartments()):
    fig, ax = plt.subplots(figsize=(12, 12))
    for col in Columns.CARD_compartments():
        if col in which_compartments:
            ax.plot(df_true['date'], df_true[col.name],
                '-o', color=col.color, label=f'{col.label} (Observed)')
            sns.lineplot(x="date", y=col.name, data=df_prediction,
                     ls='-', color=col.color, label=f'{col.label} ({model_name} Forecast)')

    axis_formatter(ax, log_scale=log_scale)
    fig.suptitle('Forecast - ({} {})'.format(region[0], region[1]), fontsize=16)

    return fig


def plot_top_k_trials(predictions_dict, k=10, vline=None, log_scale=False,
                      which_compartments=[Columns.active], plot_individual_curves=True,
                      truncate_series=True, left_truncation_buffer=30):
                
    trials_processed = predictions_dict['trials']
    top_k_losses = trials_processed['losses'][:k]
    top_k_params = trials_processed['params'][:k]
    predictions = trials_processed['predictions'][:k]
    
    df_master = predictions[0]
    for i, df in enumerate(predictions[1:]):
        df_master = pd.concat([df_master, df], ignore_index=True)
    df_true = predictions_dict['df_district']
    if truncate_series:
        df_true = df_true[df_true['date'] >
                          (predictions[0]['date'].iloc[0] - timedelta(days=left_truncation_buffer))]
        df_true.reset_index(drop=True, inplace=True)

    train_period = predictions_dict['run_params']['split']['train_period']
    val_period = predictions_dict['run_params']['split']['val_period']
    val_period = 0 if val_period is None else val_period

    plots = {}
    for compartment in which_compartments:
        fig, ax = plt.subplots(figsize=(12, 12))
        texts = []
        ax.plot(df_true[Columns.date.name].to_numpy(), df_true[compartment.name].to_numpy(),
                '-o', color='C0', label=f'{compartment.label} (Observed)')
        if plot_individual_curves:
            for i, df_prediction in enumerate(predictions):
                loss_value = np.around(top_k_losses[i], 2)
                if 'lockdown_R0' in top_k_params[i]:
                    r0 = np.around(top_k_params[i]['lockdown_R0'], 2)
                    sns.lineplot(x=Columns.date.name, y=compartment.name, data=df_prediction,
                            ls='-', label=f'{compartment.label} R0:{r0} Loss:{loss_value}')
                else:
                    beta = np.around(top_k_params[i]['beta'], 2)
                    sns.lineplot(x=Columns.date.name, y=compartment.name, data=df_prediction,
                            ls='-', label=f'{compartment.label} Beta:{beta} Loss:{loss_value}')
                texts.append(plt.text(
                    x=df_prediction[Columns.date.name].iloc[-1], 
                    y=df_prediction[compartment.name].iloc[-1], s=loss_value))
        else:
            sns.lineplot(x=Columns.date.name, y=compartment.name, data=df_master,
                         ls='-', label=f'{compartment.label}')
                
        if vline:
            plt.axvline(datetime.datetime.strptime(vline, '%Y-%m-%d'))
        ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] + 10)
        adjust_text(texts, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
        ax.axvline(x=predictions[0].iloc[0, :]['date'],
                   ls=':', color='brown', label='Train starts')
        ax.axvline(x=predictions[0].iloc[train_period+val_period-1, :]['date'],
                ls=':', color='black', label='Data Last Date')
        axis_formatter(ax, log_scale=log_scale)
        fig.suptitle('Forecast of top {} trials for {} '.format(k, compartment.name), fontsize=16)
        plots[compartment] = fig
    return plots


def plot_r0_multipliers(region_dict, predictions_mul_dict, log_scale=False):
    df_true = region_dict['df_district']
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.plot(df_true['date'], df_true['active'],
        '-o', color='orange', label='Active Cases (Observed)')
    for _, (mul, mul_dict) in enumerate(predictions_mul_dict.items()):
        df_prediction = mul_dict['df_prediction']
        true_r0 = mul_dict['params']['post_lockdown_R0']
        sns.lineplot(x="date", y="active", data=df_prediction,
                    ls='-', label=f'Active Cases ({mul} - R0 {true_r0})')
        plt.text(
            x=df_prediction['date'].iloc[-1],
            y=df_prediction['active'].iloc[-1], s=true_r0
        )
    axis_formatter(ax, log_scale=log_scale)
    state, dist = region_dict['state'], region_dict['dist']
    fig.suptitle(f'Forecast - ({state} {dist})', fontsize=16)
    return fig


def plot_errors_for_lookaheads(error_dict, path=None):
    # error dict format:
    # {'lookahead': lookaheads, 'errors': {'model1': errors, 'model2': errors, ...}}
    lookaheads = np.array(error_dict['lookahead'])
    errors = error_dict['errors']
    fig, ax = plt.subplots(figsize=(10, 10))
    width = 0.2

    for i, key in enumerate(errors):
        ax.bar(x=lookaheads+i*width, height=errors[key], label=key, width=width)
        ax.legend(loc='upper right')
    plt.xlabel('Lookahead (days)')
    plt.ylabel('MAPE')
    if path is not None:
        plt.savefig(path)

    return fig, ax
