
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import seaborn as sns
from adjustText import adjust_text
import datetime
import copy

from main.seir.forecast import get_forecast, forecast_top_k_trials
from utils.enums import Columns, SEIRParams
from utils.enums.columns import *
from viz.utils import axis_formatter



def preprocess_for_error_plot(df_prediction: pd.DataFrame, df_loss: pd.DataFrame,
                              which_compartments=['hospitalised', 'total_infected', 'deceased', 'recovered']):
    df_temp = copy.copy(df_prediction)
    df_temp.loc[:, which_compartments] = df_prediction.loc[:,
                                                           which_compartments]*(1 - 0.01*df_loss['val'])
    df_prediction = pd.concat([df_prediction, df_temp], ignore_index=True)
    df_temp = copy.copy(df_prediction)
    df_temp.loc[:, which_compartments] = df_prediction.loc[:,
                                                           which_compartments]*(1 + 0.01*df_loss['val'])
    df_prediction = pd.concat([df_prediction, df_temp], ignore_index=True)
    return df_prediction


def plot_forecast(predictions_dict: dict, region: tuple, fits_to_plot=['best'], log_scale=False, filename=None,
                  which_compartments=['hospitalised', 'total_infected', 'deceased', 'recovered'],
                  fileformat='eps', error_bars=False, days=30):
    """Function for plotting forecasts (both best fit and uncertainty deciles)

    Arguments:
        predictions_dict {dict} -- Dict of predictions for a particular district 
        region {tuple} -- Region Name eg : ('Maharashtra', 'Mumbai')

    Keyword Argument
        which_compartments {list} -- Which compartments to plot (default: {['hospitalised', 'total_infected', 'deceased', 'recovered']})
        df_prediction {pd.DataFrame} -- DataFrame of predictions (default: {None})
        both_forecasts {bool} -- If true, plot both forecasts (default: {False})
        log_scale {bool} -- If true, y is in log scale (default: {False})
        filename {str} -- If given, the plot is saved here (default: {None})
        fileformat {str} -- The format in which the plot will be saved (default: {'eps'})
        error_bars {bool} -- If true, error bars will be plotted (default: {False})

    Returns:
        ax -- Matplotlib ax figure
    """


    legend_title_dict = {}
    deciles = np.sort(np.concatenate(( np.arange(10, 100, 10), np.array([2.5, 5, 95, 97.5] ))))
    for key in deciles:
        legend_title_dict[key] = '{}th Decile'.format(int(key))

    legend_title_dict['best'] = 'Best M2'
    legend_title_dict['mean'] = 'Mean'

    legend_title_dict['testing_12'] = 'Testing Becomes 1.2x'
    legend_title_dict['testing_15'] = 'Testing Becomes 1.5x'
    legend_title_dict['testing_18'] = 'Testing Becomes 1.8x'
    legend_title_dict['testing_20'] = 'Testing Becomes 2.0x'

    linestyles_arr = ['-', '--', '-.', ':', '-x']

    if len(fits_to_plot) > 5:
        raise ValueError('Cannot plot more than 5 forecasts together')

    predictions = []
    for i, forecast in enumerate(fits_to_plot):
        predictions.append(predictions_dict['m2']['forecasts'][fits_to_plot[i]])
    
    df_true = predictions_dict['m1']['df_district']

    if error_bars:
        for i, df_prediction in enumerate(predictions):
            predictions[i] = preprocess_for_error_plot(df_prediction, predictions_dict['m1']['df_loss'],
                                                       which_compartments)

    fig, ax = plt.subplots(figsize=(12, 12))

    for compartment in compartments['base']:
        if compartment.name in which_compartments:
            ax.plot(df_true[compartments['date'][0].name], df_true[compartment.name],
                    '-o', color=compartment.color, label='{} (Observed)'.format(compartment.label))
            for i, df_prediction in enumerate(predictions):
                sns.lineplot(x=compartments['date'][0].name, y=compartment.name, data=df_prediction,
                             ls='-', color=compartment.color, 
                             label='{} ({} Forecast)'.format(compartment.label, legend_title_dict[fits_to_plot[i]]))
                ax.lines[-1].set_linestyle(linestyles_arr[i])
    
    axis_formatter(ax, log_scale=log_scale)
    fig.suptitle('Forecast - ({} {})'.format(region[0], region[1]), fontsize=16)
    if filename != None:
        plt.savefig(filename, format=fileformat)

    return fig

def plot_forecast_agnostic(df_true, df_prediction, dist, state, log_scale=False, filename=None,
                           model_name='M2', which_compartments=Columns.which_compartments()):
    fig, ax = plt.subplots(figsize=(12, 12))
    for col in Columns.which_compartments():
        if col in which_compartments:
            ax.plot(df_true['date'], df_true[col.name],
                '-o', color=col.color, label=f'{col.label} (Observed)')
            sns.lineplot(x="date", y=col.name, data=df_prediction,
                     ls='-', color=col.color, label=f'{col.label} ({model_name} Forecast)')

    axis_formatter(ax, log_scale=log_scale)
    fig.suptitle('Forecast - ({} {})'.format(dist, state), fontsize=16)
    if filename != None:
        plt.savefig(filename)

    return fig


def plot_top_k_trials(predictions_dict, train_fit='m2', k=10, trials_processed=None, vline=None, log_scale=False,
                      which_compartments=[Columns.active], plot_individual_curves=True):
                
    if trials_processed is None:
        trials_processed = forecast_top_k_trials(predictions_dict, k=k, train_fit=train_fit)
    top_k_losses = trials_processed['losses'][:k]
    top_k_params = trials_processed['params'][:k]
    predictions = trials_processed['predictions'][:k]
    
    df_master = predictions[0]
    for i, df in enumerate(predictions[1:]):
        df_master = pd.concat([df_master, df], ignore_index=True)
    df_true = predictions_dict[train_fit]['df_district']
    plots = {}
    for compartment in which_compartments:
        fig, ax = plt.subplots(figsize=(12, 12))
        texts = []
        ax.plot(df_true[Columns.date.name].to_numpy(), df_true[compartment.name].to_numpy(),
                '-o', color='C0', label=f'{compartment.label} (Observed)')
        if plot_individual_curves == True:
            for i, df_prediction in enumerate(predictions):
                loss_value = np.around(top_k_losses[i], 2)
                r0 = np.around(top_k_params[i]['lockdown_R0'], 2)
                sns.lineplot(x=Columns.date.name, y=compartment.name, data=df_prediction,
                            ls='-', label=f'{compartment.label} R0:{r0} Loss:{loss_value}')
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
        axis_formatter(ax, log_scale=log_scale)
        fig.suptitle('Forecast - ({} {})'.format(predictions_dict['state'], predictions_dict['dist']), fontsize=16)
        plots[compartment] = fig
    return plots

def plot_r0_multipliers(region_dict,best_params_dict, predictions_mul_dict, multipliers):
    df_true = region_dict['m2']['df_district']
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.plot(df_true['date'], df_true['hospitalised'],
        '-o', color='orange', label='Active Cases (Observed)')
    for i, (mul, mul_dict) in enumerate(predictions_mul_dict.items()):
        df_prediction = mul_dict['df_prediction']
        true_r0 = mul_dict['params']['post_lockdown_R0']
        sns.lineplot(x="date", y="hospitalised", data=df_prediction,
                    ls='-', label=f'Active Cases ({mul} - R0 {true_r0})')
        plt.text(
            x=df_prediction['date'].iloc[-1],
            y=df_prediction['hospitalised'].iloc[-1], s=true_r0
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
