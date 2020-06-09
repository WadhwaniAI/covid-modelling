
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import seaborn as sns
from adjustText import adjust_text
import datetime
import copy

from main.seir.forecast import get_forecast, order_trials, top_k_trials, forecast_k
from utils.enums import Columns, SEIRParams



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


def plot_forecast(predictions_dict: dict, region: tuple, both_forecasts=False, log_scale=False, filename=None,
                  which_compartments=['hospitalised',
                                      'total_infected', 'deceased', 'recovered'],
                  fileformat='eps', error_bars=False):
    """Function for plotting forecasts

    Arguments:
        predictions_dict {dict} -- Dict of predictions for a particular district 
        region {tuple} -- Region Name eg : ('Maharashtra', 'Mumbai')

    Keyword Argument
        which_compartments {list} -- Which compartments to plot (default: {['hospitalised', 'total_infected', 'deceased', 'recovered']})
        both_forecasts {bool} -- If true, plot both forecasts (default: {False})
        log_scale {bool} -- If true, y is in log scale (default: {False})
        filename {str} -- If given, the plot is saved here (default: {None})
        fileformat {str} -- The format in which the plot will be saved (default: {'eps'})
        error_bars {bool} -- If true, error bars will be plotted (default: {False})

    Returns:
        ax -- Matplotlib ax figure
    """
    df_prediction = get_forecast(predictions_dict)
    if both_forecasts:
        df_prediction_m1 = get_forecast(predictions_dict, train_fit='m1')
    df_true = predictions_dict['m1']['df_district']

    if error_bars:
        df_prediction = preprocess_for_error_plot(df_prediction, predictions_dict['m1']['df_loss'],
                                                  which_compartments)
        if both_forecasts:
            df_prediction_m1 = preprocess_for_error_plot(df_prediction_m1, predictions_dict['m1']['df_loss'],
                                                         which_compartments)

    fig, ax = plt.subplots(figsize=(12, 12))

    if 'total_infected' in which_compartments:
        ax.plot(df_true['date'], df_true['total_infected'],
                '-o', color='C0', label='Confirmed Cases (Observed)')
        sns.lineplot(x="date", y="total_infected", data=df_prediction,
                     ls='-', color='C0', label='Confirmed Cases (M2 Forecast)')
        if both_forecasts:
            sns.lineplot(x="date", y="total_infected", data=df_prediction_m1,
                         color='C0', label='Confirmed Cases (M1 Forecast)')
            ax.lines[-1].set_linestyle("--")
    if 'hospitalised' in which_compartments:
        ax.plot(df_true['date'], df_true['hospitalised'],
                '-o', color='orange', label='Active Cases (Observed)')
        sns.lineplot(x="date", y="hospitalised", data=df_prediction,
                     ls='-', color='orange', label='Active Cases (M2 Forecast)')
        if both_forecasts:
            sns.lineplot(x="date", y="hospitalised", data=df_prediction_m1,
                         color='orange', label='Active Cases (M1 Forecast)')
            ax.lines[-1].set_linestyle("--")
    if 'recovered' in which_compartments:
        ax.plot(df_true['date'], df_true['recovered'],
                '-o', color='green', label='Recovered Cases (Observed)')
        sns.lineplot(x="date", y="recovered", data=df_prediction,
                     ls='-', color='green', label='Recovered Cases (M2 Forecast)')
        if both_forecasts:
            sns.lineplot(x="date", y="recovered", data=df_prediction_m1,
                         color='green', label='Recovered Cases (M1 Forecast)')
            ax.lines[-1].set_linestyle("--")
    if 'deceased' in which_compartments:
        ax.plot(df_true['date'], df_true['deceased'],
                '-o', color='red', label='Deceased Cases (Observed)')
        sns.lineplot(x="date", y="deceased", data=df_prediction,
                     ls='-', color='red', label='Deceased Cases (M2 Forecast)')
        if both_forecasts:
            sns.lineplot(x="date", y="deceased", data=df_prediction_m1,
                         color='red', label='Deceased Cases (M1 Forecast)')
            ax.lines[-1].set_linestyle("--")

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.ylabel('No of People', fontsize=16)
    if log_scale:
        plt.yscale('log')
    plt.xlabel('Time', fontsize=16)
    plt.xticks(rotation=45, horizontalalignment='right')
    plt.legend()
    plt.title('Forecast - ({} {})'.format(region[0], region[1]), fontsize=16)
    plt.grid()
    if filename != None:
        plt.savefig(filename, format=fileformat)

    return ax


def plot_trials(predictions_dict, train_fit='m2', k=10,
        predictions=None, losses=None, params=None, vline=None,
        which_compartments=[Columns.active], plot_individual_curves=True):
    if predictions is not None:
        top_k_losses = losses[:k]
        top_k_params = params[:k]
        predictions = predictions[:k]
    else:
        predictions, top_k_losses, top_k_params = forecast_k(predictions_dict, k=k, train_fit=train_fit)
    
    df_master = predictions[0]
    for i, df in enumerate(predictions[1:]):
        df_master = pd.concat([df_master, df], ignore_index=True)
    df_true = predictions_dict[train_fit]['df_district']
    plots = {}
    for compartment in which_compartments:
        fig, ax = plt.subplots(figsize=(12, 12))
        texts = []
        ax.plot(df_true[Columns.date.name], df_true[compartment.name],
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
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.ylabel('No of People', fontsize=16)
        plt.yscale('log')
        plt.xlabel('Time', fontsize=16)
        plt.xticks(rotation=45, horizontalalignment='right')
        plt.legend()
        plt.title('Forecast - ({} {})'.format(predictions_dict['state'], predictions_dict['dist']), fontsize=16)
        plt.grid()
        plots[compartment] = ax
    return plots

def plot_r0_multipliers(region_dict,best_params_dict, predictions_mul_dict, multipliers):
    df_true = region_dict['m2']['df_district']
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.plot(df_true['date'], df_true['hospitalised'],
        '-o', color='orange', label='Active Cases (Observed)')
    # all_plots = {}
    for i, (mul, mul_dict) in enumerate(predictions_mul_dict.items()):
        df_prediction = mul_dict['df_prediction']
        true_r0 = mul_dict['params']['post_lockdown_R0']
        # loss_value = np.around(np.sort(losses_array)[:10][i], 2)
        
        # df_loss = calculate_loss(df_train_nora, df_val_nora, df_predictions, train_period=7,
        #         which_compartments=['hospitalised', 'total_infected', 'deceased', 'recovered'])
        sns.lineplot(x="date", y="hospitalised", data=df_prediction,
                    ls='-', label=f'Active Cases ({mul} - R0 {true_r0})')
        plt.text(
            x=df_prediction['date'].iloc[-1],
            y=df_prediction['hospitalised'].iloc[-1], s=true_r0
        )
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.ylabel('No of People', fontsize=16)
    # plt.yscale('log')
    plt.xticks(rotation=45,horizontalalignment='right')
    plt.xlabel('Time', fontsize=16)
    plt.legend()
    state, dist = region_dict['state'], region_dict['dist']
    plt.title(f'Forecast - ({state} {dist})', fontsize=16)
        # plt.grid()
        # all_plots[mul] = ax
    return ax