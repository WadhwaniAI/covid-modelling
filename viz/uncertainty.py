import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import seaborn as sns
from adjustText import adjust_text
import datetime
from copy import copy

from utils.generic.enums import Columns, SEIRParams
from viz.utils import axis_formatter


def plot_ptiles(predictions_dict, train_fit='m2', vline=None, which_compartments=[Columns.active], 
                plot_individual_curves=True, log_scale=False):
    predictions = copy(predictions_dict[train_fit]['forecasts'])
    del predictions['best']

    df_master = list(predictions.values())[0]
    for df in list(predictions.values())[1:]:
        df_master = pd.concat([df_master, df], ignore_index=True)
    
    df_true = predictions_dict[train_fit]['df_district']

    plots = {}
    for compartment in which_compartments:
        fig, ax = plt.subplots(figsize=(12, 12))
        texts = []
        ax.plot(df_true[Columns.date.name].to_numpy(), df_true[compartment.name].to_numpy(),
                '-o', color='C0', label=f'{compartment.label} (Observed)')
        if plot_individual_curves == True:
            for i, (ptile, df_prediction) in enumerate(predictions.items()):
                sns.lineplot(x=Columns.date.name, y=compartment.name, data=df_prediction,
                            ls='-', label=f'{compartment.label} Percentile :{ptile}')
                texts.append(plt.text(
                    x=df_prediction[Columns.date.name].iloc[-1], 
                    y=df_prediction[compartment.name].iloc[-1], s=ptile))
        else:
            sns.lineplot(x=Columns.date.name, y=compartment.name, data=df_master,
                         ls='-', label=f'{compartment.label}')
                
        if vline:
            plt.axvline(datetime.datetime.strptime(vline, '%Y-%m-%d'))

        ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] + 10)
        adjust_text(texts, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
        axis_formatter(ax, log_scale=log_scale)
        fig.suptitle('Forecast of all deciles for {} '.format(compartment.name), fontsize=16)
        plots[compartment] = fig
    
    return plots


def plot_ptiles_reichlab(df_comb, model, location, compartment='deceased', plot_individual_curves=True):

    compartment = Columns.from_name(compartment)
    df_plot = copy(df_comb.loc[(df_comb['model'] == model) & (df_comb['location_name'] == location), :])
    fig, ax = plt.subplots(figsize=(12, 12))
    texts = []
    df_true = df_plot.groupby('target_end_date').mean().reset_index()
    ax.plot(df_true['target_end_date'].to_numpy(), df_true['true_value'].to_numpy(),
            '--o', color=compartment.color, label=f'{compartment.label} (Observed)')
    df_quantiles = df_plot[df_plot['type'] == 'quantile']
    quantiles = df_quantiles.groupby('quantile').sum().index
    if plot_individual_curves:
        for _, qtile in enumerate(quantiles):
            df_qtile = df_quantiles[df_quantiles['quantile'] == qtile]
            sns.lineplot(x=Columns.date.name, y='forecast_value', data=df_qtile,
                         ls='-', label=f'{compartment.label} Percentile :{round(qtile*100)}')
            texts.append(plt.text(
                x=df_qtile[Columns.date.name].iloc[-1], 
                y=df_qtile['forecast_value'].iloc[-1], s=round(qtile*100)))
    else:
        sns.lineplot(x=Columns.date.name, y='forecast_value', data=df_quantiles,
                        ls='-', label=f'{compartment.label}')
            

    ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] + 10)
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
    axis_formatter(ax)
    fig.suptitle('Forecast of all deciles for {}, {}, {}'.format(model, location, compartment.label), fontsize=16)
    
    return fig, ax

def plot_beta_loss(dict_of_trials):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(list(dict_of_trials.keys()), list(dict_of_trials.values()))
    ax.set_ylabel('Loss value')
    ax.set_xlabel('Beta value')
    ax.set_title('How the beta loss changes with beta')
    return fig, ax
