import matplotlib.pyplot as plt
from datetime import timedelta
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import seaborn as sns
from adjustText import adjust_text
import datetime
from copy import copy

from utils.generic.enums import Columns
from viz.utils import axis_formatter


def plot_ptiles(predictions_dict, train_fit='m2', vline=None, which_compartments=[Columns.active], 
                plot_individual_curves=True, log_scale=False, truncate_series=True, 
                left_truncation_buffer=30):
    predictions = copy(predictions_dict[train_fit]['forecasts'])
    try:
        del predictions['best']
    except:
        pass

    df_master = list(predictions.values())[0]
    for df in list(predictions.values())[1:]:
        if isinstance(df, pd.DataFrame):
           df = df.reset_index()
        else:
            df = df['df_prediction']
        df_master = pd.concat([df_master, df], ignore_index=True)
    
    train_period = predictions_dict[train_fit]['run_params']['split']['train_period']
    val_period = predictions_dict[train_fit]['run_params']['split']['val_period']
    val_period = 0 if val_period is None else val_period
    df_true = predictions_dict[train_fit]['df_district']
    if truncate_series:
        df_true = df_true[df_true['date'] >
                          (list(predictions.values())[0]['date'].iloc[0] -
                              timedelta(days=left_truncation_buffer))]
        df_true.reset_index(drop=True, inplace=True)

    plots = {}
    for compartment in which_compartments:
        fig, ax = plt.subplots(figsize=(12, 12))
        texts = []
        ax.plot(df_true[Columns.date.name].to_numpy(), df_true[compartment.name].to_numpy(),
                '-o', color='C0', label=f'{compartment.label} (Observed)')
        if plot_individual_curves == True:
            for i, (ptile, df_prediction) in enumerate(predictions.items()):
                if isinstance(df_prediction, pd.DataFrame):
                    df_prediction = df_prediction.reset_index()
                else:
                    df_prediction = df_prediction['df_prediction']
                sns.lineplot(x=Columns.date.name, y=compartment.name, data=df_prediction,
                            ls='-', label=f'{compartment.label} Percentile :{ptile}')
                texts.append(plt.text(
                    x=df_prediction[Columns.date.name].iloc[-1], 
                    y=df_prediction[compartment.name].iloc[-1], s=ptile))
        else:
            sns.lineplot(x=Columns.date.name, y=compartment.name, data=df_master,
                         ls='-', label=f'{compartment.label}',ci=100)
                
        if vline:
            plt.axvline(datetime.datetime.strptime(vline, '%Y-%m-%d'))
        
        ax.axvline(x=list(predictions.values())[0].iloc[0, :]['date'],
                   ls=':', color='brown', label='Train starts')
        ax.axvline(x=list(predictions.values())[0].iloc[train_period+val_period-1, :]['date'],
                ls=':', color='black', label='Data Last Date')

        ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] + 10)
        adjust_text(texts, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
        axis_formatter(ax, log_scale=log_scale)
        fig.suptitle('Forecast of all deciles for {} '.format(compartment.name), fontsize=16)
        plots[compartment] = fig
    
    return plots


def plot_ptiles_reichlab(df_comb, model, location, target='inc death', plot_true=False, 
                         plot_point=True, plot_individual_curves=True):
    compartment = 'deceased' if 'death' in target else 'total'
    mode = 'incident' if 'inc' in target else 'cumulative'
    compartment = Columns.from_name(compartment)
    df_plot = copy(df_comb.loc[(df_comb['model'] == model) & (
        df_comb['location'] == location), :])
    df_plot = df_plot[[target in x for x in df_plot['target']]]
    fig, ax = plt.subplots(figsize=(12, 12))
    texts = []
    if plot_true:
        df_true = df_plot.groupby('target_end_date').mean().reset_index()
        ax.plot(df_true['target_end_date'].to_numpy(), df_true['true_value'].to_numpy(),
                '--o', color=compartment.color)
    if plot_point:
        df_point = df_plot[df_plot['type'] == 'point']
        ax.plot(df_point['target_end_date'].to_numpy(), df_point['value'].to_numpy(),
                '-o', color='black')

    df_quantiles = df_plot[df_plot['type'] == 'quantile']
    quantiles = df_quantiles.groupby('quantile').sum().index
    if plot_individual_curves:
        for _, qtile in enumerate(quantiles):
            df_qtile = df_quantiles[df_quantiles['quantile']
                                    == qtile].infer_objects()
            label = round(qtile*100) if qtile * \
                100 % 1 < 1e-8 else round(qtile*100, 1)
            sns.lineplot(x='target_end_date', y='value', data=df_qtile, ls='-')
            texts.append(plt.text(
                x=df_qtile['target_end_date'].iloc[-1],
                y=df_qtile['value'].iloc[-1], s=label))
    else:
        sns.lineplot(x=Columns.date.name, y='value', data=df_quantiles,
                     ls='-', label=f'{compartment.label}')

    ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] + 10)
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
    axis_formatter(ax)
    legend_elements = []
    if plot_true:
        legend_elements += [
            Line2D([0], [0], ls='--', marker='o', color=compartment.color,
                   label=f'{mode.title()} {compartment.label} (Observed)')]
    if plot_point:
        legend_elements += [
            Line2D([0], [0], ls='-', marker='o', color='black',
                   label=f'{mode.title()} {compartment.label} Point Forecast')]

    legend_elements += [
        Line2D([0], [0], ls='-', color='blue',
               label=f'{mode.title()} {compartment.label} Percentiles'),
    ]
    ax.legend(handles=legend_elements)
    fig.suptitle('Forecast for {}, {}, {} {}'.format(model, location,
                                                     mode.title(), compartment.label), fontsize=16)
    fig.subplots_adjust(top=0.96)

    return fig, ax


def plot_beta_loss(dict_of_trials):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(list(dict_of_trials.keys()), list(dict_of_trials.values()))
    ax.set_ylabel('Loss value')
    ax.set_xlabel('Beta value')
    ax.set_title('How the beta loss changes with beta')
    return fig, ax


def plot_ptiles_comp(PD, train_fit='m1', vline=None, which_compartments=[Columns.active], 
                plot_individual_curves=True, log_scale=False):
    predictions_mcmc =  PD['MCMC']['uncertainty_forecasts']
    predictions_bo =  PD['BO']['uncertainty_forecasts']
    df_master_mcmc = list(predictions_mcmc.values())[0]['df_prediction']
    for df in list(predictions_mcmc.values())[1:]:
        if isinstance(df, pd.DataFrame):
           df = df.reset_index()
        else:
            df = df['df_prediction']
        df_master_mcmc = pd.concat([df_master_mcmc, df], ignore_index=True)
    df_master_bo = list(predictions_bo.values())[0]['df_prediction']
    for df in list(predictions_bo.values())[1:]:
        if isinstance(df, pd.DataFrame):
           df = df.reset_index()
        else:
            df = df['df_prediction']
        df_master_bo = pd.concat([df_master_bo, df], ignore_index=True)
    
    df_true = PD['MCMC']['m1']['df_district']

    plots = {}
    for compartment in which_compartments:
        fig, ax = plt.subplots(figsize=(12, 12))
        texts = []
        ax.plot(df_true[Columns.date.name].to_numpy(), df_true[compartment.name].to_numpy(),
                '-o', color='C0', label=f'{compartment.label} (Observed)')
        ax.set_xlim([datetime.date(2020, 9, 26), datetime.date(2021, 1, 5)])
        sns.lineplot(x=Columns.date.name, y=compartment.name, data=df_master_mcmc,
                         ls='-', label='MCMC',ci = 100)
        sns.lineplot(x=Columns.date.name, y=compartment.name, data=df_master_bo,
                         ls='-', label='BO',ci=100) 
        if vline:
            plt.axvline(datetime.datetime.strptime(vline, '%Y-%m-%d'))

        ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] + 10)
        adjust_text(texts, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
        axis_formatter(ax, log_scale=log_scale)
        fig.suptitle('Forecast of all deciles for {} '.format(compartment.name), fontsize=16)
        plots[compartment] = fig
    
    return plots