import datetime
from copy import copy
from datetime import timedelta

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from utils.generic.enums import Columns
from viz.utils import axis_formatter


def plot_ptiles(predictions_dict, train_fit='m2', vline=None, which_compartments=[Columns.active], 
                plot_individual_curves=True, log_scale=False, truncate_series=True, 
                left_truncation_buffer=30, ci_lb=2.5, ci_ub=97.5):
    predictions = copy(predictions_dict[train_fit]['forecasts'])
    try:
        del predictions['best']
    except:
        pass

    df_master = list(predictions.values())[0]
    for df in list(predictions.values())[1:]:
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
        if plot_individual_curves:
            for i, (ptile, df_prediction) in enumerate(predictions.items()):
                sns.lineplot(x=Columns.date.name, y=compartment.name, data=df_prediction,
                            ls='-', label=f'{compartment.label} Percentile :{ptile}')
                texts.append(plt.text(
                    x=df_prediction[Columns.date.name].iloc[-1], 
                    y=df_prediction[compartment.name].iloc[-1], s=ptile))
        else:
            ax.plot(df_master[Columns.date.name], df_master[compartment.name],
                    ls='-', label=f'{compartment.label}')
            ax.fill_between(predictions[ci_lb][Columns.date.name], predictions[ci_lb][compartment.name],
                            predictions[ci_ub][compartment.name], ls='-', label=f'{compartment.label}')
                
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


def plot_ptiles_reichlab(df_comb, model, location, target='inc death', plot_true=False, plot_point=True,
                         plot_individual_curves=True, ci_lb=2.5, ci_ub=97.5, color='C0', ax=None, latex=False):
    compartment = 'deceased' if 'death' in target else 'total'
    mode = 'inc' if 'inc' in target else 'cum'
    compartment = Columns.from_name(compartment)
    df_plot = copy(df_comb.loc[(df_comb['model'] == model) & (
        df_comb['location'] == location), :])
    df_plot = df_plot[[target in x for x in df_plot['target']]]
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))
    else:
        fig = None
    
    if plot_true:
        df_true = df_plot.groupby('target_end_date').mean().reset_index()
        ax.plot(df_true['target_end_date'].to_numpy(), df_true['true_value'].to_numpy(),
                '--o', color=compartment.color)
    if plot_point:
        df_point = df_plot[df_plot['type'] == 'point']
        ax.plot(df_point['target_end_date'].to_numpy(), df_point['forecast_value'].to_numpy(),
                '-o', color=color)
    if latex:
        model = model.replace('_', '\_')
    texts = []
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
        df_ci_lb = df_quantiles[df_quantiles['quantile']
                                == ci_lb*0.01].infer_objects()
        df_ci_ub = df_quantiles[df_quantiles['quantile']
                                == ci_ub*0.01].infer_objects()
        ax.fill_between(df_ci_ub['target_end_date'], df_ci_lb['forecast_value'],
                        df_ci_ub['forecast_value'], color=color, alpha=0.1, label=f'{model} 95% CI')

    ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] + 10)
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
    axis_formatter(ax)
    legend_elements = []
    if plot_true:
        legend_elements += [
            Line2D([0], [0], ls='--', marker='o', color=compartment.color,
                   label=f'{target.title()} (Observed)')]
    if plot_point:
        legend_elements += [
            Line2D([0], [0], ls='-', marker='o', color=color,
                   label=f'{model} {target.title()} Point Forecast')]

    if plot_individual_curves:
        legend_elements += [
            Line2D([0], [0], ls='-', color='blue',
                   label=f'{model} {target.title()} Percentiles'),
        ]
    else:
        legend_elements += [
            Patch(facecolor=color, edgecolor=color, alpha=0.1,
                  label=f'{model} {target.title()} 95% CI'),
        ]
    ax.legend(handles=legend_elements)
    ax.set_title('Forecast for {}, {}, {} {}'.format(model, location,
                                                     mode.title(), compartment.label), fontsize=16)

    return fig, ax


def plot_beta_loss(dict_of_trials):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(list(dict_of_trials.keys()), list(dict_of_trials.values()))
    ax.set_ylabel('Loss value')
    ax.set_xlabel('Beta value')
    ax.set_title('How the beta loss changes with beta')
    return fig, ax
