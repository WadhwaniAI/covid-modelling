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
import os
import pdb
import json
import argparse

import pandas as pd
from tqdm import tqdm
from os.path import exists, join, splitext
# from utils.generic.config import read_config
from main.seir.mcmc import MCMC
from utils.fitting.mcmc_utils import predict, get_state
from main.seir.forecast import get_forecast, forecast_all_trials, create_all_trials_csv, create_decile_csv_new
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
        if plot_individual_curves:
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


def plot_ptiles_comp(PD, train_fit='m1', vline=None, compartment=Columns.active, 
                plot_individual_curves=True, log_scale=False,ax = None):
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

    if ax is None:
        fig, ax = plt.subplots(figsize=(24, 24))
    else:
        fig = None
    
    texts = []
    sns.set_color_codes()
    df_true = df_true[(df_true['date'] >= PD['BO']['ensemble_mean_forecast']['df_prediction']['date'][0]) & (df_true['date']<=PD['BO']['ensemble_mean_forecast']['df_prediction']['date'].iloc[-1])]
    ax.plot(df_true[Columns.date.name].to_numpy(), df_true[compartment.name].to_numpy(),
            '--', color=compartment.color, label= 'Observed',lw = 3.5)
    print(compartment.name)
    ax.plot(PD['MCMC']['ensemble_mean_forecast']['df_prediction']['date'],PD['MCMC']['ensemble_mean_forecast']['df_prediction'][compartment.name],c ='tab:blue',lw = 2)
    ax.fill_between(list(predictions_mcmc.values())[0]['df_prediction']['date'],list(predictions_bo.values())[0]['df_prediction'][compartment.name],list(predictions_bo.values())[-1]['df_prediction'][compartment.name],color='tab:orange',alpha = .2,lw = 0.01)
    ax.fill_between(list(predictions_mcmc.values())[0]['df_prediction']['date'],list(predictions_mcmc.values())[0]['df_prediction'][compartment.name],list(predictions_mcmc.values())[-1]['df_prediction'][compartment.name],color='tab:blue',alpha = 0.2,lw = 0.01)
    ax.plot(PD['BO']['ensemble_mean_forecast']['df_prediction']['date'],PD['BO']['ensemble_mean_forecast']['df_prediction'][compartment.name],c ='tab:orange',lw = 2)
    if vline:
        plt.axvline(datetime.datetime.strptime(vline, '%Y-%m-%d'))
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
    axis_formatter(ax, log_scale=log_scale)
    ax.grid(False)
    ax.set_xlabel("")
    if(compartment == Columns.total):
        ax.text(.5, .915, 'Confirmed', horizontalalignment='center', fontsize=18, transform=ax.transAxes, backgroundcolor='white')
    else:
        ax.text(.5, .915, compartment.name.title(), horizontalalignment='center', fontsize=18, transform=ax.transAxes, backgroundcolor='white')
    # ax.set_title(compartment.name.title())
    
    return fig,ax


def plot_chains(mcmc: MCMC):
    """Summary
    
    Args:
        mcmc (MCMC): Description
        out_dir (str): Description
    """
    color = plt.cm.rainbow(np.linspace(0, 1, mcmc.n_chains))
    params = [*mcmc.prior_ranges.keys()]

    for param in params:
        plt.figure(figsize=(20, 20))
        plt.subplot(2,1,1)

        for i, chain in enumerate(mcmc.chains):
            df = pd.DataFrame(chain[0])
            samples = np.array(df[param])
            # plt.scatter(list(range(len(samples))), samples, s=4, c=color[i].reshape(1,-1), label='chain {}'.format(i+1))
            plt.plot(list(range(len(samples))), samples, label='chain {}'.format(i+1))

        plt.xlabel("iterations")
        plt.title("Accepted {} samples".format(param))
        plt.legend()

        plt.subplot(2,1,2)

        for i, chain in enumerate(mcmc.chains):
            df = pd.DataFrame(chain[1])
            try:
                samples = np.array(df[param])
                # plt.scatter(list(range(len(samples))), samples, s=4, c=color[i].reshape(1,-1), label='chain {}'.format(i+1))
                plt.scatter(list(range(len(samples))), samples, s=4, label='chain {}'.format(i+1))
            except:
                continue

        plt.xlabel("iterations")
        plt.title("Rejected {} samples".format(param))
        plt.legend()

    for param in params:
        plt.figure(figsize=(20, 10))
        for i, chain in enumerate(mcmc.chains):
            df = pd.DataFrame(chain[0])
            samples = np.array(df[param])
            mean = np.mean(samples)
            # plt.scatter(list(range(len(samples))), samples, s=4, c=color[i].reshape(1,-1), label='chain {}'.format(i+1))
            sns.kdeplot(np.array(samples), bw=0.005*mean)
        plt.title("Density plot of {} samples".format(param))
        plt.show()

# def plot_comp_CI95(PD):
    
#     predictions_dict_b = PD['BO']
#     predictions_dict_m = PD['MCMC']
#     config_filename1 = 'default.yaml'
#     config_filename2 = 'uncer.yaml'
#     config1 = read_config(config_filename1)
#     config2 = read_config(config_filename2)
#     columns = ['total','active','recovered','deceased' ]
#     fig,axs = plt.subplots(figsize=(12,12),nrows=2,ncols=2)
    
#     for i,col in enumerate(columns):
#         print('Sorting trials by ',col)
#         config1['uncertainty']['uncertainty_params']['sort_trials_by_column'] = Columns.from_name(col)
#         config2['uncertainty']['uncertainty_params']['sort_trials_by_column'] = Columns.from_name(col)
#         uncertainty_args_m = {'predictions_dict': predictions_dict_m, 'fitting_config': config2['fitting'],
#                         'forecast_config': config2['forecast'], **config2['uncertainty']['uncertainty_params']}
#         uncertainty_args_b = {'predictions_dict': predictions_dict_b, 'fitting_config': config1['fitting'],
#                             'forecast_config': config1['forecast'], **config1['uncertainty']['uncertainty_params']}
        
#         predictions_dict_m['m1']['trials_processed'] = forecast_all_trials(predictions_dict_m, train_fit='m1', 
#                                                                             model=config2['fitting']['model'], 
#                                                                             forecast_days=config2['forecast']['forecast_days'])
#         predictions_dict_b['m1']['trials_processed'] = forecast_all_trials(predictions_dict_b, train_fit='m1', 
#                                                                             model=config2['fitting']['model'], 
#                                                                             forecast_days=config2['forecast']['forecast_days'])
#         print(config1['uncertainty']['uncertainty_params']['sort_trials_by_column'])
#         uncertainty_m = config2['uncertainty']['method'](**uncertainty_args_m)
#         predictions_dict_m['uncertainty_forecasts'] = uncertainty_m.get_forecasts()
#         predictions_dict_m['ensemble_mean_forecast'] = uncertainty_m.ensemble_mean_forecast
#         uncertainty_b = config1['uncertainty']['method'](**uncertainty_args_b)
#         predictions_dict_b['uncertainty_forecasts'] = uncertainty_b.get_forecasts()
#         predictions_dict_b['ensemble_mean_forecast'] = uncertainty_b.ensemble_mean_forecast
#         PD_plot= {}
#         try:
#             del predictions_dict_b['m1']['plots']
#         except:
#             continue
#         try:
#             del predictions_dict_m['m1']['plots']
#         except:
#             continue
#         PD_plot['MCMC'] = deepcopy(predictions_dict_m)
#         PD_plot['BO'] = deepcopy(predictions_dict_b)
#         plot_ptiles_comp(PD_plot, compartment=Columns.from_name(col),ax=axs.flat[i])