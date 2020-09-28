import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd
import numpy as np
import copy

from functools import reduce
from scipy.stats import entropy

from utils.generic.enums.columns import *
from main.seir.forecast import _order_trials_by_loss
from viz.utils import axis_formatter

def plot_fit(df_prediction, df_train, df_val, df_district, train_period, state, district,
             which_compartments=['active', 'total'], description='', savepath=None):
    """Helper function for creating plots for the training pipeline

    Arguments:
        df_prediction {pd.DataFrame} -- The prediction dataframe outputted by the model
        df_train {pd.DataFrame} -- The train dataset (with rolling average)
        df_val {pd.DataFrame} -- The val dataset (with rolling average)
        df_train_nora {pd.DataFrame} -- The train dataset (with no rolling average)
        df_val_nora {pd.DataFrame} -- The val dataset (with no rolling average)
        train_period {int} -- Length of train period
        state {str} -- Name of state
        district {str} -- Name of district

    Keyword Arguments:
        which_compartments {list} -- Which buckets to plot (default: {['active', 'total']})
        description {str} -- Additional description for the plots (if any) (default: {''})

    Returns:
        ax -- Matplotlib ax object
    """
    # Create plots
    if isinstance(df_val, pd.DataFrame):
        df_true_plotting_rolling = pd.concat(
            [df_train, df_val], ignore_index=True)
    else:
        df_true_plotting_rolling = df_train
    df_true_plotting = copy.copy(df_district)
    df_predicted_plotting = df_prediction.loc[df_prediction['date'].isin(
        df_true_plotting['date']), ['date']+which_compartments]

    plot_ledger = {
        'base': True,
        'severity': True,
        'bed': True
    }
    for i, key in enumerate(plot_ledger.keys()):
        names = [x.name for x in compartments[key]]
        if np.sum(np.in1d(names, which_compartments)) == 0:
            plot_ledger[key] = False

    n_rows = np.sum(list(plot_ledger.values()))
    fig, axs = plt.subplots(nrows=n_rows, figsize=(12, 10*n_rows))
    fig.suptitle('{} {}, {}'.format(description, district, state))
    i = 0
    for key in plot_ledger.keys():
        if not plot_ledger[key]:
            continue
        if n_rows > 1:
            ax = axs[i]
        else:
            ax = axs
        names = [x.name for x in compartments[key]]
        comp_subset = np.array(which_compartments)[np.in1d(which_compartments, names)]
        legend_elements = []
        for compartment in compartments[key]:
            if compartment.name in comp_subset:
                ax.plot(df_true_plotting[compartments['date'][0].name].to_numpy(), 
                        df_true_plotting[compartment.name].to_numpy(),
                        '-o', color=compartment.color, label='{} (Observed)'.format(compartment.label))
                ax.plot(df_true_plotting_rolling[compartments['date'][0].name].to_numpy(), 
                        df_true_plotting_rolling[compartment.name].to_numpy(),
                        '-', color=compartment.color, label='{} (Obs RA)'.format(compartment.label))
                ax.plot(df_predicted_plotting[compartments['date'][0].name].to_numpy(), 
                        df_predicted_plotting[compartment.name].to_numpy(),
                        '-.', color=compartment.color, label='{} (Predicted)'.format(compartment.label))

                legend_elements.append(
                    Line2D([0], [0], color=compartment.color, label=compartment.label))
    
        ax.axvline(x=df_train.iloc[-train_period, :]['date'], ls=':', color='brown', label='Train starts')
        if isinstance(df_val, pd.DataFrame) and len(df_val) > 0:
            ax.axvline(x=df_val.iloc[0, ]['date'], ls=':', color='black', label='Val starts')
        
        axis_formatter(ax, legend_elements, custom_legend=False)
        i += 1
        
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    return fig


def plot_fit_multiple_preds(predictions_dict, which_fit='m1'):

    df_train = predictions_dict[which_fit]['df_train']
    df_val = predictions_dict[which_fit]['df_val']
    df_true_plotting_rolling = pd.concat([df_train, df_val], ignore_index=True)
    df_true_plotting = predictions_dict[which_fit]['df_district']
    df_prediction = predictions_dict[which_fit]['df_prediction']
    train_period = predictions_dict[which_fit]['which_fit_params']['train_period']

    fig, ax = plt.subplots(figsize=(12, 12))
    for compartment in compartments['base']:
        ax.plot(df_true_plotting[compartments['date'][0].name], df_true_plotting[compartment.name],
                '-o', color=compartment.color, label='{} (Observed)'.format(compartment.label))
        ax.plot(df_true_plotting_rolling[compartments['date'][0].name], df_true_plotting_rolling[compartment.name],
                '-', color=compartment.color, label='{} (Obs RA)'.format(compartment.label))
        ax.plot(df_prediction[compartments['date'][0].name], df_prediction[compartment.name],
                '-.', color=compartment.color, label='{} BO Best (Predicted)'.format(compartment.label))
        try:
            df_prediction_decile50 = predictions_dict[which_fit]['df_prediction_decile50']
            ax.plot(df_prediction_decile50[compartments['date'][0].name], df_prediction_decile50[compartment.name],
                    '--', color=compartment.color, label='{} 50th Decile (Predicted)'.format(compartment.label))
        except:
            print('')

        try:
            df_prediction_gsbo = predictions_dict[which_fit]['df_prediction_gsbo']
            ax.plot(df_prediction_gsbo[compartments['date'][0].name], df_prediction_gsbo[compartment.name],
                    '-x', color=compartment.color, label='{} GS+BO (Predicted)'.format(compartment.label))
        except:
            print('')


        ax.axvline(x=df_train.iloc[-train_period, :]['date'],
                ls=':', color='brown', label='Train starts')
        if isinstance(df_val, pd.DataFrame) and len(df_val) > 0:
            ax.axvline(x=df_val.iloc[0, ]['date'], ls=':',
                    color='black', label='Val starts')

    axis_formatter(ax, None, custom_legend=False)

    plt.tight_layout()
    return fig


def plot_histogram(predictions_dict, fig, axs, weighting='exp', which_fit='m1', plot_lines=False, weighted=True, 
                   savefig=False, filename=None, label=None):
    params_array, losses_array = _order_trials_by_loss(predictions_dict[which_fit])
    params_dict = {param: [param_dict[param] for param_dict in params_array]
                   for param in params_array[0].keys()}
    if weighting == 'exp':
        weights = np.exp(-np.array(losses_array))
    elif weighting == 'inverse':
        weights = 1/np.array(losses_array)
    else:
        weights = np.ones(np.array(losses_array).shape)

    label = which_fit if label is None else label

    histograms = {}
    for i, param in enumerate(params_dict.keys()):
        histograms[param] = {}
        ax = axs.flat[i]
        if plot_lines:
            bar_heights, endpoints = np.histogram(params_dict[param], density=True, bins=20, weights=weights)
            centers = (endpoints[1:] + endpoints[:-1]) / 2
            ax.plot(centers, bar_heights, label=which_fit)
        else:
            if weighted:
                histogram = ax.hist(params_dict[param], density=True, histtype='bar', bins=20, 
                                    weights=weights, label=which_fit, alpha=1)
            else:
                histogram = ax.hist(params_dict[param], density=True, histtype='bar', bins=20, 
                                    label=which_fit, alpha=1)
            bar_heights, endpoints = histogram[0], histogram[1]
            centers = (endpoints[1:] + endpoints[:-1]) / 2
        
        ax.set_title(f'Histogram of parameter {param}')
        ax.set_ylabel('Density')
        ax.legend()
            
        histograms[param]['density'] = bar_heights
        histograms[param]['endpoints'] = endpoints
        histograms[param]['centers'] = centers
        histograms[param]['probability'] = bar_heights*np.mean(np.diff(endpoints))
        
    if savefig:
        fig.savefig(filename)
    return histograms

def plot_all_histograms(predictions_dict, description, weighting='exp'):
    params_array, _ = _order_trials_by_loss(predictions_dict['m1'])

    fig, axs = plt.subplots(nrows=len(params_array[0].keys())//2, ncols=2, 
                            figsize=(18, 6*(len(params_array[0].keys())//2)))
    histograms = {}
    for run in predictions_dict.keys():
        histograms[run] = plot_histogram(predictions_dict, fig, axs, 
                                         weighting=weighting, which_fit=run)

    fig.suptitle(f'Histogram plots for {description}')
    fig.subplots_adjust(top=0.96)
    return fig, axs, histograms


def plot_mean_variance(predictions_dict, description, weighting='exp'):
    params_array, _ = _order_trials_by_loss(predictions_dict['m1'])
    params = list(params_array[0].keys())
    df_mean_var = pd.DataFrame(columns=list(predictions_dict.keys()),
                               index=pd.MultiIndex.from_product([params,
                                                                ['mean', 'std']]))

    for run in predictions_dict.keys():
        params_array, losses_array = _order_trials_by_loss(predictions_dict[run])
        params_dict = {param: [param_dict[param] for param_dict in params_array]
                       for param in params_array[0].keys()}
        if weighting == 'exp':
            weights = np.exp(-np.array(losses_array))
        elif weighting == 'inverse':
            weights = 1/np.array(losses_array)
        else:
            weights = np.ones(np.array(losses_array).shape)
        for param in params_dict.keys():
            mean = np.average(params_dict[param], weights=weights)
            variance = np.average((params_dict[param] - mean)**2, weights=weights)
            df_mean_var.loc[(param, 'mean'), run] = mean
            df_mean_var.loc[(param, 'std'), run] = np.sqrt(variance)

    cmap = plt.get_cmap('plasma')
    fig, axs = plt.subplots(nrows=len(params)//2, ncols=2,
                            figsize=(18, 6*(len(params)//2)))
    for i, param in enumerate(params):
        ax = axs.flat[i]
        ax.bar(np.arange(len(predictions_dict)), df_mean_var.loc[(param, 'mean'), :],
            yerr=df_mean_var.loc[(param, 'std'), :], tick_label=df_mean_var.columns, color=cmap(i/len(params)))
        ax.set_title(f'Mean and variance values for parameter {param}')
        ax.set_ylabel(param)
    fig.suptitle(f'Mean Variance plots for {description}')
    fig.subplots_adjust(top=0.96)
    return fig, axs, df_mean_var


def plot_kl_divergence(histograms_dict, description, cmap='Reds', shared_cmap_axes=True):
    params = histograms_dict['m1'].keys()
    fig, axs = plt.subplots(nrows=len(params)//2, ncols=2,
                            figsize=(18, 6*(len(params)//2)))
    kl_dict_reg = {}
    for i, param in enumerate(params):
        kl_matrix = [[entropy(histograms_dict[run1][param]['probability'],
                              histograms_dict[run2][param]['probability'])
                        for run2 in histograms_dict.keys()] for run1 in histograms_dict.keys()]
        kl_dict_reg[param] = kl_matrix

    all_kls = [kl_dict_reg[key] for key in kl_dict_reg.keys()]
    if shared_cmap_axes:
        vmin, vmax = (np.min(all_kls), np.max(all_kls))
    else:
        vmin, vmax = (None, None)

    for i, param in enumerate(params):
        ax = axs.flat[i]
        sns.heatmap(np.array(kl_dict_reg[param]), annot=True, xticklabels=histograms_dict.keys(), 
                    yticklabels=histograms_dict.keys(), vmin=vmin, vmax=vmax, cmap='Reds', ax=ax)
        ax.set_title(f'KL Divergence matrix of parameter {param}')
    
    fig.suptitle(f'KL divergence heatmaps plots for {description}')
    fig.subplots_adjust(top=0.96)

    return fig, axs, kl_dict_reg



def plot_scatter(mean_var_dict, var_1, var_2, statistical_var='mean'):
    fig, axs = plt.subplots(figsize=(12, 6*len(mean_var_dict)//2), 
                            nrows=len(mean_var_dict)//2, ncols=2)

    for i, (key, df_mean_var) in enumerate(mean_var_dict.items()):
        ax = axs.flat[i]
        var_1_means = list(df_mean_var.loc[(var_1, statistical_var), :])
        var_2_means = list(df_mean_var.loc[(var_2, statistical_var), :])
        ax.scatter(var_1_means, var_2_means)
        ax.set_xlabel(var_1)
        ax.set_ylabel(var_2)
        ax.set_title(f'{key}')
    
    fig.suptitle(f'Scatter plot of {statistical_var} across runs for all locations - {var_1} vs {var_2}')
    fig.subplots_adjust(top=0.96)
    
    return fig, axs