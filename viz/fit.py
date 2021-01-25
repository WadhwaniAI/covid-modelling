import matplotlib.pyplot as plt
from datetime import timedelta
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd
import numpy as np
import copy

from scipy.stats import entropy

from utils.generic.enums.columns import *
from main.seir.forecast import _order_trials_by_loss
from viz.utils import axis_formatter

def plot_fit(df_prediction, df_train, df_val, df_district, train_period, location_description,
             which_compartments=['active', 'total'], description='', savepath=None, 
             truncate_series=True, left_truncation_buffer=30):
    """Helper function for creating plots for the training pipeline

    Arguments:
        df_prediction {pd.DataFrame} -- The prediction dataframe outputted by the model
        df_train {pd.DataFrame} -- The train dataset (with rolling average)
        df_val {pd.DataFrame} -- The val dataset (with rolling average)
        df_train_nora {pd.DataFrame} -- The train dataset (with no rolling average)
        df_val_nora {pd.DataFrame} -- The val dataset (with no rolling average)
        train_period {int} -- Length of train period
        location_description {str} -- Information about location

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

    if truncate_series:
        df_true_plotting = df_true_plotting[df_true_plotting['date'] > \
            (df_predicted_plotting['date'].iloc[0] - timedelta(days=left_truncation_buffer))]
        df_true_plotting_rolling = df_true_plotting_rolling[df_true_plotting_rolling['date'] > \
            (df_predicted_plotting['date'].iloc[0] - timedelta(days=left_truncation_buffer))]
        df_true_plotting.reset_index(drop=True, inplace=True)
        df_true_plotting_rolling.reset_index(drop=True, inplace=True)

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
    fig.suptitle('{} {}'.format(description, location_description))
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
            ax.axvline(x=df_val.iloc[0, ]['date'],
                       ls=':', color='black', label='Val starts')
            ax.axvline(x=df_val.iloc[-1, ]['date'], ls=':',
                       color='black', label='Train+Val ends')
        else:
            ax.axvline(x=df_train.iloc[-1, ]['date'], ls=':',
                       color='black', label='Train+Val ends')
        
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


def plot_histogram(predictions_dict, fig, axs, weighting='exp', beta=1, plot_lines=False, weighted=True, 
                   savefig=False, filename=None, label=None):
    """Plots histograms for all the sampled params for a particular run in a particular fig.
       The ith subplot will have the histogram corresponding to the ith parameter

    Args:
        predictions_dict (dict): predictions_dict for a particular run
        fig (mpl.Figure): The mpl.Figure to plot in
        axs (mpl.Axes): The mpl.Axes to plot in
        weighting (str, optional): The weighting function.
        If 'exp', np.exp(-beta*loss) is the weighting function used. (beta is separate param here)
        If 'inv', 1/loss is used. Else, uniform weighting is used. Defaults to 'exp'.
        beta (float, optional): beta param for exponential weighting 
        plot_lines (bool, optional): If true line joining top of histogram bars is instead plotted. 
        Defaults to False.
        weighted (bool, optional): If false uniform weighting is applied. Defaults to True.
        savefig (bool, optional): If true the figure is saved. Defaults to False.
        filename (str, optional): if savefig is true what filename to save as. Defaults to None.
        label (str, optional): What is the label of the histogram. Defaults to None.

    Returns:
        dict: a dict of histograms of all the params for a particular run
    """
    params_array, losses_array = _order_trials_by_loss(predictions_dict)
    params_dict = {param: [param_dict[param] for param_dict in params_array]
                   for param in params_array[0].keys()}
    if weighting == 'exp':
        weights = np.exp(-np.array(losses_array))
    elif weighting == 'inverse':
        weights = 1/np.array(losses_array)
    else:
        weights = np.ones(np.array(losses_array).shape)

    histograms = {}
    for i, param in enumerate(params_dict.keys()):
        histograms[param] = {}
        ax = axs.flat[i]
        if plot_lines:
            bar_heights, endpoints = np.histogram(params_dict[param], density=True, bins=20, weights=weights)
            centers = (endpoints[1:] + endpoints[:-1]) / 2
            ax.plot(centers, bar_heights, label=label)
        else:
            if weighted:
                histogram = ax.hist(params_dict[param], density=True, histtype='bar', bins=20, 
                                    weights=weights, label=label, alpha=1)
            else:
                histogram = ax.hist(params_dict[param], density=True, histtype='bar', bins=20, 
                                    label=label, alpha=1)
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

def plot_all_histograms(predictions_dict, description, weighting='exp', beta=1):
    """Plots histograms for all the sampled params for all runs. 
       It is assumed that the user provides a dict of N elements, 
       each corresponding to the predictions_dict of 1 run. 
       It is assumed that each run will have the same set of P parameters
       If N runs are provided, a figure with P subplots is created (in a P//2*2 grid).
       Each subplot corresponds to one parameter 
       The ith subplot will have N histograms with each jth histogram corresponding to the ith 
       parameter in the corresponding jth run

    Args:
        predictions_dict (dict): Dict of all predictions
        description (str): Description of all the N runs given by the user. Used in the suptitle function
        weighting (str, optional): The weighting function. 
        If 'exp', np.exp(-beta*loss) is the weighting function used. (beta is separate param here)
        If 'inv', 1/loss is used. Else, uniform weighting is used. Defaults to 'exp'.
        beta (float, optional): beta param for exponential weighting 

    Returns:
        mpl.Figure, mpl.Axes, pd.DataFrame: The matplotlib figure, matplotlib axes, 
        a dict of histograms of all the params for all the runs
    """
    params_array, _ = _order_trials_by_loss(predictions_dict['m1'])

    fig, axs = plt.subplots(nrows=len(params_array[0].keys())//2, ncols=2, 
                            figsize=(18, 6*(len(params_array[0].keys())//2)))
    histograms = {}
    for run in predictions_dict.keys():
        histograms[run] = plot_histogram(predictions_dict[run], fig, axs, 
                                         weighting=weighting, label=run)

    fig.suptitle(f'Histogram plots for {description}')
    fig.subplots_adjust(top=0.96)
    return fig, axs, histograms


def plot_mean_variance(predictions_dict, description, weighting='exp', beta=1):
    """Plots mean and variance bar graphs for all the sampled params for all runs. 
       It is assumed that the user provides a dict of N elements, 
       each corresponding to the predictions_dict of 1 run. 
       It is assumed that each run will have the same set of P parameters
       If N runs are provided, a figure with P subplots is created (in a P//2*2 grid).
       Each subplot corresponds to one parameter 
       The ith subplot will have N bars with each jth bar corresponding to mean and variance of ith 
       parameter in the corresponding jth run

    Args:
        predictions_dict (dict): Dict of all predictions
        description (str): Description of all the N runs given by the user. Used in the suptitle function
        weighting (str, optional): The weighting function. 
        If 'exp', np.exp(-beta*loss) is the weighting function used. (beta is separate param here)
        If 'inv', 1/loss is used. Else, uniform weighting is used. Defaults to 'exp'.
        beta (float, optional): beta param for exponential weighting 

    Returns:
        mpl.Figure, mpl.Axes, pd.DataFrame: The matplotlib figure, matplotlib axes, 
        a dataframe of mean and variance values for all parameters and all runs
    """
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
            weights = np.exp(-beta*np.array(losses_array))
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
    """Plots KL divergence heatmaps for all the sampled params for all runs. 
       It is assumed that the user provides a dict of N elements, 
       each corresponding to the histogram dict of 1 run (as returned by plot_all_histograms)
       It is assumed that each run will have the same set of P parameters
       If N runs are provided, a figure with P subplots is created (in a P//2*2 grid).
       Each subplot corresponds to one parameter 
       The ith subplot will be an N * N matrix with each element in at (j, k) index is the 
       KL divergence of ith parmeter in the kth run wrt the ith parmeter in the jth run

    Args:
        histograms_dict (dict): dict of histograms as returned by plot_all_histograms
        description (str): description of the plot (used by fig.suptitle)
        cmap (str, optional): matplotlib colormap for heatmap. Defaults to 'Reds'.
        shared_cmap_axes (bool, optional): Whether all colormaps in all subplots
        will have the same axes. Defaults to True.

    Returns:
        mpl.Figure, mpl.Axes, dict: The matplotlib figure, matplotlib axes, 
        a dict of KL divergence matrices for all parameters
    """
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



def plot_scatter(mean_var_dict, var_1, var_2, stat_measure='mean'):
    """Plots a scatter plot between the means of var_1 vs var_2 across all the runs
    Expects an input of mean_var_dict, which is a dict of Q elements, 
    where each element is the output of plot_mean_variance.
    It creates a figure with Q subplots in a Q//2*2 grid
    Each subplot is a scatter plot of means of var_1 vs var_2

    Args:
        mean_var_dict (dict): A dict as described above 
        var_1 (str): Variable on x axis
        var_2 (str): Variable on x axis
        stat_measure (str, optional): Which statistical measure to plot ('mean'/'std'). Defaults to 'mean'.

    Returns:
        mpl.Figure, mpl.Axes: The matplotlib figure, matplotlib axes
    """
    fig, axs = plt.subplots(figsize=(12, 6*len(mean_var_dict)//2), 
                            nrows=len(mean_var_dict)//2, ncols=2)

    for i, (key, df_mean_var) in enumerate(mean_var_dict.items()):
        ax = axs.flat[i]
        var_1_means = list(df_mean_var.loc[(var_1, stat_measure), :])
        var_2_means = list(df_mean_var.loc[(var_2, stat_measure), :])
        ax.scatter(var_1_means, var_2_means)
        ax.set_xlabel(var_1)
        ax.set_ylabel(var_2)
        ax.set_title(f'{key}')
    
    fig.suptitle(f'Scatter plot of {stat_measure} across runs for all locations - {var_1} vs {var_2}')
    fig.subplots_adjust(top=0.96)
    
    return fig, axs


def plot_heatmap_distribution_sigmas(mean_var_dict, stat_measure='mean', cmap='Reds'):
    """Plots a heatmap of sigma/mu where sigma/mu will be defined as follows : 
    Suppose there are Q locations for which we are doing distribution analyis,
    Each time we run the config N times, and there are P parameters.
    There will therefore be Q, P, N triplets for which we have a mean, variance value
    For a particular location q, and parameter p, we can calculate mean of means. 
    That would be X_double_bar. We can also calculate sigma of means (X_bar_sigma).
    sigma_by_mu would be defined as X_bar_sigma/X_double_bar.

    We would have a Q*P matrix that is the heatmap that we plot.
    We can do the same process with either the mean or the variance

    Args:
        mean_var_dict (dict): a dict of Q elements, 
        where each element is the output of plot_mean_variance.
        stat_measure (str, optional): Which stat measure to calculate sigma_by_mu over. 
        Defaults to 'mean'.
        cmap (str, optional): mpl colormap to use. Defaults to 'Reds'.

    Returns:
        [type]: [description]
    """
    params = [x[0] for x in list(mean_var_dict.values())[0].loc[(slice(None), [stat_measure]), :].index]

    columns = pd.MultiIndex.from_product(
        [params, ['X_double_bar', 'X_bar_sigma', 'sigma_by_mu']])
    df_comparison = pd.DataFrame(columns=columns, index=mean_var_dict.keys())

    for key, df_loc in mean_var_dict.items():
        X_double_bar = df_loc.loc[(slice(None), [stat_measure]), :].mean(axis=1).values
        X_bar_sigma = df_loc.loc[(slice(None), [stat_measure]), :].std(axis=1).values

        df_comparison.loc[key, (slice(None), ['X_double_bar'])] = X_double_bar
        df_comparison.loc[key, (slice(None), ['X_bar_sigma'])] = X_bar_sigma
        df_comparison.loc[key, (slice(None), ['sigma_by_mu'])] = X_bar_sigma/X_double_bar

    df_sigma_mu = df_comparison.loc[:, (slice(None), ['sigma_by_mu'])]

    fig, ax = plt.subplots(figsize=(10, 16))
    sns.heatmap(df_sigma_mu.values.astype(float), annot=True, cmap=cmap, ax=ax,
                xticklabels=[x[0] for x in df_sigma_mu.columns], 
                yticklabels=[f'{x[0]}, {x[1]}' for x in df_sigma_mu.index])
    ax.set_title(f'Heatmap of sigma/mu values for all the {stat_measure}s calculated across all the identical runs')

    return fig, df_comparison
