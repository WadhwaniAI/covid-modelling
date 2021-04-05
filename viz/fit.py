import copy
import math
from datetime import timedelta

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import entropy

from utils.generic.enums.columns import compartments
from utils.generic.stats import get_param_stats, get_loss_stats
from viz.utils import axis_formatter

def plot_buckets(df_prediction, title, which_buckets=None):
    if (which_buckets == None):
        which_buckets = df_prediction.columns.to_list()
        which_buckets.remove('date')
    plt.figure(figsize=(20,35))
    fig,a =  plt.subplots(math.ceil(len(which_buckets)/2),2, figsize=(20,30))
    fig.suptitle(title, fontsize=16)
    col = 0
    for i in range(math.ceil(len(which_buckets)/2)):
        for j in range(2):
            # import pdb; pdb.set_trace()
            if (col >= len(which_buckets)):
                break
            a[i][j].plot(df_prediction['date'], df_prediction[which_buckets[col]], label=which_buckets[col])
            plt.sca(a[i][j])
            plt.xticks(rotation=45)
            plt.ylabel(which_buckets[col])
            plt.legend(loc='best')
            col += 1
    plt.show()


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
                ax.plot(df_true_plotting[compartments['date'].name].to_numpy(), 
                        df_true_plotting[compartment.name].to_numpy(),
                        '-o', color=compartment.color, label='{} (Observed)'.format(compartment.label))
                ax.plot(df_true_plotting_rolling[compartments['date'].name].to_numpy(), 
                        df_true_plotting_rolling[compartment.name].to_numpy(),
                        '-', color=compartment.color, label='{} (Obs RA)'.format(compartment.label))
                ax.plot(df_predicted_plotting[compartments['date'].name].to_numpy(), 
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



def plot_histogram(predictions_dict, fig, axs, weighting='exp', beta=1, plot_lines=False, weighted=True,
                   true_val= None, savefig=False, filename=None, label=None):
    """Plots histograms for all the sampled params for a particular run in a particular fig.
       The ith subplot will have the histogram corresponding to the ith parameter
    Args:
        predictions_dict (dict): predictions_dict for a particular run
        true_val(dict):True values of the parameters if available. 
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
    params_array = predictions_dict['trials']['params']
    params_dict = {param: [param_dict[param] for param_dict in params_array]
                   for param in list(params_array[0].keys())}
    histograms = {}
    for i, param in enumerate(list(params_array[0].keys())):
        if (param == 'gamma'):
            continue
        ax = axs.flat[i]
        if true_val is not None:
            ax.axvline(x=true_val[param],linewidth=2, color='r',label='True value')
        sns.distplot(params_dict[param],norm_hist = True, kde = False,bins= 100,ax=ax)
        y = ax.get_ylim()
        ax.errorbar(x = np.mean(params_dict[param]),y = 0.5*y[1],xerr = np.std(params_dict[param]),fmt='*',capthick=2,capsize=y[1]*.05,color = 'purple',label = 'Mean and std')
        ax.set_title(f'Denisty Plot of parameter {param}')
        ax.set_ylabel('Density')
        ax.legend()
    return histograms



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
    params_array = predictions_dict['trials']['params']
    params = list(params_array[0].keys())
    df_mean_var = pd.DataFrame(columns=list(predictions_dict.keys()),
                               index=pd.MultiIndex.from_product([params,
                                                                ['mean', 'std']]))

    for run in predictions_dict.keys():
        params_array = predictions_dict[run]['trials']['params']
        losses_array = predictions_dict[run]['trials']['losses']
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

def get_losses(model_dict, method='best', start_iter=100):
    if method=='best':
        best_losses = []
        for trial in model_dict['m0']['trials']:
            if len(best_losses) == 0:
                best_losses.append(trial['result']['loss'])
            else:
                best_losses.append(min(best_losses[-1], trial['result']['loss']))
        return best_losses[start_iter:]
    elif method=='aggregate':
        best_losses = np.array([0.0 for i in range(len(model_dict['m0']['trials']))])
        for _, run_dict in model_dict.items():
            min_loss = 1e+5
            for i,trial in enumerate(run_dict['trials']):
                best_losses[i] += min(min_loss,trial['result']['loss'])
                min_loss = min(min_loss,trial['result']['loss'])
        best_losses /= len(model_dict)
        return list(best_losses[start_iter:])
 
def plot_variation_with_iterations(predictions_dict, compare='model', method='best', start_iter=0):
    # compute the total number of graphs => #locations * #models
    ncols = 3
    nrows = len(predictions_dict)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, 
                            figsize=(18, 8*nrows))
    fig.suptitle('Loss Vs #iterations')
    ax_counter = 0
    for tag, tag_dict in predictions_dict.items():
        ax = axs.flat[ax_counter]
        for model, model_dict in tag_dict.items():
            losses = get_losses(model_dict, method=method, start_iter=start_iter)
            ax.plot(range(start_iter, start_iter + len(losses)), losses, label=model)
            plt.sca(ax)
            plt.ylabel('loss')
            plt.xlabel('iteration')
            plt.legend(loc='best')
            plt.title(tag)
        ax_counter += 1
    for i in range(ax_counter,nrows*ncols):
        fig.delaxes(axs.flat[i])
    plt.show()

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

def plot_all_losses(predictions_dict, which_losses=['train', 'val'], method='best_loss_nora', weighting='exp'):
    """Plots mean and variance bar graphs for losses from all the compartments for different (scenario, config). 
       It is assumed that the user provides a dict with 1st layer of keys as the scenarios and 2nd layer 
       of keys as the config file used. For each (scenario, config), the user should provide prediction_dicts
       corresponding to multiple runs and a "method" ('best' or 'ensemble') specifying how to aggregate 
       these runs.
       Each subplot corresponds to one compartment. Each subplot will have a set of bars corresponding
       to each scenario. Within a set, each bar corresponds to a config file used.

    Args:
        predictions_dict (dict): Dict of all predictions in above mentioned format
        which_losses: Which losses have to considered? train or val
        method (str, optional): The method of aggregation of different runs. 
            possible values: 'best_loss_nora', 'best_loss_ra', 'ensemble_loss_ra'
        weighting (str, optional): The weighting function. 
            If 'exp', np.exp(-beta*loss) is the weighting function used. (beta is separate param here)
            If 'inv', 1/loss is used. Else, uniform weighting is used. Defaults to 'exp'.

    Returns:
        mpl.Figure: The matplotlib figure
    """
    loss_wise_stats = {which_loss : {} for which_loss in which_losses}
    num_compartments = 0
    for loc, loc_dict in predictions_dict.items():
        for config, config_dict in loc_dict.items():
            for which_loss in which_losses:
                loss_stats = get_loss_stats(config_dict, which_loss=which_loss, method=method)
                for compartment in loss_stats.columns:
                    if compartment not in loss_wise_stats[which_loss]:
                        loss_wise_stats[which_loss][compartment] = {}
                    if config not in loss_wise_stats[which_loss][compartment]:
                        loss_wise_stats[which_loss][compartment][config] = {'mean':{}, 'std':{}}
                    loss_wise_stats[which_loss][compartment][config]['mean'][loc] = loss_stats[compartment]['mean']
                    loss_wise_stats[which_loss][compartment][config]['std'][loc] = loss_stats[compartment]['std']
                    num_compartments += 1
    
    n_subplots = num_compartments
    ncols = 3
    nrows = math.ceil(n_subplots/ncols)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, 
                            figsize=(18, 8*nrows))
    bar_width = (1-0.2)/num_compartments

    ax_counter=0
    for which_loss in which_losses:
        for compartment, compartment_values in loss_wise_stats[which_loss].items():
            ax = axs.flat[ax_counter]
            mean_vals, std_vals = {},{}
            for m,model in enumerate(compartment_values.keys()):
                mean_vals[model] = compartment_values[model]['mean']
                std_vals[model] = compartment_values[model]['std']
                pos = [m*bar_width+n for n in range(len(mean_vals[model]))]
                ax.bar(pos, mean_vals[model].values(), width=bar_width, align='center', alpha=0.5, label=model)
                ax.errorbar(pos, mean_vals[model].values(), yerr=std_vals[model].values(), fmt='o', color='k')
            plt.sca(ax)
            plt.title(which_loss)
            plt.ylabel(compartment)
            xtick_vals = mean_vals[model].keys()
            plt.xticks(range(len(xtick_vals)), xtick_vals, rotation=45)
            plt.legend(loc='best')
            ax_counter += 1
    for i in range(ax_counter,nrows*ncols):
        fig.delaxes(axs.flat[i])
    return fig

def plot_all_params(predictions_dict, method='best', weighting='exp'):
    """Plots mean and variance bar graphs for all the sampled params for different (scenario, config). 
       It is assumed that the user provides a dict with 1st layer of keys as the scenarios and 2nd layer 
       of keys as the config file used. For each (scenario, config), the user should provide prediction_dicts
       corresponding to multiple runs and a "method" ('best' or 'ensemble') specifying how to aggregate 
       these runs.
       Each subplot corresponds to one parameter. Each subplot will have a set of bars corresponding
       to each scenario. Within a set, each bar corresponds to a config file used.

    Args:
        predictions_dict (dict): Dict of all predictions in above mentioned format
        method (str, optional): The method of aggregation of different runs ('best' or 'ensemble')
        weighting (str, optional): The weighting function. 
            If 'exp', np.exp(-beta*loss) is the weighting function used. (beta is separate param here)
            If 'inv', 1/loss is used. Else, uniform weighting is used. Defaults to 'exp'.

    Returns:
        mpl.Figure: The matplotlib figure
    """
    param_wise_stats = {}
    for loc, loc_dict in predictions_dict.items():
        for config, config_dict in loc_dict.items():
            param_stats = get_param_stats(config_dict, method=method, weighting=weighting)
            for param in param_stats:
                if param not in param_wise_stats:
                    param_wise_stats[param] = {}
                param_wise_stats[param][config] = {'mean':{},'std':{}}
                param_wise_stats[param][config]['mean'][loc] = param_stats[param]['mean']
                param_wise_stats[param][config]['std'][loc] = param_stats[param]['std']

    n_subplots = len(param_wise_stats)
    ncols = 3
    nrows = math.ceil(n_subplots/ncols)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, 
                            figsize=(18, 8*nrows))
    bar_width = (1-0.2)/len(param_wise_stats)

    ax_counter=0
    for param, param_values in param_wise_stats.items():
        ax = axs.flat[ax_counter]
        mean_vals, std_vals = {},{}
        for k,model in enumerate(param_values.keys()):
            mean_vals[model] = param_values[model]['mean']
            std_vals[model] = param_values[model]['std']
            pos = [k*bar_width+j for j in range(len(mean_vals[model]))]
            ax.bar(pos, mean_vals[model].values(), width=bar_width, align='center', alpha=0.5, label=model)
            ax.errorbar(pos, mean_vals[model].values(), yerr=std_vals[model].values(), fmt='o', color='k')
        plt.sca(ax)
        plt.ylabel(param)
        plt.xticks(range(len(mean_vals[model].keys())), mean_vals[model].keys(), rotation=45)
        plt.legend(loc='best')
        ax_counter += 1
    for i in range(ax_counter,nrows*ncols):
        fig.delaxes(axs.flat[i])
    return fig
