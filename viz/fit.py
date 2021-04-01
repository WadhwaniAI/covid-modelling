import matplotlib.pyplot as plt
from datetime import timedelta
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd
import numpy as np
import copy
import math

from scipy.stats import entropy

from utils.generic.enums.columns import *
from utils.generic.stats import *
from main.seir.forecast import _order_trials_by_loss
from viz.utils import axis_formatter

def plot_fit(df_prediction, df_train, df_val, df_district, train_period, location_description,
             which_compartments=['active', 'total'], description='', savepath=None, 
             plotting_config={}):
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

    if plotting_config['truncate_series']:
        df_true_plotting = df_true_plotting[df_true_plotting['date'] > \
            (df_predicted_plotting['date'].iloc[0] - \
                timedelta(days=plotting_config['left_truncation_buffer']))]
        df_true_plotting_rolling = df_true_plotting_rolling[df_true_plotting_rolling['date'] > \
            (df_predicted_plotting['date'].iloc[0] - \
                timedelta(days=plotting_config['left_truncation_buffer']))]
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


def plot_comp_density_plots(predictions_dict,fig,axs):
    params_dict_mcmc = {}
    params_dict_bo = {}
    params = list(predictions_dict['m0']['BO']['m1']['best_params'].keys())
    loss_arr = []
    for  param in params:
        params_dict_bo[param] = []
        params_dict_mcmc[param] = []
    for run_number,run_dict in predictions_dict.items():
        params_array_mcmc, loss_mcmc = _order_trials_by_loss(run_dict['MCMC']['m1'])
        params_array_bo, loss_bo = _order_trials_by_loss(run_dict['BO']['m1'])
        loss_arr.extend(loss_bo)
        for param_set in params_array_mcmc:
            for param in params:
                params_dict_mcmc[param].append(param_set[param])
        for param_set in params_array_bo:
            for param in params:
                params_dict_bo[param].append(param_set[param])
    
    weights = np.exp(-1*np.array(loss_arr))
    W =  weights / np.sum(weights)
    params_dict_bo['W'] = W
    latex  = {"lockdown_R0": r'$\mathcal{R}_0$',
    "T_inc": r"$T_{\rm inc}$",
    "T_inf": r"$T_{\rm inf}$",
    "T_recov": r"$T_{\rm recov}$",
    "T_recov_fatal": r"$T_{\rm fatal}$" ,
    "P_fatal": r"$P_{\rm fatal}$",
    "E_hosp_ratio": r"$E_{\rm active\_ratio}$",
    "I_hosp_ratio": r"$I_{\rm active\_ratio}$"}
    for i, param in enumerate(params_dict_mcmc.keys()):
        if (param == 'gamma'):
            continue
        ax = axs.flat[i]
        a = min(params_dict_mcmc[param])
        b = max(params_dict_mcmc[param])
        sns.distplot(params_dict_mcmc[param], hist = True,bins= 100,hist_kws={'weights':np.ones(len(W))/len(W),"range":[a,b],"alpha":0.5},color = 'tab:blue',kde = False,label = ['MCMC'] ,ax=ax)
        sns.distplot(params_dict_bo[param], hist = True,bins= 100,hist_kws={'weights':W,"range":[a,b],"alpha":0.5},color = 'tab:orange',kde = False,label = ['ABMA'] ,ax=ax)
        ax.set_xlabel(latex[param],fontsize = 30)
        # ax.text(.5, .85, latex[param], horizontalalignment='center', fontsize=30, transform=ax.transAxes, backgroundcolor='white')
        if(i%3==0):
            ax.set_ylabel('Density')
        if(param == 'P_fatal'):
            ax.legend(loc = 'upper right')
    return fig,axs
    #     params_array_mcmc, loss_mcmc = _order_trials_by_loss(predictions_dict['MCMC']['m1'])
    # params_dict_mcmc = {param: [param_dict[param] for param_dict in params_array_mcmc]
    #                for param in params_array_mcmc[0].keys()}
    # params_array_bo, loss_bo = _order_trials_by_loss(predictions_dict['BO']['m1'])
    # params_dict_bo = {param: [param_dict[param] for param_dict in params_array_bo]
    #                for param in params_array_bo[0].keys()}
    # for i, param in enumerate(params_dict_mcmc.keys()):
    #     if (param == 'gamma'):
    #         continue
    #     ax = axs.flat[i]
    #     sns.distplot(params_dict_mcmc[param], hist = True, kde = False,
    #              kde_kws = {'shade': True, 'linewidth': 3},label = 'MCMC' ,ax=ax)
    #     sns.distplot(params_dict_bo[param], hist = True, kde = False,
    #              kde_kws = {'shade': True, 'linewidth': 3},label = 'BO', ax=ax)

    #     ax.set_title(f'Denisty Plot of parameter {param}')
    #     ax.set_ylabel('Density')
    #     ax.legend()
    # return fig,axs
    
def plot_log_density(predictions_dict, arr, true_val, fig, axs, weighting='exp', beta=1, 
                     plot_lines=False, weighted=True, savefig=False, filename=None, label=None):
    params_array,_ = _order_trials_by_loss(predictions_dict)
    
    params_array = predictions_dict['trials']['params']
    params_dict = {param: [param_dict[param] for param_dict in params_array]
                   for param in arr}
    histograms = {}
    for i, param in enumerate(arr):
        if (param == 'gamma'):
            continue
        ax = axs.flat[i]
        ax.axvline(x=true_val[param],linewidth=1, color='r',label='True value',ls = '--')
        density,ranges = np.histogram(params_dict[param],bins = 50,range = [0.001,2*true_val[param]],density = True)
        density[density == 0] = 0.000001
        l_density = -1 *np.log(density)
        ad_ranges = [(ranges[i]+ranges[i+1])/2 for i in np.arange(len(ranges) -1 )]
        ax.plot(ad_ranges,l_density)
        # ax.set_ylim(-3,8)
        ax.set_title(param)
        ax.legend()
    return histograms

def plot_histogram(predictions_dict, arr, true_val, fig, axs, weighting='exp', beta=1, 
                   plot_lines=False, weighted=True, savefig=False, filename=None, label=None):
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
    param_range = {"beta": [0, 1],
        "T_inc": [1, 100],
        "T_inf": [1, 100],
        "T_recov": [1, 100],
        "T_recov_fatal": [1, 100],
        "P_fatal": [0, 1],
        'E_hosp_ratio': [0, 5],
        'I_hosp_ratio': [0, 5]}
    params_array,_ = _order_trials_by_loss(predictions_dict)
    params_dict = {param: [param_dict[param] for param_dict in params_array]
                   for param in arr}
    histograms = {}
    for i, param in enumerate(arr):
        if (param == 'gamma'):
            continue
        ax = axs.flat[i]
        ax.axvline(x=true_val[param],linewidth=2, color='r',label='True value')
        sns.distplot(params_dict[param],norm_hist = True, kde = False,bins= 100,ax=ax)
        y = ax.get_ylim()
        ax.set_xlim(param_range[param][0],param_range[param][1])
        ax.errorbar(x = np.mean(params_dict[param]),y = 0.5*y[1],xerr = np.std(params_dict[param]),fmt='*',capthick=2,capsize=y[1]*.05,color = 'purple',label = 'Mean and std')
        if param == 'T_recov_fatal':
            ax.set_title(f'Denisty Plot of parameter T_Death')
        else:
            ax.set_title(f'Denisty Plot of parameter {param}')
        ax.set_ylabel('Density')
        ax.legend()
    return histograms

def plot_2_histogram(SD, key1, key2, arr, true_val, fig, axs, weighting='exp', beta=1, 
                     plot_lines=False, weighted=True, savefig=False, filename=None, label=None):
    
    param_range = {"lockdown_R0": [0.5, 2],
        "T_inc": [0, 15],
        "T_inf": [0, 15],
        "T_recov": [0, 40],
        "T_recov_fatal": [0, 40],
        "P_fatal": [0, 0.5],
        'E_hosp_ratio': [0, 2],
        'I_hosp_ratio': [0, 2]}
    # cmap = {
    #     'exp0': 'y',
    #     'exp1':'b',
    #     'exp2':'r',
    #     'exp3':'g',
    #     'exp4':'orchid',
    #     'exp5':'darkviolet',
    #     'exp6':'deeppink'
    # }
    cmap = {
        'exp0': 'y',
        'exp1':'b',
        'exp2':'r',
        'exp3':'g',
        'exp4':'orchid',
        'exp5':'saddlebrown',
        'exp6':'forestgreen'
    }
    latex  = {"lockdown_R0": r'$\mathcal{R}_0$',
    "T_inc": r"$T_{\rm inc}$",
    "T_inf": r"$T_{\rm inf}$",
    "T_recov": r"$T_{\rm recov}$",
    "T_recov_fatal": r"$T_{\rm fatal}$" ,
    "P_fatal": r"$P_{\rm fatal}$",
    "E_hosp_ratio": r"$E_{\rm active\_ratio}$",
    "I_hosp_ratio": r"$I_{\rm active\_ratio}$"}
    params_array1,_ = _order_trials_by_loss(SD[key1])
    params_dict1 = {param: [param_dict[param] for param_dict in params_array1]
                   for param in arr}
    params_array2,_ = _order_trials_by_loss(SD[key2])
    params_dict2 = {param: [param_dict[param] for param_dict in params_array2]
                   for param in arr}
    params_to_plot = {
        'exp0': arr,
        'exp1':[i for i in arr if i not in ['T_recov_fatal']],
        'exp2':[i for i in arr if i not in ['T_recov_fatal','T_inf']],
        'exp3':[i for i in arr if i not in ['T_recov_fatal','T_inc','T_inf']],
        'exp4':[i for i in arr if i not in ['T_recov_fatal','T_inc','T_inf','T_recov']],
        'exp5': arr,
        'exp6': arr
    }

    histograms = {}
    for i, param in enumerate(arr):
        if (param == 'gamma'):
            continue
        ax = axs.flat[i]
        ax.axvline(x=true_val[param],linewidth=2, color='black',label='True value',ls = '--')
        if(param in params_to_plot[key1]):
            # ax.axvline(x=np.mean(params_dict1[param]),linewidth=3,ls = '--', color=cmap[key1],label='Mean '+ key1)
            sns.distplot(params_dict1[param],norm_hist = True, kde = False,bins= 80, 
                         ax=ax,color = cmap[key1],label = 'Constrained', 
                         hist_kws={"alpha":0.55,"range":(param_range[param][0],param_range[param][1])})
        if(param in params_to_plot[key2]):
            # ax.axvline(x=np.mean(params_dict2[param]),linewidth=3,ls='--', color=cmap[key2],label='Mean '+key2)
            sns.distplot(params_dict2[param],norm_hist = True, kde = False, bins= 80, 
                         ax=ax,color = cmap[key2],label = "Unconstrained",
                         hist_kws={"alpha":0.4,'range':(param_range[param][0],param_range[param][1])})
        # y = ax.get_ylim()
        # ax.set_xlim(param_range[param][0],param_range[param][1])
        # ax.errorbar(x = np.mean(params_dict[param]),y = 0.5*y[1],xerr = np.std(params_dict[param]),fmt='*',capthick=2,capsize=y[1]*.05,color = 'purple',label = 'Mean and std')
        # ax.text(.5, .85, latex[param], horizontalalignment='center', fontsize=20, transform=ax.transAxes, backgroundcolor='white')
        ax.set_xlabel(latex[param],fontsize = 23)

        if(param == 'I_hosp_ratio' or param == 'E_hosp_ratio'):
            ax.set_ylabel('Density')
        if(param =='T_inc'):
            ax.legend(prop={"size":15},loc ='upper right')
    return histograms

def plot_all_histograms(SD, arr, true_val, fig, axs, weighting='exp', beta=1, plot_lines=False, weighted=True, 
                        savefig=False, filename=None, label=None):
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
    param_range = {"lockdown_R0": [0.1, 9.5],
        "T_inc": [1, 100],
        "T_inf": [1, 100],
        "T_recov": [1, 250],
        "T_recov_fatal": [1, 250],
        "P_fatal": [0, 1],
        'E_hosp_ratio': [0, 5],
        'I_hosp_ratio': [0, 5]}
    params_to_plot = {
        'exp0': arr,
        'exp1':[i for i in arr if i not in ['T_recov_fatal']],
        'exp2':[i for i in arr if i not in ['T_recov_fatal','T_inf']],
        'exp3':[i for i in arr if i not in ['T_recov_fatal','T_inc','T_inf']],
        'exp4':[i for i in arr if i not in ['T_recov_fatal','T_inc','T_inf','T_recov']]
    }
    for j,predictions_dict in SD.items():
        params_array, _ = _order_trials_by_loss(predictions_dict)
        params_dict = {param: [param_dict[param] for param_dict in params_array]
                    for param in arr}
        for i, param in enumerate(arr):
            if (param == 'gamma'):
                continue
            if(param  not in params_to_plot[j]):
                continue 
            ax = axs.flat[i]
            if(j == 'exp0'):
                ax.axvline(x=true_val[param],linewidth=2, color='r',label='True value')
            sns.distplot(params_dict[param],norm_hist = True, kde = False,bins= 100,ax=ax,label = j)
            ax.set_xlim(param_range[param][0],param_range[param][1])
            if param == 'T_recov_fatal':
                ax.set_title(f'Denisty Plot of parameter T_Death')
            else:
                ax.set_title(f'Denisty Plot of parameter {param}')
            ax.set_ylabel('Density')
            ax.legend()
    return


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
    fig, axs = plt.subplots(nrows=round(len(params)/2), ncols=2,
                            figsize=(18, 6*(len(params)/2)))
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
        for run, run_dict in model_dict.items():
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

def plot_all_buckets(predictions_dict, which_buckets=[], compare='model', param_method='ensemble_combined', model_types=None):
    extra_cols = ['date', 'active', 'total', 'recovered', 'deceased']
    buckets_values = {which_bucket:{} for which_bucket in which_buckets}
    layer2_vals = []
    for loc, loc_dict in predictions_dict.items():
        for model_name, model_dict in loc_dict.items():
            params = get_param_stats(model_dict, param_method).loc[['mean']].to_dict('records')[0]

            first_run_dict = model_dict[list(model_dict.keys())[0]]
            model_type = model_types[model_name]
            solver = eval(model_type)(**params, **first_run_dict['default_params'])
            total_days = (list(first_run_dict['df_prediction']['date'])[-1] - list(first_run_dict['df_prediction']['date'])[0]).days
            df_prediction = solver.predict(total_days=total_days)
            cols = df_prediction.columns.to_list()
            needed_cols = [col for col in cols if col not in extra_cols]
            df_prediction['N'] = df_prediction[needed_cols].sum(axis=1)

            for bucket in which_buckets:
                if bucket not in df_prediction.columns:
                    continue
                if compare=='model':
                    if loc not in buckets_values[bucket]:
                        buckets_values[bucket][loc] = {}
                    if model_name not in layer2_vals :
                        layer2_vals.append(model_name)
                    buckets_values[bucket][loc][model_name] = df_prediction[['date',bucket]]
                elif compare=='location':
                    if model_name not in buckets_values[bucket]:
                        buckets_values[bucket][model_name] = {}
                    if loc not in layer2_vals :
                        layer2_vals.append(loc)
                    buckets_values[bucket][model_name][loc] = df_prediction[['date',bucket]]

    # upper limit of n_subplots
    if compare == 'model' :
        n_subplots = len(which_buckets)*len(predictions_dict)
    elif compare == 'location':
        n_subplots = len(which_buckets)*len(list(predictions_dict.values())[0])
    ncols = 3
    nrows = math.ceil(n_subplots/ncols)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, 
                            figsize=(18, 8*nrows))
    fig.suptitle('Buckets')

    colors = "bgrcmy"
    color_map = {}
    for i,layer2_val in enumerate(layer2_vals):
        color_map[layer2_val] = colors[i]
    
    ax_counter = 0
    for bucket, bucket_dict in buckets_values.items():
        for layer_1, layer_1_dict in bucket_dict.items():
            ax = axs.flat[ax_counter]
            for k, layer_2 in enumerate(layer_1_dict):
                ax.plot(layer_1_dict[layer_2]['date'], layer_1_dict[layer_2][bucket], color=color_map[layer_2], label=layer_2)
            plt.sca(ax)
            plt.ylabel(bucket)
            plt.xticks(rotation=45)
            plt.legend(loc='best')
            plt.title(layer_1)
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
    fig, axs = plt.subplots(nrows=round(len(params)/2), ncols=2,
                            figsize=(18, 6*round((len(params)/2))))
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


def plot_heatmap_distribution_sigmas(mean_var_dict, stat_measure='mean', cmap='Reds', figsize=(10, 16)):
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

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df_sigma_mu.values.astype(float), annot=True, cmap=cmap, ax=ax,
                xticklabels=[x[0] for x in df_sigma_mu.columns], 
                yticklabels=[f'{x[0]}, {x[1]}' for x in df_sigma_mu.index])
    ax.set_title(f'Heatmap of sigma/mu values for all the {stat_measure}s calculated across all the identical runs')

    return fig, df_comparison

def plot_all_losses(predictions_dict, which_losses=['train', 'val'], method='best_loss_nora', weighting='exp', 
                    which_compartments=['total', 'active', 'recovered']):
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
    all_compartments = []
    # TODO: handle when which_compartments is None
    for compartments in which_compartments:
        for which_loss in which_losses:
            all_compartments += [compartment for compartment in compartments if compartment not in all_compartments]
    all_compartments.append('agg')
    loss_wise_stats = {}
    for which_loss in which_losses:
        loss_wise_stats[which_loss] = {compartment:{} for compartment in all_compartments}
    for loc, loc_dict in predictions_dict.items():
        for model, model_dict in loc_dict.items():
            for which_loss in which_losses:
                loss_values_stats = get_loss_stats(model_dict['m1'], which_loss=which_loss)
                # import pdb; pdb.set_trace()
                for compartment in loss_values_stats.columns:
                    if model not in loss_wise_stats[which_loss][compartment]:
                        loss_wise_stats[which_loss][compartment][model] = {'mean':{}, 'std':{}}
                    loss_wise_stats[which_loss][compartment][model]['mean'][loc] = loss_values_stats[compartment]['mean']
                    loss_wise_stats[which_loss][compartment][model]['std'][loc] = loss_values_stats[compartment]['std']

    n_subplots = len(all_compartments)*len(which_losses)
    ncols = 5
    nrows = math.ceil(n_subplots/ncols)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, 
                            figsize=(18, 8*nrows))
    colors = "grmybc"
    color_map = {}
    for i,model in enumerate(list( list( predictions_dict.values() )[0].keys() )):
        color_map[model] = colors[i]

    bar_width = (1-0.4)/len(which_compartments)
    ax_counter=0
    for which_loss in which_losses:
        for compartment in all_compartments:
            ax = axs.flat[ax_counter]
            compartment_values = loss_wise_stats[which_loss][compartment]
            mean_vals, std_vals = {},{}
            for m,model in enumerate(compartment_values.keys()):
                mean_vals[model] = compartment_values[model]['mean']
                std_vals[model] = compartment_values[model]['std']
                pos = [m*bar_width+n for n in range(len(mean_vals[model]))]
                ax.bar(pos, mean_vals[model].values(), width=bar_width, color=color_map[model], align='center', alpha=0.5, label=model)
                ax.errorbar(pos, mean_vals[model].values(), yerr=std_vals[model].values(), fmt='o', color='k')
            plt.sca(ax)
            plt.title(which_loss)
            plt.xlabel(compartment)
            xtick_vals = mean_vals[model].keys()
            plt.xticks(range(len(xtick_vals)), xtick_vals, rotation=45)
            plt.legend(loc='best')
            ax_counter += 1
    for i in range(ax_counter,nrows*ncols):
        fig.delaxes(axs.flat[i])
    plt.show()


def def comp_bar(PD, loss_type):
    which_compartments = ['total', 'active', 'recovered', 'deceased', 'agg']
    title_comp = ['Confirmed', 'Active', 'Recovered', 'Deceased', 'Aggregate']
    df_compiled = {"MCMC": [], "BO": []}
    for run, run_dict in PD.items():
        for model, model_dict in run_dict.items():
            if loss_type in ['train', 'test']:
                df = model_dict['m1']['df_loss'][loss_type]
                df['agg'] = df.mean()
                df_compiled[model].append(df)
            else:
                df = model_dict['ensemble_mean_forecast']['df_loss']
                df['agg'] = np.mean(list(df.values()))
                df2 = {comp: df[comp] for comp in which_compartments}
                df_compiled[model].append(df2)
    stats = {}
    n = len(df_compiled["MCMC"])
    stats['MCMC'] = (pd.DataFrame(df_compiled['MCMC']
                                  ).describe()).loc[['mean', 'std']]
    stats['BO'] = (pd.DataFrame(df_compiled['BO']).describe()
                   ).loc[['mean', 'std']]
    barWidth = .4
    bars1 = stats['MCMC'].loc[['mean']].values[0]
    bars2 = stats['BO'].loc[['mean']].values[0]
    yer1 = stats['MCMC'].loc[['std']].values[0]/np.sqrt(n)
    yer2 = stats['BO'].loc[['std']].values[0]/np.sqrt(n)
    yticks = np.arange(len(bars1))
    r1 = yticks - (barWidth/2)
    r2 = yticks + (barWidth/2)
    plt.barh(r1, width=bars1, height=barWidth, color='tab:blue', edgecolor='tab:blue', xerr=yer1,
             capsize=0, label='MCMC', alpha=0.5, linewidth=0, error_kw={"elinewidth": 2}, ecolor='black')
    plt.barh(r2, width=bars2, height=barWidth, color='tab:orange', edgecolor='tab:orange', xerr=yer2,
             capsize=0, label='ABMA', alpha=0.5, linewidth=0, error_kw={"elinewidth": 2}, ecolor='black')
    plt.yticks(yticks, labels=title_comp)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.25)
    plt.xlabel('MAPE loss (\%)')
    # if(loss_type == 'train'):
    #     ax.text(.025,4.25,'\\textbf{(a)}',fontweight='bold')
    # elif(loss_type == 'test'):
    #     ax.text(.025,30,'\\textbf{(b)}',fontweight='bold')
    # else:
    #     ax.text(.025,17.3,'\\textbf{(c)}',fontweight='bold')
    plt.legend()(PD, loss_type):
    which_compartments = ['total', 'active', 'recovered', 'deceased','agg']
    title_comp = ['Confirmed', 'Active', 'Recovered', 'Deceased','Aggregate']
    df_compiled = {"MCMC":[],"BO":[]}
    for run,run_dict in PD.items():
        for model,model_dict in run_dict.items():
            if loss_type in ['train','test']:
                df = model_dict['m1']['df_loss'][loss_type]
                df['agg'] = df.mean()
                df_compiled[model].append(df)
            else:
                df = model_dict['ensemble_mean_forecast']['df_loss']
                df['agg'] = np.mean(list(df.values()))
                df2 = {comp:df[comp] for comp in which_compartments}
                df_compiled[model].append(df2)
    stats = {}
    n = len(df_compiled["MCMC"])
    stats['MCMC'] = (pd.DataFrame(df_compiled['MCMC']).describe()).loc[['mean','std']]
    stats['BO'] = (pd.DataFrame(df_compiled['BO']).describe()).loc[['mean','std']]
    barWidth = .4
    bars1 = stats['MCMC'].loc[['mean']].values[0]
    bars2 = stats['BO'].loc[['mean']].values[0]
    yer1 = stats['MCMC'].loc[['std']].values[0]/np.sqrt(n)
    yer2 = stats['BO'].loc[['std']].values[0]/np.sqrt(n)
    yticks = np.arange(len(bars1))
    r1 = yticks - (barWidth/2)
    r2 = yticks + (barWidth/2)
    plt.barh(r1, width=bars1  ,height= barWidth, color = 'tab:blue', edgecolor = 'tab:blue', xerr=yer1, capsize=0, label='MCMC',alpha = 0.5,linewidth=0,error_kw = {"elinewidth":2},ecolor = 'black')
    plt.barh(r2, width= bars2  ,height= barWidth, color = 'tab:orange', edgecolor = 'tab:orange', xerr=yer2, capsize=0, label='ABMA',alpha = 0.5,linewidth=0,error_kw = {"elinewidth":2},ecolor = 'black')
    plt.yticks(yticks,labels = title_comp)
    plt.gca().invert_yaxis()
    plt.grid(axis='x',alpha =  0.25)
    plt.xlabel('MAPE loss (\%)')
    # if(loss_type == 'train'):
    #     ax.text(.025,4.25,'\\textbf{(a)}',fontweight='bold')
    # elif(loss_type == 'test'):
    #     ax.text(.025,30,'\\textbf{(b)}',fontweight='bold')
    # else:
    #     ax.text(.025,17.3,'\\textbf{(c)}',fontweight='bold')
    plt.legend()

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
    all_params = []
    model_params = {}
    loc = list(predictions_dict.keys())[0]
    for model_name, model_dict in predictions_dict[loc].items():
        model_params[model_name] = list(model_dict['m0']['best_params'].keys())
    
    # TODO: change to set
    for _, params in model_params.items():
        all_params += [param for param in params if param not in all_params]

    param_wise_stats = { param:{} for param in all_params }
    for loc, loc_dict in predictions_dict.items():
        for model, model_dict in loc_dict.items():
            param_values_stats = get_param_stats(model_dict, method, weighting)
            for param in param_values_stats.columns:
                if model not in param_wise_stats[param]:
                    param_wise_stats[param][model] = {'mean':{},'std':{}}
                param_wise_stats[param][model]['mean'][loc] = param_values_stats[param]['mean']
                param_wise_stats[param][model]['std'][loc] = param_values_stats[param]['std']

    n_subplots = len(all_params)
    ncols = 5
    nrows = math.ceil(n_subplots/ncols)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, 
                            figsize=(18, 8*nrows))
    colors = "bgrcmy"
    color_map = {}
    for i,model in enumerate(list(model_params.keys())):
        color_map[model] = colors[i]
    
    bar_width = (1-0.5)/len(model_params)
    ax_counter=0
    for param in all_params:
        ax = axs.flat[ax_counter]
        param_values = param_wise_stats[param]
        mean_vals, std_vals = {},{}
        for k,model in enumerate(param_values.keys()):
            mean_vals[model] = param_values[model]['mean']
            std_vals[model] = param_values[model]['std']
            pos = [k*bar_width+j for j in range(len(mean_vals[model]))]
            ax.bar(pos, mean_vals[model].values(), width=bar_width, color=color_map[model], align='center', alpha=0.5, label=model)
            ax.errorbar(pos, mean_vals[model].values(), yerr=std_vals[model].values(), fmt='o', color='k')
        plt.sca(ax)
        plt.xlabel(param)
        xtick_vals = mean_vals[model].keys()
        plt.xticks(range(len(mean_vals[model].keys())), mean_vals[model].keys(), rotation=45)
        plt.legend(loc='best')
        ax_counter += 1
    for i in range(ax_counter,nrows*ncols):
        fig.delaxes(axs.flat[i])
    plt.show()
