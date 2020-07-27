import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import copy

from functools import reduce

from utils.enums.columns import *
from viz.utils import axis_formatter

def plot_fit(df_prediction, df_train, df_val, df_district, train_period, state, district,
             which_compartments=['hospitalised', 'total_infected'], description='', savepath=None):
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
        which_compartments {list} -- Which buckets to plot (default: {['hospitalised', 'total_infected']})
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
    train_period = predictions_dict[which_fit]['run_params']['train_period']

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
