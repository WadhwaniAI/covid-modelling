import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np

from functools import reduce

from utils.enums.columns import *


def axis_formatter(ax, legend_elements, custom_legend=False):
    """Helper function for formatting axis

    Arguments:
        ax -- Matplotlib ax object
    """
    
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.set_ylabel('No of People')
    ax.set_xlabel('Time')
    ax.tick_params('x', labelrotation=45)
    legend_elements += [
        Line2D([0], [0], ls='-', marker='o', color='black', label='Observed'),
        Line2D([0], [0], ls = '-', color='black', label='Observed Roll Avg'), 
        Line2D([0], [0], ls='-.', color='black', label='Predicted'),
        Line2D([0], [0], ls=':', color='brown', label='Train starts'),
        Line2D([0], [0], ls=':', color='black', label='Train starts')
        ]
    if custom_legend:
        ax.legend(handles=legend_elements)
    else:
        ax.legend()
    ax.grid()


def plot_fit(df_prediction, df_train, df_val, df_train_nora, df_val_nora, train_period, state, district,
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
        df_true_plotting = pd.concat(
            [df_train_nora, df_val_nora], ignore_index=True)
    else:
        df_true_plotting_rolling = df_train
        df_true_plotting = df_train_nora
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
                ax.plot(df_true_plotting[compartments['date'][0].name], df_true_plotting[compartment.name],
                        '-o', color=compartment.color, label='{} (Observed)'.format(compartment.label))
                ax.plot(df_true_plotting_rolling[compartments['date'][0].name], df_true_plotting_rolling[compartment.name],
                        '-', color=compartment.color, label='{} (Obs RA)'.format(compartment.label))
                ax.plot(df_predicted_plotting[compartments['date'][0].name], df_predicted_plotting[compartment.name],
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
