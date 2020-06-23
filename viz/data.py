import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from utils.enums.columns import *
from viz.utils import setup_plt

def plot_smoothing(orig_df_district, new_df_district, state, district,
                   which_compartments=['hospitalised', 'total_infected', 'recovered', 'deceased'], description='Smoothing'):
    """Helper function for creating plots for the smoothing

    Arguments:
        orig_df_district {pd.DataFrame} -- unsmoothed data
        new_df_district {pd.DataFrame} -- smoothed data
        train_period {int} -- Length of train period
        state {str} -- Name of state
        district {str} -- Name of district

    Keyword Arguments:
        which_compartments {list} -- Which buckets to plot (default: {['hospitalised', 'total_infected', 'recovered', 'deceased']})
        description {str} -- Additional description for the plots (if any) (default: {''})

    Returns:
        ax -- Matplotlib ax object
    """
    # Create plots
    fig, ax = plt.subplots(figsize=(12, 12))

    for compartment in compartments['base']:
        if compartment.name in which_compartments:
                ax.plot(orig_df_district[compartments['date'][0].name], orig_df_district[compartment.name],
                        '-o', color=compartment.color, label='{} (Observed)'.format(compartment.label))
                ax.plot(new_df_district[compartments['date'][0].name], new_df_district[compartment.name],
                        '-', color=compartment.color, label='{} (Smoothed)'.format(compartment.label))

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.ylabel('No of People')
    plt.xlabel('Time')
    plt.xticks(rotation=45, horizontalalignment='right')
    plt.legend()
    plt.title('{} - ({} {})'.format(description, state, district))
    plt.grid()

    return ax

def plot(x, y, title, yaxis_name=None, log=False, scatter=False, savepath=None):
    plt.title(title)
    setup_plt(yaxis_name)
    yscale = 'log' if log else "linear"
    plt.yscale(yscale)

    # plot error
    if scatter:
        plt.scatter(x,y,c='dodgerblue', marker='+')
    else:
        plt.plot(x, y, ls='-', c='crimson')
    if savepath is not None:
        plt.savefig(savepath)
    return