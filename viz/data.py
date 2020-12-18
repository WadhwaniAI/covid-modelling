import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from utils.generic.enums.columns import *
from viz.utils import setup_plt, axis_formatter

def plot_smoothing(orig_df_district, new_df_district, location_description,
                   which_compartments=['active', 'total', 'recovered', 'deceased'], 
                   description='Smoothing'):
    """Helper function for creating plots for the smoothing

    Arguments:
        orig_df_district {pd.DataFrame} -- unsmoothed data
        new_df_district {pd.DataFrame} -- smoothed data
        train_period {int} -- Length of train period
        state {str} -- Name of state
        district {str} -- Name of district

    Keyword Arguments:
        which_compartments {list} -- Which buckets to plot (default: {['active', 'total', 'recovered', 'deceased']})
        description {str} -- Additional description for the plots (if any) (default: {''})

    Returns:
        ax -- Matplotlib ax object
    """
    # Create plots
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
        for compartment in compartments[key]:
            if compartment.name in comp_subset:
                ax.plot(orig_df_district[compartments['date'][0].name].to_numpy(), 
                        orig_df_district[compartment.name].to_numpy(),
                        '-o', color=compartment.color, label='{} (Observed)'.format(compartment.label))
                ax.plot(new_df_district[compartments['date'][0].name].to_numpy(), 
                        new_df_district[compartment.name].to_numpy(),
                        '-', color=compartment.color, label='{} (Smoothed)'.format(compartment.label))
        axis_formatter(ax)
        i += 1
    plt.tight_layout()
    return fig


def plot_data(df, which_compartments=['active', 'total', 'recovered', 'deceased']):
    fig, ax = plt.subplots(figsize=(12, 12))
    for comp in which_compartments:
        compartment = Columns.from_name(comp)
        ax.plot(df[compartments['date'][0].name].to_numpy(),
                df[compartment.name].to_numpy(),
                '-o', color=compartment.color, label='{}'.format(compartment.label))
    axis_formatter(ax)
    plt.tight_layout()
    return fig

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
