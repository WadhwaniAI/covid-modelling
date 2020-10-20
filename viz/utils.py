import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
from matplotlib.lines import Line2D


def axis_formatter(ax, legend_elements=None, custom_legend=False, log_scale=False):
    """Helper function for formatting axis

    Arguments:
        ax -- Matplotlib ax object
    """
    
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.set_ylabel('No of People')
    ax.set_xlabel('Time')
    ax.tick_params('x', labelrotation=45)
    ax.grid()
    if log_scale:
        ax.set_yscale('log')
    if custom_legend:
        legend_elements += [
            Line2D([0], [0], ls='-', marker='o', color='black', label='Observed'),
            Line2D([0], [0], ls = '-', color='black', label='Observed Roll Avg'), 
            Line2D([0], [0], ls='-.', color='black', label='Predicted'),
            Line2D([0], [0], ls=':', color='brown', label='Train starts'),
            Line2D([0], [0], ls=':', color='black', label='Val starts')
            ]
        ax.legend(handles=legend_elements)
    else:
        ax.legend()

def setup_plt(ycol, yscale='log'):
    sns.set()
    register_matplotlib_converters()
    plt.yscale(yscale)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    plt.xlabel("Date")
    plt.ylabel(ycol)
