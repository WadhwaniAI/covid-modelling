import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
from matplotlib.lines import Line2D


def axis_formatter(ax, legend_elements=None, custom_legend=False, log_scale=False):
    """Helper function for formatting axis

    Arguments:
        ax -- Matplotlib ax object
    """
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=WE, interval=2))
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    # ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=WE,interval=2))
    # ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.set_ylabel('Case Count')
    ax.set_xlabel('Time')
    ax.tick_params('x')
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
        ax.legend(loc = [0.02,.78])


def add_inset_subplot_to_axes(ax, rect):
    """General purpose helper function for adding an inset subplot within a particular subplot

    Args:
        ax (mpl.Axes): The axes in which the inset plot is to be added
        rect (list): List providing details of the size of the subplot. of the form [x, y, w, h], 
        where x, y is the bottom left coordinate of the rectangle, and w and h are the width and 
        height of the rectangle respectively.

    Returns:
        mpl.Axes: A mpl Axes variable corresponding to the inset subplot.
    """
    fig = plt.gcf()
    box = ax.get_position() 
    inax_position = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    width = box.width * rect[2]
    height = box.height * rect[3]
    subax = fig.add_axes([infig_position[0], infig_position[1], 
                          width, height])
    x_labelsize = subax.get_xticklabels()[0].get_size() * (rect[2]**0.5)
    y_labelsize = subax.get_yticklabels()[0].get_size() * (rect[3]**0.5)
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax


def show_figure(fig):
    """Helper function for plotting figs not created from pyplot and loaded from pickle file.
    Create a dummy figure and uses its manager to display "fig"

    Args:
        fig (mpl.Figure): The figure you want to plot.
    """
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)

def setup_plt(ycol, yscale='log'):
    sns.set()
    register_matplotlib_converters()
    plt.yscale(yscale)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    plt.xlabel("Date")
    plt.ylabel(ycol)
