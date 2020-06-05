import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pandas.plotting import register_matplotlib_converters

def setup_plt(ycol, yscale='log'):
    sns.set()
    register_matplotlib_converters()
    plt.yscale(yscale)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    plt.xlabel("Date")
    plt.ylabel(ycol)
