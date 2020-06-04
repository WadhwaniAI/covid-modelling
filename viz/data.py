import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


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

    if 'total_infected' in which_compartments:
        ax.plot(orig_df_district['date'], orig_df_district['total_infected'],
                '-o', color='C0', label='Confirmed Cases (Observed)')
        ax.plot(new_df_district['date'], new_df_district['total_infected'],
                '-', color='C0', label='Confirmed Cases (Smoothed)')
    if 'hospitalised' in which_compartments:
        ax.plot(orig_df_district['date'], orig_df_district['hospitalised'],
                '-o', color='orange', label='Active Cases (Observed)')
        ax.plot(new_df_district['date'], new_df_district['hospitalised'],
                '-', color='orange', label='Active Cases (Smoothed)')
    if 'recovered' in which_compartments:
        ax.plot(orig_df_district['date'], orig_df_district['recovered'],
                '-o', color='green', label='Recovered Cases (Observed)')
        ax.plot(new_df_district['date'], new_df_district['recovered'],
                '-', color='green', label='Recovered Cases (SmoothedA)')
    if 'deceased' in which_compartments:
        ax.plot(orig_df_district['date'], orig_df_district['deceased'],
                '-o', color='red', label='Deceased Cases (Observed)')
        ax.plot(new_df_district['date'], new_df_district['deceased'],
                '-', color='red', label='Deceased Cases (Smoothed)')

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

