import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

def plot_fit(df_prediction, df_train, df_val, df_train_nora, df_val_nora, train_period, state, district,
                 which_compartments=['hospitalised', 'total_infected'], description=''):
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
    fig, ax = plt.subplots(figsize=(12, 12))
    if isinstance(df_val, pd.DataFrame):
        df_true_plotting_rolling = pd.concat(
            [df_train, df_val], ignore_index=True)
        df_true_plotting = pd.concat(
            [df_train_nora, df_val_nora], ignore_index=True)
    else:
        df_true_plotting_rolling = df_train
        df_true_plotting = df_train_nora
    df_predicted_plotting = df_prediction.loc[df_prediction['date'].isin(
        df_true_plotting['date']), ['date', 'hospitalised', 'total_infected', 'deceased', 'recovered']]

    if 'total_infected' in which_compartments:
        ax.plot(df_true_plotting['date'], df_true_plotting['total_infected'],
                '-o', color='C0', label='Confirmed Cases (Observed)')
        ax.plot(df_true_plotting_rolling['date'], df_true_plotting_rolling['total_infected'],
                '-', color='C0', label='Confirmed Cases (Obs RA)')
        ax.plot(df_predicted_plotting['date'], df_predicted_plotting['total_infected'],
                '-.', color='C0', label='Confirmed Cases (Predicted)')
    if 'hospitalised' in which_compartments:
        ax.plot(df_true_plotting['date'], df_true_plotting['hospitalised'],
                '-o', color='orange', label='Active Cases (Observed)')
        ax.plot(df_true_plotting_rolling['date'], df_true_plotting_rolling['hospitalised'],
                '-', color='orange', label='Active Cases (Obs RA)')
        ax.plot(df_predicted_plotting['date'], df_predicted_plotting['hospitalised'],
                '-.', color='orange', label='Active Cases (Predicted)')
    if 'recovered' in which_compartments:
        ax.plot(df_true_plotting['date'], df_true_plotting['recovered'],
                '-o', color='green', label='Recovered Cases (Observed)')
        ax.plot(df_true_plotting_rolling['date'], df_true_plotting_rolling['recovered'],
                '-', color='green', label='Recovered Cases (Obs RA)')
        ax.plot(df_predicted_plotting['date'], df_predicted_plotting['recovered'],
                '-.', color='green', label='Recovered Cases (Predicted)')
    if 'deceased' in which_compartments:
        ax.plot(df_true_plotting['date'], df_true_plotting['deceased'],
                '-o', color='red', label='Deceased Cases (Observed)')
        ax.plot(df_true_plotting_rolling['date'], df_true_plotting_rolling['deceased'],
                '-', color='red', label='Deceased Cases (Obs RA)')
        ax.plot(df_predicted_plotting['date'], df_predicted_plotting['deceased'],
                '-.', color='red', label='Deceased Cases (Predicted)')

    ax.plot([df_train.iloc[-train_period, :]['date'], df_train.iloc[-train_period, :]['date']],
            [min(df_train['deceased']), max(df_train['total_infected'])], '--', color='brown', label='Train starts')
    if isinstance(df_val, pd.DataFrame):
        ax.plot([df_val.iloc[0, :]['date'], df_val.iloc[0, :]['date']],
                [min(df_val['deceased']), max(df_val['total_infected'])], '--', color='black', label='Val starts')

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
